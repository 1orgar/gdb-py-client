# GDB Python Client (`gdb-client`)

[![PyPI version](https://img.shields.io/pypi/v/gdb-client.svg)](https://pypi.org/project/gdb-client/)
[![Python versions](https://img.shields.io/pypi/pyversions/gdb-client.svg)](https://pypi.org/project/gdb-client/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/1orgar/gdb-py-client/actions/workflows/ci.yml/badge.svg)](https://github.com/1orgar/gdb-py-client/actions/workflows/ci.yml)

Official high-performance Python client for **[GDB (Distributed HTAP In-Memory Graph Database)](https://github.com/1orgar/gdb)**.

Powered by **Polars** SIMD vectorized expressions and **Apache Arrow Flight** high-speed TCP streaming, `gdb-client` delivers ultra-high-throughput parallel scatter-ingest directly into in-memory partition stores of cluster nodes.

---

## ⚡ Key Features

- **Polars-Powered Scatter-Ingest**: Automatically partitions massive DataFrames using Polars SIMD expressions and streams partitions in parallel directly to primary replica nodes in the cluster hash ring.
- **Dedicated Arrow Flight Ingestion Ports (`:8860+`)**: Bypasses HTTP JSON serialization overhead, streaming binary Arrow `RecordBatch` payloads directly into GDB's in-memory `DeltaMemTable`.
- **Automatic Cluster Hash Ring Discovery**: Queries cluster coordinator topology (`/cluster`) to resolve ring tokens ($u \pmod N$), partition mapping, and per-node Arrow Flight endpoints.
- **Native Polars Query Results**: Runs openCypher / GQL queries and yields results directly as `polars.DataFrame` via Arrow Flight `do_get` with zero-copy deserialization.
- **Graceful Fallbacks**: Works out of the box with standard library fallbacks if optional dependencies are missing.

---

## 📦 Installation

Install from PyPI:

```bash
pip install gdb-client
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/1orgar/gdb-py-client.git
```

### Requirements

- Python >= 3.9
- `polars >= 1.0.0`
- `pyarrow >= 14.0.0`
- `requests >= 2.28.0`

---

## 🚀 Quickstart

```python
import polars as pl
from gdb_client import GdbClient

# 1. Connect to any cluster node (auto-discovers all ring peers and Flight ports)
client = GdbClient(seed_url="http://localhost:8847")

# 2. Ingest 1,000,000 vertices in parallel directly into target nodes
users_df = pl.DataFrame({
    "id": range(1, 1_000_001),
    "name": [f"User_{i}" for i in range(1, 1_000_001)],
    "age": [20 + (i % 50) for i in range(1, 1_000_001)],
})

res_v = client.scatter_ingest_vertices(users_df, label="User", id_col="id")
print(f"Vertices: {res_v['rows_ingested']} rows in {res_v['elapsed_seconds']}s "
      f"({res_v['rows_per_second']:,.0f} rows/s)")

# 3. Ingest edges in parallel (automatically hashed by source vertex)
edges_df = pl.DataFrame({
    "src": range(1, 500_001),
    "dst": range(500_001, 1_000_001),
})

res_e = client.scatter_ingest_edges(edges_df, edge_type="FOLLOWS")
print(f"Edges: {res_e['rows_ingested']} rows in {res_e['elapsed_seconds']}s "
      f"({res_e['rows_per_second']:,.0f} rows/s)")

# 4. Trigger background compaction into immutable Chunked-CSR
client.compact()

# 5. Query graph using openCypher directly into a Polars DataFrame
df = client.query("MATCH (a:User)-[:FOLLOWS]->(b:User) RETURN a.name, b.name LIMIT 10;")
print(df)

# 6. Close Flight TCP connections
client.close()
```

---

## 🏗 Architecture: How Scatter-Ingest Works

In a traditional database setup, client libraries send bulk requests to a single coordinator or load balancer, which must unpack the data, hash every key, and re-transmit records over the network to the correct storage nodes.

`gdb-client` moves partition routing to the client side:

```
                  ┌─────────────────────────────────────────┐
                  │           polars.DataFrame              │
                  │       (10,000,000 Rows in RAM)          │
                  └────────────────────┬────────────────────┘
                                       │
                      Polars SIMD Hash: (col % TotalNodes)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   [Partition Chunk 0]        [Partition Chunk 1]        [Partition Chunk 2]
            │                          │                          │
   PyArrow RecordBatch        PyArrow RecordBatch        PyArrow RecordBatch
            │                          │                          │
  Arrow Flight do_put        Arrow Flight do_put        Arrow Flight do_put
   (TCP Port :8860)           (TCP Port :8861)           (TCP Port :8862)
            ▼                          ▼                          ▼
   ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
   │  GDB Node #1    │        │  GDB Node #2    │        │  GDB Node #3    │
   │  DeltaMemTable  │        │  DeltaMemTable  │        │  DeltaMemTable  │
   └─────────────────┘        └─────────────────┘        └─────────────────┘
```

1. **Client-Side Vectorized Hashing**: Polars computes target partitions for all rows simultaneously using native CPU SIMD instructions.
2. **Direct Multi-Node Streaming**: Partitions are streamed over concurrent TCP Flight channels directly to the responsible nodes' `--client-flight-port` (`:8860`, `:8861`, `:8862`, etc.).
3. **Zero Coordinator Bottleneck**: Cluster inter-node network bandwidth is preserved for MPP shuffle exchange and graph traversal.

---

## 📖 API Reference

### `GdbClient`

```python
GdbClient(seed_url: str = "http://localhost:8847", auth_token: Optional[str] = None)
```

- **`seed_url`**: HTTP address of any healthy node in the GDB cluster.
- **`auth_token`**: Optional Bearer token for clusters with authorization enabled.

#### Methods

| Method | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `scatter_ingest_vertices` | `df: pl.DataFrame, label: str = "User", id_col: str = "id", max_workers: int = 8` | `dict` | Partitions vertices and streams Arrow RecordBatches directly to target nodes. |
| `scatter_ingest_edges` | `df: pl.DataFrame, edge_type: str = "KNOWS", src_col: str = "src", dst_col: str = "dst", rank_col: Optional[str] = None, max_workers: int = 8` | `dict` | Partitions edges by source ID and streams Arrow RecordBatches to target nodes. |
| `query` | `cypher: str, node_id: Optional[int] = None` | `pl.DataFrame` | Executes an openCypher query and returns the tabular result as a Polars DataFrame. |
| `compact` | — | `dict` | Triggers asynchronous CSR compaction across cluster nodes. |
| `refresh_topology` | — | `None` | Queries `/cluster` and reconnects Flight clients to active nodes. |
| `close` | — | `None` | Closes open Arrow Flight TCP client connections. |

---

### Ingestion Performance Metrics

`scatter_ingest_vertices` and `scatter_ingest_edges` return a detailed performance dictionary:

```python
{
    "rows_ingested": 1000000,
    "elapsed_seconds": 0.4281,
    "rows_per_second": 2335902.8,
    "partitions": 3
}
```

---

## 🛠 Local Development & Building Wheel

To build the wheel and source distribution locally:

```bash
# Clone the repository
git clone https://github.com/1orgar/gdb-py-client.git
cd gdb-py-client

# Install build dependencies
pip install -e ".[dev]"

# Run tests
python -m unittest discover tests

# Build .whl and .tar.gz distributions
python -m build

# Validate package metadata with twine
twine check dist/*
```

Built artifacts will be located in `dist/`:
- `dist/gdb_client-0.3.0-py3-none-any.whl`
- `dist/gdb_client-0.3.0.tar.gz`

---

## 🚀 Publishing to PyPI

This project is configured with GitHub Actions for automated building and publishing:

1. **Automated CI (`.github/workflows/ci.yml`)**:
   Runs tests and verifies wheel builds on Python 3.9, 3.10, 3.11, 3.12, and 3.13 on every push and PR.

2. **Automated PyPI Release (`.github/workflows/publish.yml`)**:
   Triggered on git tags matching `v*` (e.g., `v0.3.0`) or manually via GitHub Actions `workflow_dispatch`.
   - Builds distribution wheels and source archives.
   - Creates a GitHub Release with assets attached.
   - Publishes to PyPI using [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC token authentication, no hardcoded API tokens required).

To release a new version manually using `twine`:

```bash
twine upload dist/*
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
