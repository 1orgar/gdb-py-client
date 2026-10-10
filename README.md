# GDB Python Client (`gdb-client`)

[![PyPI version](https://img.shields.io/pypi/v/gdb-client.svg)](https://pypi.org/project/gdb-client/)
[![Python versions](https://img.shields.io/pypi/pyversions/gdb-client.svg)](https://pypi.org/project/gdb-client/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/1orgar/gdb-py-client/actions/workflows/ci.yml/badge.svg)](https://github.com/1orgar/gdb-py-client/actions/workflows/ci.yml)

Official high-performance Python client and SDK for **[GDB (Distributed HTAP In-Memory Graph Database)](https://github.com/1orgar/gdb)**.

Combines high-level graph database operations (openCypher/GQL queries, multi-statement DDL/DML, vector similarity search, CBO optimization) with **Polars** SIMD vectorized expressions and **Apache Arrow Flight** high-speed streaming for ultra-high-throughput parallel scatter-ingest directly into cluster nodes.

---

## ⚡ Key Features

- **Unified Query Interface**: Execute openCypher, GQL, DDL, DML, and graph algorithms (`CALL algo.*`).
- **Rich `QueryResult` Abstraction**: Zero-copy export to **Polars** (`res.to_polars()`), **PyArrow** (`res.to_arrow()`), **Pandas** (`res.to_pandas()`), and **NetworkX** (`res.to_networkx()`).
- **Polars-Powered Scatter-Ingest**: Automatically partitions massive DataFrames using Polars SIMD expressions and streams partitions in parallel directly to primary replica nodes in the cluster hash ring via Arrow Flight.
- **Dedicated Arrow Flight Streaming (`:8860+`)**: Bypasses HTTP JSON serialization overhead, streaming binary Arrow `RecordBatch` payloads directly into GDB's in-memory `DeltaMemTable`.
- **Automatic Cluster Hash Ring Discovery**: Queries cluster coordinator topology (`/cluster`) to resolve ring tokens ($u \pmod N$), partition mapping, and per-node Arrow Flight endpoints.
- **Vector Search & Embeddings**: Native support for `VECTOR(dim)` types, cosine/euclidean similarity, and Node2Vec embeddings.
- **Cluster Diagnostics & CBO**: Built-in methods for `analyze()` (CBO statistics), `compact()`, `health()`, `cluster()`, `schema()`, `resources()`, and `gpu()`.

---

## 📦 Installation

Install from PyPI:

```bash
pip install gdb-client
```

Or install with all optional analytics dependencies (Polars, PyArrow, NetworkX, Pandas):

```bash
pip install "gdb-client[all]"
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/1orgar/gdb-py-client.git
```

### Requirements

- Python >= 3.9
- `requests >= 2.28.0`
- `pyarrow >= 14.0.0`
- `polars >= 0.20.0`
- Optional: `networkx >= 3.0`, `pandas >= 1.5.0`

---

## 🚀 Quickstart

### 1. Connecting to GDB

```python
from gdb_client import GdbClient

# Connect to any node in the cluster (auto-discovers topology and Arrow Flight ports)
client = GdbClient(endpoint="http://localhost:8847")
```

### 2. Multi-Statement Schema Setup & Vector Search

```python
# Create schema with vector embeddings
client.execute_script("""
    CREATE VERTEX Article (title STRING, emb VECTOR(3));
    INSERT VERTEX Article (id, title, emb) VALUES (1, 'Graph ML', [1.0, 0.0, 0.0]);
    INSERT VERTEX Article (id, title, emb) VALUES (2, 'Vector DB', [0.9, 0.1, 0.0]);
    INSERT VERTEX Article (id, title, emb) VALUES (3, 'Astronomy', [0.0, 0.0, 1.0]);
""")

# Run Cosine Vector Similarity Search
res = client.query("CALL vector.similaritySearch('Article', 'emb', [1.0, 0.0, 0.0], 5, 'cosine') YIELD vertex_id, score;")
print(res.to_polars())
```

### 3. High-Throughput Parallel Scatter-Ingest (Arrow Flight + Polars)

```python
import polars as pl

# Ingest 1,000,000 vertices in parallel directly into target cluster nodes
users_df = pl.DataFrame({
    "id": range(1, 1_000_001),
    "name": [f"User_{i}" for i in range(1, 1_000_001)],
    "age": [20 + (i % 50) for i in range(1, 1_000_001)],
})

res_v = client.scatter_ingest_vertices(users_df, label="User", id_col="id")
print(f"Vertices: {res_v['rows_ingested']} rows in {res_v['elapsed_seconds']}s "
      f"({res_v['rows_per_second']:,.0f} rows/s)")

# Ingest edges in parallel (automatically hashed by source vertex)
edges_df = pl.DataFrame({
    "src": range(1, 500_001),
    "dst": range(500_001, 1_000_001),
})

res_e = client.scatter_ingest_edges(edges_df, edge_type="FOLLOWS")
print(f"Edges: {res_e['rows_ingested']} rows in {res_e['elapsed_seconds']}s "
      f"({res_e['rows_per_second']:,.0f} rows/s)")

# Trigger CSR compaction
client.compact()
```

### 4. Graph Querying and NetworkX Export

```python
res = client.query("MATCH (a:User)-[:FOLLOWS]->(b:User) RETURN a.id, b.id LIMIT 100;")

# Export to Polars DataFrame
df = res.to_polars()
print(df)

# Export directly to NetworkX DiGraph
G = res.to_networkx()
print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
```

### 5. Cost-Based Optimizer (CBO) & Hardware Acceleration

```python
# Compute cardinality and degree statistics for Cost-Based Optimizer
res = client.analyze()
print("CBO:", res.message)

# Check GPU hardware acceleration status
gpu_info = client.gpu()
print("GPU Status:", gpu_info)
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
GdbClient(
    endpoint: str = "http://localhost:8847",
    seed_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    client_flight_port: int = 8860,
    timeout: float = 30.0,
)
```

#### Core Query & Ingestion Methods

| Method | Description |
| :--- | :--- |
| `query(cypher, as_df=False, use_flight=False)` | Executes Cypher query. Returns `QueryResult` (or `polars.DataFrame` if `as_df=True`). |
| `query_df(cypher)` | Executes query and returns results directly as a Polars DataFrame. |
| `execute(stmt)` | Alias for `query()`. |
| `execute_script(script)` | Executes semicolon-separated multi-statement Cypher script. |
| `batch(queries)` | Executes list of queries sequentially via `/batch` endpoint. |
| `insert_vertices(label, data, batch_size=500)` | Batch insert vertices via multi-row `INSERT VERTEX ... VALUES ...`. |
| `insert_edges(edge_type, edges, batch_size=500)` | Batch insert edges via multi-row statements. |
| `scatter_ingest_vertices(df, label="User", id_col="id")` | High-throughput parallel partition streaming via Arrow Flight. |
| `scatter_ingest_edges(df, edge_type="KNOWS", src_col="src")` | High-throughput parallel edge partition streaming via Arrow Flight. |

#### Administrative & Diagnostic Methods

| Method | Description |
| :--- | :--- |
| `analyze()` | Triggers `ANALYZE GRAPH;` computing CBO degree & cardinality statistics. |
| `compact()` | Triggers asynchronous CSR compaction across cluster nodes. |
| `health()` | Checks node and cluster health (`GET /health`). |
| `cluster()` | Fetches cluster hash ring topology and node replica mappings (`GET /cluster`). |
| `schema()` | Fetches registered vertex labels, edge types, and properties (`GET /schema`). |
| `resources()` | Fetches memory, CSR edge count, and Delta MemTable statistics (`GET /resources`). |
| `gpu()` | Fetches GPU acceleration telemetry, device ID, and kernels (`GET /gpu`). |
| `refresh_topology()` | Refreshes cluster nodes and recreates Flight connections. |
| `close()` | Closes open Flight connections and HTTP sessions. |

---

## 📄 License

Licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
