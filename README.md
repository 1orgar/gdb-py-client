# GDB Python Client (`gdb-client`)

Official high-performance Python client for **GDB (Distributed HTAP In-Memory Graph Database)**.

## Key Features

- **Polars-Powered Scatter-Ingest**: Automatically partitions huge DataFrames using fast Polars SIMD expressions and streams each partition in parallel directly to target cluster nodes.
- **Dedicated Arrow Flight Ingest Ports**: Bypasses HTTP JSON overhead, writing directly into in-memory `DeltaMemTable` over TCP Arrow Flight (`--client-flight-port`).
- **Cluster Hash Ring Discovery**: Discovers live topology, seed nodes, replication factors, and partition tokens via `/cluster`.
- **Zero-Copy Flight Queries**: Executes Cypher/openCypher queries and streams results back as native `polars.DataFrame`.

## Installation

```bash
pip install polars pyarrow requests
pip install -e /Users/kirill/Documents/projects/gdb-py-client
```

## Quickstart

```python
import polars as pl
from gdb_client import GdbClient

# 1. Connect to coordinator node (discovers all cluster peer ports)
client = GdbClient(seed_url="http://localhost:8847")

# 2. Ingest 1,000,000 vertices in parallel directly into target nodes
users_df = pl.DataFrame({
    "id": range(1, 1_000_001),
    "name": [f"User_{i}" for i in range(1, 1_000_001)],
    "age": [20 + (i % 50) for i in range(1, 1_000_001)],
})

res = client.scatter_ingest_vertices(users_df, label="User", id_col="id")
print(f"Ingested {res['rows_ingested']} rows in {res['elapsed_seconds']}s ({res['rows_per_second']:.0f} rows/s)")

# 3. Ingest edges in parallel
edges_df = pl.DataFrame({
    "src": range(1, 500_001),
    "dst": range(500_001, 1_000_001),
})
client.scatter_ingest_edges(edges_df, edge_type="FOLLOWS")

# 4. Trigger compaction
client.compact()

# 5. Query graph with openCypher directly into Polars DataFrame
df = client.query("MATCH (a:User)-[:FOLLOWS]->(b:User) RETURN a.name, b.name LIMIT 10")
print(df)
```
