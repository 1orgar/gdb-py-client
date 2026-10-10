from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pyarrow as pa
    import pyarrow.flight as flight
except ImportError:
    pa = None
    flight = None

try:
    import requests
except ImportError:
    requests = None

try:
    import networkx as nx
except ImportError:
    nx = None

from .topology import ClusterTopology, NodeInfo


class QueryResult:
    """Represents the execution result of a GDB query or batch.

    Provides zero-copy and convenient export methods to Polars DataFrame,
    PyArrow Table, Pandas DataFrame, and NetworkX graph models.
    """

    def __init__(self, raw: Optional[Dict[str, Any]] = None, arrow_table: Optional[Any] = None):
        self.raw: Dict[str, Any] = raw or {}
        self.status: str = self.raw.get("status", "ok")
        self.message: str = self.raw.get("message", "")
        self.error: Optional[str] = self.raw.get("error")
        self.elapsed_us: int = self.raw.get("elapsed_us", 0)
        self.num_rows: int = self.raw.get("num_rows", 0)
        self.rows_affected: int = self.raw.get("rows_affected", 0)
        self.columns: List[str] = self.raw.get("columns", [])
        self.rows: List[List[Any]] = self.raw.get("rows", [])
        self.statements_executed: int = self.raw.get(
            "statements_executed", 1 if self.status == "ok" else 0
        )
        self._arrow_table = arrow_table

        if self._arrow_table is not None:
            if not self.columns:
                self.columns = list(self._arrow_table.column_names)
            if not self.num_rows:
                self.num_rows = self._arrow_table.num_rows

    @property
    def is_ok(self) -> bool:
        """Returns True if the query completed successfully."""
        return self.status == "ok"

    def to_polars(self):
        """Convert tabular result to a Polars DataFrame (zero-copy when available)."""
        if pl is None:
            raise ImportError("polars is not installed. Install via `pip install polars`.")
        if self._arrow_table is not None:
            return pl.from_arrow(self._arrow_table)
        if not self.columns:
            return pl.DataFrame()
        return pl.DataFrame(self.rows, schema=self.columns, orient="row")

    def to_arrow(self):
        """Convert tabular result to a PyArrow Table."""
        if pa is None:
            raise ImportError("pyarrow is not installed. Install via `pip install pyarrow`.")
        if self._arrow_table is not None:
            return self._arrow_table
        if not self.columns:
            return pa.Table.from_arrays([], names=[])
        pydict = {col: [row[i] for row in self.rows] for i, col in enumerate(self.columns)}
        return pa.Table.from_pydict(pydict)

    def to_pandas(self):
        """Convert tabular result to a Pandas DataFrame."""
        if self._arrow_table is not None:
            return self._arrow_table.to_pandas()
        if pl is not None:
            return self.to_polars().to_pandas()
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_networkx(self):
        """Convert graph query results (first 2 columns as source, target) to NetworkX DiGraph."""
        if nx is None:
            raise ImportError("networkx is not installed. Install via `pip install networkx`.")
        G = nx.DiGraph()
        if len(self.columns) < 2:
            return G
        if self.rows:
            for row in self.rows:
                u, v = row[0], row[1]
                props = {}
                for col_idx, col_name in enumerate(self.columns[2:], start=2):
                    props[col_name] = row[col_idx]
                G.add_edge(u, v, **props)
        elif self._arrow_table is not None and pl is not None:
            df = self.to_polars()
            cols = self.columns
            for row in df.iter_rows():
                u, v = row[0], row[1]
                props = {cols[i]: row[i] for i in range(2, len(cols))}
                G.add_edge(u, v, **props)
        return G

    def __len__(self) -> int:
        if self.rows:
            return len(self.rows)
        if self._arrow_table is not None:
            return self._arrow_table.num_rows
        return self.num_rows

    def __iter__(self):
        if not self.rows and self._arrow_table is not None and pl is not None:
            df = self.to_polars()
            return iter(df.iter_rows())
        return iter(self.rows)

    def __getitem__(self, index):
        if not self.rows and self._arrow_table is not None and pl is not None:
            df = self.to_polars()
            return df.row(index)
        return self.rows[index]

    def __repr__(self) -> str:
        if not self.is_ok:
            return f"<QueryResult status=ERROR error={self.error!r}>"
        return f"<QueryResult status=OK rows={len(self)} columns={self.columns} elapsed={self.elapsed_us / 1000.0:.2f}ms>"


class GdbClient:
    """Official High-Performance Python Client & SDK for GDB Graph Database.

    Features:
    - Polars-powered parallel scatter-ingest directly to target cluster nodes via Arrow Flight.
    - Zero-copy Flight DoGet queries and REST HTTP API queries with QueryResult abstraction.
    - Export query results directly to Polars, PyArrow, Pandas, and NetworkX.
    - Multi-statement DML / DDL execution and batching.
    - Administrative operations (compact, analyze, health, cluster, schema, resources, gpu).
    - Automatic cluster hash ring discovery and partition routing.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8847",
        seed_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        client_flight_port: int = 8860,
        timeout: float = 30.0,
    ):
        target = seed_url if seed_url is not None else endpoint
        self.endpoint = target.rstrip("/")
        self.seed_url = self.endpoint
        self.auth_token = auth_token
        self.client_flight_port = client_flight_port
        self.timeout = timeout
        self.session = requests.Session() if requests is not None else None
        if self.auth_token and self.session is not None:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

        self.topology = ClusterTopology(seed_url=self.seed_url)
        self._flight_clients: Dict[int, flight.FlightClient] = {}
        try:
            self._connect_flight_clients()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _connect_flight_clients(self):
        if flight is None:
            return
        self._flight_clients.clear()
        for node in self.topology.nodes:
            location = f"grpc://{node.host}:{node.client_flight_port}"
            try:
                client = flight.FlightClient(location)
                self._flight_clients[node.node_id] = client
            except Exception:
                pass

        if not self._flight_clients:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(self.endpoint)
                host = parsed.hostname or "127.0.0.1"
                location = f"grpc://{host}:{self.client_flight_port}"
                client = flight.FlightClient(location)
                self._flight_clients[1] = client
            except Exception:
                pass

    def refresh_topology(self):
        """Refreshes cluster nodes and recreates Flight connections."""
        self.topology.refresh()
        self._connect_flight_clients()

    def close(self):
        """Closes all active Flight client connections and HTTP session."""
        self._flight_clients.clear()
        if self.session is not None:
            self.session.close()

    def scatter_ingest_vertices(
        self,
        df: pl.DataFrame,
        label: str = "User",
        id_col: str = "id",
        max_workers: int = 8,
    ) -> Dict[str, Union[int, float]]:
        """Partitions vertex DataFrame in parallel using Polars expressions,
        and streams each partition directly to the primary replica node via Arrow Flight.
        """
        if pl is None or flight is None:
            raise ImportError("polars and pyarrow are required for scatter_ingest_vertices")

        start_time = time.perf_counter()
        total_nodes = self.topology.total_nodes
        if total_nodes == 0:
            raise RuntimeError("No active cluster nodes discovered in topology")

        # 1. Compute target node partition in Polars (vectorized zero-copy)
        if df[id_col].dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            part_expr = (pl.col(id_col) % total_nodes).cast(pl.UInt32).alias("_partition")
        else:
            part_expr = (pl.col(id_col).hash() % total_nodes).cast(pl.UInt32).alias("_partition")

        partitioned_df = df.with_columns(part_expr)
        partitions = partitioned_df.partition_by("_partition", as_dict=True)

        # 2. Dispatch partitions to target nodes concurrently
        total_rows = 0

        def _send_partition(part_idx_tuple, sub_df: pl.DataFrame) -> int:
            part_idx = part_idx_tuple[0] if isinstance(part_idx_tuple, tuple) else part_idx_tuple
            target_node = self.topology.nodes[part_idx % total_nodes]
            client = self._flight_clients.get(target_node.node_id)
            if client is None:
                client = flight.FlightClient(f"grpc://{target_node.host}:{target_node.client_flight_port}")

            clean_df = sub_df.drop("_partition")
            arrow_table = clean_df.to_arrow()

            descriptor_cmd = json.dumps({"type": "vertex", "label": label})
            descriptor = flight.FlightDescriptor.for_command(descriptor_cmd.encode("utf-8"))

            writer, reader = client.do_put(descriptor, arrow_table.schema)
            writer.write_table(arrow_table)
            writer.close()

            for _ in reader:
                pass

            return len(sub_df)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(partitions) or 1)) as executor:
            futures = [
                executor.submit(_send_partition, part_key, sub_df)
                for part_key, sub_df in partitions.items()
            ]
            for f in as_completed(futures):
                total_rows += f.result()

        elapsed = time.perf_counter() - start_time
        throughput = total_rows / elapsed if elapsed > 0 else 0

        return {
            "rows_ingested": total_rows,
            "elapsed_seconds": round(elapsed, 4),
            "rows_per_second": round(throughput, 2),
            "partitions": len(partitions),
        }

    def scatter_ingest_edges(
        self,
        df: pl.DataFrame,
        edge_type: str = "KNOWS",
        src_col: str = "src",
        dst_col: str = "dst",
        rank_col: Optional[str] = None,
        max_workers: int = 8,
    ) -> Dict[str, Union[int, float]]:
        """Partitions edge DataFrame by source vertex ID in parallel using Polars,
        and streams directly to primary replica nodes via Arrow Flight.
        """
        if pl is None or flight is None:
            raise ImportError("polars and pyarrow are required for scatter_ingest_edges")

        start_time = time.perf_counter()
        total_nodes = self.topology.total_nodes
        if total_nodes == 0:
            raise RuntimeError("No active cluster nodes discovered in topology")

        if df[src_col].dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            part_expr = (pl.col(src_col) % total_nodes).cast(pl.UInt32).alias("_partition")
        else:
            part_expr = (pl.col(src_col).hash() % total_nodes).cast(pl.UInt32).alias("_partition")

        partitioned_df = df.with_columns(part_expr)
        partitions = partitioned_df.partition_by("_partition", as_dict=True)

        total_rows = 0

        def _send_edge_partition(part_idx_tuple, sub_df: pl.DataFrame) -> int:
            part_idx = part_idx_tuple[0] if isinstance(part_idx_tuple, tuple) else part_idx_tuple
            target_node = self.topology.nodes[part_idx % total_nodes]
            client = self._flight_clients.get(target_node.node_id)
            if client is None:
                client = flight.FlightClient(f"grpc://{target_node.host}:{target_node.client_flight_port}")

            clean_df = sub_df.drop("_partition")
            arrow_table = clean_df.to_arrow()

            descriptor_cmd = json.dumps({"type": "edge", "edge_type": edge_type})
            descriptor = flight.FlightDescriptor.for_command(descriptor_cmd.encode("utf-8"))

            writer, reader = client.do_put(descriptor, arrow_table.schema)
            writer.write_table(arrow_table)
            writer.close()

            for _ in reader:
                pass

            return len(sub_df)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(partitions) or 1)) as executor:
            futures = [
                executor.submit(_send_edge_partition, part_key, sub_df)
                for part_key, sub_df in partitions.items()
            ]
            for f in as_completed(futures):
                total_rows += f.result()

        elapsed = time.perf_counter() - start_time
        throughput = total_rows / elapsed if elapsed > 0 else 0

        return {
            "rows_ingested": total_rows,
            "elapsed_seconds": round(elapsed, 4),
            "rows_per_second": round(throughput, 2),
            "partitions": len(partitions),
        }

    def query(
        self,
        cypher: str,
        as_df: bool = False,
        use_flight: bool = False,
        node_id: Optional[int] = None,
    ) -> Union[QueryResult, pl.DataFrame]:
        """Executes an openCypher query and returns the results.

        By default, returns a `QueryResult` object supporting `.to_polars()`,
        `.to_arrow()`, `.to_pandas()`, and `.to_networkx()`.
        If `as_df=True`, returns a `polars.DataFrame` directly.
        """
        if use_flight and flight is not None:
            target_client = None
            if node_id is not None and node_id in self._flight_clients:
                target_client = self._flight_clients[node_id]
            elif self._flight_clients:
                target_client = next(iter(self._flight_clients.values()))

            if target_client is not None:
                try:
                    ticket = flight.Ticket(cypher.encode("utf-8"))
                    reader = target_client.do_get(ticket)
                    table = reader.read_all()
                    result = QueryResult(
                        raw={
                            "status": "ok",
                            "columns": list(table.column_names),
                            "num_rows": table.num_rows,
                        },
                        arrow_table=table,
                    )
                    return result.to_polars() if as_df else result
                except Exception:
                    pass

        target_url = f"{self.endpoint}/query"
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.post(
                target_url,
                json={"query": cypher},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                target_url,
                data=json.dumps({"query": cypher}).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        res = QueryResult(data)
        if not res.is_ok:
            raise RuntimeError(f"GDB query failed: {res.error}")

        if as_df:
            return res.to_polars()
        return res

    def query_df(self, cypher: str, use_flight: bool = True) -> pl.DataFrame:
        """Executes query and returns results directly as a Polars DataFrame."""
        return self.query(cypher, as_df=True, use_flight=use_flight)

    def execute(self, stmt: str) -> QueryResult:
        """Alias for query()."""
        return self.query(stmt)

    def execute_script(self, script: str) -> QueryResult:
        """Execute a multi-statement Cypher script separated by semicolons."""
        return self.query(script)

    def batch(self, queries: List[str]) -> List[QueryResult]:
        """Execute a list of queries sequentially using /batch endpoint."""
        url = f"{self.endpoint}/batch"
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.post(
                url,
                json={"queries": queries},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps({"queries": queries}).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        return [QueryResult(r) for r in data.get("results", [])]

    def insert_vertices(
        self,
        label: str,
        data: Union[List[Dict[str, Any]], Any],
        batch_size: int = 500,
    ) -> int:
        """Batch insert vertices using multi-row VALUES syntax via REST API."""
        if pl is not None and isinstance(data, pl.DataFrame):
            dicts = data.to_dicts()
        elif hasattr(data, "to_dict"):
            dicts = data.to_dict(orient="records")
        else:
            dicts = list(data)

        if not dicts:
            return 0

        total_affected = 0
        props = list(dicts[0].keys())
        prop_list_str = ", ".join(props)

        for i in range(0, len(dicts), batch_size):
            chunk = dicts[i : i + batch_size]
            val_strs = []
            for item in chunk:
                vals = []
                for p in props:
                    val = item[p]
                    if val is None:
                        vals.append("null")
                    elif isinstance(val, str):
                        escaped = val.replace("'", "\\'")
                        vals.append(f"'{escaped}'")
                    elif isinstance(val, (list, tuple)):
                        vals.append(f"[{', '.join(str(float(x)) for x in val)}]")
                    elif isinstance(val, bool):
                        vals.append("true" if val else "false")
                    else:
                        vals.append(str(val))
                val_strs.append(f"({', '.join(vals)})")

            query = f"INSERT VERTEX {label} ({prop_list_str}) VALUES {', '.join(val_strs)};"
            res = self.query(query)
            total_affected += res.rows_affected or len(chunk)

        return total_affected

    def insert_edges(
        self,
        edge_type: str,
        edges: Union[List[Tuple[int, int]], Any],
        batch_size: int = 500,
    ) -> int:
        """Batch insert edges using multi-row syntax or batch statements via REST API."""
        if hasattr(edges, "to_numpy"):
            arr = edges.to_numpy()
            edge_list = [(int(row[0]), int(row[1])) for row in arr]
        else:
            edge_list = list(edges)

        if not edge_list:
            return 0

        total_affected = 0
        for i in range(0, len(edge_list), batch_size):
            chunk = edge_list[i : i + batch_size]
            statements = [
                f"INSERT EDGE {edge_type} FROM {u} TO {v};"
                for u, v in chunk
            ]
            batch_script = " ".join(statements)
            res = self.query(batch_script)
            total_affected += res.rows_affected or len(chunk)

        return total_affected

    def compact(self) -> Dict[str, Any]:
        """Trigger asynchronous background CSR compaction across the cluster."""
        target_url = f"{self.endpoint}/compact"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.post(target_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                target_url,
                data=b"{}",
                headers={"Content-Type": "application/json", **headers},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def analyze(self) -> QueryResult:
        """Compute cardinality and degree statistics for Cost-Based Optimizer (CBO)."""
        return self.query("ANALYZE GRAPH;")

    def health(self) -> Dict[str, Any]:
        """Get node and cluster health."""
        target_url = f"{self.endpoint}/health"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.get(target_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def cluster(self) -> Dict[str, Any]:
        """Get cluster hash ring topology and node replica mappings."""
        target_url = f"{self.endpoint}/cluster"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.get(target_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def schema(self) -> Dict[str, Any]:
        """Get registered vertex labels, edge types, and properties."""
        target_url = f"{self.endpoint}/schema"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.get(target_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def resources(self) -> Dict[str, Any]:
        """Get live memory, CSR edge count, and Delta MemTable statistics."""
        target_url = f"{self.endpoint}/resources"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.get(target_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def gpu(self) -> Dict[str, Any]:
        """Get GPU acceleration status, device ID, memory architecture, and kernels."""
        target_url = f"{self.endpoint}/gpu"
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.session is not None:
            resp = self.session.get(target_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
