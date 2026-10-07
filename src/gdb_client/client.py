from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union
import json

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

from .topology import ClusterTopology, NodeInfo


class GdbClient:
    """High-performance Python client for GDB.

    Features:
    - Polars-powered parallel scatter-ingest directly to target cluster nodes via Arrow Flight.
    - Zero-copy Flight DoGet queries returning polars.DataFrame directly.
    - Automatic cluster hash ring discovery and partition routing.
    """

    def __init__(self, seed_url: str = "http://localhost:8847", auth_token: Optional[str] = None):
        self.seed_url = seed_url.rstrip("/")
        self.auth_token = auth_token
        self.topology = ClusterTopology(seed_url=self.seed_url)
        self._flight_clients: Dict[int, flight.FlightClient] = {}
        self._connect_flight_clients()

    def _connect_flight_clients(self):
        self._flight_clients.clear()
        for node in self.topology.nodes:
            location = f"grpc://{node.host}:{node.client_flight_port}"
            try:
                client = flight.FlightClient(location)
                self._flight_clients[node.node_id] = client
            except Exception as e:
                # Retry or log warning
                pass

    def refresh_topology(self):
        """Refreshes cluster nodes and recreates Flight connections."""
        self.topology.refresh()
        self._connect_flight_clients()

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

            # Attach ingest descriptor command
            descriptor_cmd = json.dumps({"type": "vertex", "label": label})
            descriptor = flight.FlightDescriptor.for_command(descriptor_cmd.encode("utf-8"))

            writer, reader = client.do_put(descriptor, arrow_table.schema)
            writer.write_table(arrow_table)
            writer.close()

            # Drain result
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

    def query(self, cypher: str, node_id: Optional[int] = None) -> pl.DataFrame:
        """Executes an openCypher query and returns the results directly as a Polars DataFrame.

        Prefers Arrow Flight DoGet for maximum throughput; falls back to HTTP REST API.
        """
        # Choose client node
        target_client = None
        if node_id is not None and node_id in self._flight_clients:
            target_client = self._flight_clients[node_id]
        elif self._flight_clients:
            target_client = next(iter(self._flight_clients.values()))

        # Attempt Flight DoGet first
        if target_client is not None:
            try:
                ticket = flight.Ticket(cypher.encode("utf-8"))
                reader = target_client.do_get(ticket)
                table = reader.read_all()
                return pl.from_arrow(table)
            except Exception:
                pass

        # Fallback to HTTP API
        target_url = f"{self.seed_url}/query"
        if requests is not None:
            resp = requests.post(target_url, json={"query": cypher}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                target_url,
                data=json.dumps({"query": cypher}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        columns = data.get("columns", [])
        rows = data.get("rows", [])

        if not columns or not rows:
            return pl.DataFrame() if pl is not None else {}

        if pl is not None:
            col_dict = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            return pl.DataFrame(col_dict)
        return {"columns": columns, "rows": rows}

    def compact(self) -> dict:
        """Triggers asynchronous compaction across the cluster."""
        target_url = f"{self.seed_url}/compact"
        if requests is not None:
            resp = requests.post(target_url, timeout=10)
            return resp.json()
        else:
            import urllib.request
            req = urllib.request.Request(target_url, data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

    def close(self):
        """Closes all active Flight client connections."""
        self._flight_clients.clear()
