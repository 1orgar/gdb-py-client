import unittest
from unittest.mock import MagicMock, patch
from gdb_client import GdbClient, QueryResult
from gdb_client.topology import ClusterTopology, NodeInfo

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class TestQueryResult(unittest.TestCase):
    def test_ok_query_result(self):
        raw = {
            "status": "ok",
            "message": "Query executed",
            "columns": ["id", "name", "score"],
            "rows": [[1, "Alice", 95.5], [2, "Bob", 88.0]],
            "elapsed_us": 1200,
            "num_rows": 2,
            "rows_affected": 0,
        }
        res = QueryResult(raw)
        self.assertTrue(res.is_ok)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.elapsed_us, 1200)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], [1, "Alice", 95.5])
        self.assertEqual(res[1], [2, "Bob", 88.0])
        self.assertEqual(list(iter(res)), raw["rows"])
        self.assertIn("status=OK rows=2", repr(res))

    def test_error_query_result(self):
        raw = {
            "status": "error",
            "error": "Syntax error at line 1",
            "elapsed_us": 300,
        }
        res = QueryResult(raw)
        self.assertFalse(res.is_ok)
        self.assertEqual(res.error, "Syntax error at line 1")
        self.assertIn("status=ERROR", repr(res))

    @unittest.skipUnless(HAS_POLARS, "polars is required for this test")
    def test_to_polars(self):
        raw = {
            "status": "ok",
            "columns": ["u", "v"],
            "rows": [[1, 2], [3, 4]],
        }
        res = QueryResult(raw)
        df = res.to_polars()
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.columns, ["u", "v"])
        self.assertEqual(df["u"].to_list(), [1, 3])

    @unittest.skipUnless(HAS_PYARROW, "pyarrow is required for this test")
    def test_to_arrow(self):
        raw = {
            "status": "ok",
            "columns": ["u", "v"],
            "rows": [[1, 2], [3, 4]],
        }
        res = QueryResult(raw)
        tbl = res.to_arrow()
        self.assertEqual(tbl.num_rows, 2)
        self.assertEqual(tbl.num_columns, 2)
        self.assertEqual(tbl.column_names, ["u", "v"])

    @unittest.skipUnless(HAS_NETWORKX, "networkx is required for this test")
    def test_to_networkx(self):
        raw = {
            "status": "ok",
            "columns": ["src", "dst", "weight"],
            "rows": [[10, 20, 1.5], [20, 30, 2.5]],
        }
        res = QueryResult(raw)
        G = res.to_networkx()
        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 2)
        self.assertEqual(G[10][20]["weight"], 1.5)


class TestGdbClient(unittest.TestCase):
    def test_client_init_and_context(self):
        with patch.object(ClusterTopology, "refresh"):
            with GdbClient(endpoint="http://127.0.0.1:8847", auth_token="tok123") as client:
                self.assertEqual(client.endpoint, "http://127.0.0.1:8847")
                self.assertEqual(client.seed_url, "http://127.0.0.1:8847")
                self.assertEqual(client.auth_token, "tok123")

    def test_query_success(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "status": "ok",
                "columns": ["cnt"],
                "rows": [[42]],
                "elapsed_us": 500,
            }
            client.session.post = MagicMock(return_value=mock_resp)

            res = client.query("MATCH (n) RETURN count(n) AS cnt;")
            self.assertTrue(res.is_ok)
            self.assertEqual(res.rows, [[42]])
            self.assertEqual(res[0][0], 42)

    def test_query_failure(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "status": "error",
                "error": "Table not found",
            }
            client.session.post = MagicMock(return_value=mock_resp)

            with self.assertRaises(RuntimeError) as ctx:
                client.query("MATCH (x:Unknown) RETURN x;")
            self.assertIn("Table not found", str(ctx.exception))

    def test_insert_vertices_formatting(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            client.query = MagicMock(return_value=QueryResult({"status": "ok", "rows_affected": 2}))

            data = [
                {"id": 1, "name": "Alice", "score": 9.5, "emb": [0.1, 0.2]},
                {"id": 2, "name": "Bob", "score": 8.0, "emb": [0.3, 0.4]},
            ]
            count = client.insert_vertices("User", data, batch_size=10)
            self.assertEqual(count, 2)
            called_query = client.query.call_args[0][0]
            self.assertIn("INSERT VERTEX User", called_query)
            self.assertIn("(1, 'Alice', 9.5, [0.1, 0.2])", called_query)

    def test_insert_edges_formatting(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            client.query = MagicMock(return_value=QueryResult({"status": "ok", "rows_affected": 2}))

            edges = [(1, 2), (2, 3)]
            count = client.insert_edges("KNOWS", edges, batch_size=10)
            self.assertEqual(count, 2)
            called_query = client.query.call_args[0][0]
            self.assertIn("INSERT EDGE KNOWS FROM 1 TO 2;", called_query)
            self.assertIn("INSERT EDGE KNOWS FROM 2 TO 3;", called_query)


class TestTopologyAndRouting(unittest.TestCase):
    def test_topology_routing(self):
        topology = ClusterTopology.__new__(ClusterTopology)
        topology.nodes = [
            NodeInfo(node_id=1, http_url="http://127.0.0.1:8847", flight_port=8848, client_flight_port=8860),
            NodeInfo(node_id=2, http_url="http://127.0.0.1:8846", flight_port=8849, client_flight_port=8861),
            NodeInfo(node_id=3, http_url="http://127.0.0.1:8845", flight_port=8850, client_flight_port=8862),
        ]
        topology.replication_factor = 3
        topology.partitions = 8

        self.assertEqual(topology.total_nodes, 3)

        target_0 = topology.target_node_for_key(0)
        self.assertEqual(target_0.node_id, 1)
        self.assertEqual(target_0.client_flight_port, 8860)

        target_1 = topology.target_node_for_key(1)
        self.assertEqual(target_1.node_id, 2)
        self.assertEqual(target_1.client_flight_port, 8861)

        target_2 = topology.target_node_for_key(2)
        self.assertEqual(target_2.node_id, 3)
        self.assertEqual(target_2.client_flight_port, 8862)

        target_3 = topology.target_node_for_key(3)
        self.assertEqual(target_3.node_id, 1)

    @unittest.skipUnless(HAS_POLARS, "polars is required for this test")
    def test_polars_partitioning_logic(self):
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["A", "B", "C", "D", "E", "F"],
        })
        total_nodes = 3
        df_part = df.with_columns((pl.col("id") % total_nodes).cast(pl.UInt32).alias("_partition"))
        partitions = df_part.partition_by("_partition", as_dict=True)

        self.assertEqual(len(partitions), 3)
        self.assertEqual(sum(len(sub) for sub in partitions.values()), 6)


if __name__ == "__main__":
    unittest.main()
