import unittest
from unittest.mock import MagicMock, patch
from gdb_client import (
    CheckpointTuple,
    Document,
    GdbCheckpointSaver,
    GdbClient,
    GdbGraphQAChain,
    GdbGraphRAGToolkit,
    GdbGraphStore,
    GdbVectorStore,
    QueryResult,
)
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

    def test_node_info_address_resolution(self):
        # 1. Explicit remote client_flight_address
        node = NodeInfo.from_dict({
            "node_id": 1,
            "http_url": "http://10.0.0.1:8847",
            "flight_address": "10.0.0.1",
            "flight_port": 8848,
            "client_flight_address": "mpp.gdb.cluster.internal",
            "client_flight_port": 8860,
        })
        self.assertEqual(node.host, "mpp.gdb.cluster.internal")
        self.assertEqual(node.flight_address, "10.0.0.1")
        self.assertEqual(node.client_flight_address, "mpp.gdb.cluster.internal")

        # 2. Server returned loopback inside Docker, but client connected via remote seed_host
        node_docker = NodeInfo.from_dict({
            "node_id": 1,
            "http_url": "http://127.0.0.1:8847",
            "flight_address": "127.0.0.1",
            "flight_port": 8848,
            "client_flight_address": "127.0.0.1",
            "client_flight_port": 8860,
        }, seed_host="192.168.1.100")
        self.assertEqual(node_docker.host, "192.168.1.100")

        # 3. Default fallback from http_url
        node_default = NodeInfo.from_dict({
            "node_id": 2,
            "http_url": "http://10.0.0.2:8847",
        })
        self.assertEqual(node_default.host, "10.0.0.2")

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

    @unittest.skipUnless(HAS_POLARS, "polars is required for this test")
    def test_scatter_ingest_schema_validation_unknown_label(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            client.schema = MagicMock(return_value={
                "status": "ok",
                "vertices": [{"label": "User"}],
                "edges": [{"edge_type_name": "KNOWS"}],
            })
            df = pl.DataFrame({"id": [1, 2], "name": ["A", "B"]})
            with self.assertRaises(ValueError) as ctx:
                client.scatter_ingest_vertices(df, label="Post")
            self.assertIn("Vertex label 'Post' does not exist in graph schema", str(ctx.exception))
            self.assertIn("User", str(ctx.exception))

    @unittest.skipUnless(HAS_POLARS, "polars is required for this test")
    def test_scatter_ingest_schema_validation_unknown_edge(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            client.schema = MagicMock(return_value={
                "status": "ok",
                "vertices": [{"label": "User"}],
                "edges": [{"edge_type_name": "KNOWS"}],
            })
            df = pl.DataFrame({"src": [1, 2], "dst": [3, 4]})
            with self.assertRaises(ValueError) as ctx:
                client.scatter_ingest_edges(df, edge_type="LIKES")
            self.assertIn("Edge type 'LIKES' does not exist in graph schema", str(ctx.exception))
            self.assertIn("KNOWS", str(ctx.exception))

    def test_target_replica_nodes_for_key(self):
        topo = ClusterTopology.__new__(ClusterTopology)
        topo.nodes = [
            NodeInfo(node_id=1, http_url="http://node1:8847", flight_port=8848, client_flight_port=8860),
            NodeInfo(node_id=2, http_url="http://node2:8847", flight_port=8848, client_flight_port=8860),
            NodeInfo(node_id=3, http_url="http://node3:8847", flight_port=8848, client_flight_port=8860),
        ]
        # RF = 1 (pure sharding)
        topo.replication_factor = 1
        reps_rf1 = topo.target_replica_nodes_for_key(0)
        self.assertEqual(len(reps_rf1), 1)
        self.assertEqual(reps_rf1[0].node_id, 1)

        reps_rf1_key4 = topo.target_replica_nodes_for_key(4)
        self.assertEqual(len(reps_rf1_key4), 1)
        self.assertEqual(reps_rf1_key4[0].node_id, 2)  # 4 % 3 = 1 -> node_id 2

        # RF = 2
        topo.replication_factor = 2
        reps_rf2 = topo.target_replica_nodes_for_key(1)
        self.assertEqual(len(reps_rf2), 2)
        self.assertEqual([n.node_id for n in reps_rf2], [2, 3])

        # RF = 3 (full replication)
        topo.replication_factor = 3
        reps_rf3 = topo.target_replica_nodes_for_key(2)
        self.assertEqual(len(reps_rf3), 3)
        self.assertEqual([n.node_id for n in reps_rf3], [3, 1, 2])

    @unittest.skipUnless(HAS_POLARS, "polars is required for this test")
    def test_scatter_ingest_edges_rf1_and_rf2(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient()
            client.topology.nodes = [
                NodeInfo(node_id=1, http_url="http://node1:8847", flight_port=8848, client_flight_port=8860),
                NodeInfo(node_id=2, http_url="http://node2:8847", flight_port=8848, client_flight_port=8860),
            ]
            client.schema = MagicMock(return_value={
                "status": "ok",
                "edges": [{"edge_type_name": "CONNECTED"}],
            })

            mock_flight_1 = MagicMock()
            mock_flight_2 = MagicMock()
            mock_flight_1.do_put.return_value = (MagicMock(), MagicMock())
            mock_flight_2.do_put.return_value = (MagicMock(), MagicMock())
            client._flight_clients = {1: mock_flight_1, 2: mock_flight_2}

            # 1. Test RF=1 (Pure sharding: 0% duplication)
            client.topology.replication_factor = 1
            df = pl.DataFrame({"src": [1, 2], "dst": [2, 3]})
            res1 = client.scatter_ingest_edges(df, edge_type="CONNECTED")
            self.assertEqual(res1["rows_ingested"], 2)
            # Each partition sent to exactly 1 node (total 2 do_put calls across 2 nodes)
            total_calls_rf1 = mock_flight_1.do_put.call_count + mock_flight_2.do_put.call_count
            self.assertEqual(total_calls_rf1, 2)

            mock_flight_1.reset_mock()
            mock_flight_2.reset_mock()

            # 2. Test RF=2 (Replicated: sent to both nodes)
            client.topology.replication_factor = 2
            res2 = client.scatter_ingest_edges(df, edge_type="CONNECTED")
            self.assertEqual(res2["rows_ingested"], 2)
            total_calls_rf2 = mock_flight_1.do_put.call_count + mock_flight_2.do_put.call_count
            # 2 partitions * 2 replicas = 4 do_put calls
            self.assertEqual(total_calls_rf2, 4)


class TestGraphSpacesAndAiIntegrations(unittest.TestCase):
    def test_graph_space_lifecycle_methods(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient(endpoint="http://127.0.0.1:8847", graph="fraud_space")
            self.assertEqual(client.graph, "fraud_space")

            client.query = MagicMock(
                side_effect=lambda q: QueryResult({"status": "ok", "active_graph": client.graph})
            )
            client.create_graph("analytics_space")
            client.query.assert_called_with("CREATE GRAPH IF NOT EXISTS analytics_space;")

            active = client.use_graph("analytics_space")
            self.assertEqual(active, "analytics_space")
            self.assertEqual(client.graph, "analytics_space")
            client.query.assert_called_with("USE GRAPH analytics_space;")

            client.drop_graph("analytics_space")
            self.assertEqual(client.graph, "default")
            client.query.assert_called_with("DROP GRAPH IF EXISTS analytics_space;")

    def test_langchain_gdb_vector_store_and_graph_qa(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient(graph="rag_space")
            client.schema = MagicMock(return_value={"vertices": [], "edges": []})
            executed_queries = []

            def fake_query(q):
                executed_queries.append(q)
                if q.startswith("CALL vector.similaritySearch"):
                    return QueryResult(
                        {
                            "status": "ok",
                            "columns": ["vertex_id", "score"],
                            "rows": [[101, 0.98], [102, 0.87]],
                        }
                    )
                if "MATCH (n:Document {id: 101})" in q:
                    return QueryResult(
                        {
                            "status": "ok",
                            "columns": ["n.text", "n.metadata"],
                            "rows": [["GDB supports native HNSW indexing", '{"topic": "hnsw"}']],
                        }
                    )
                if "MATCH (n:Document {id: 102})" in q:
                    return QueryResult(
                        {
                            "status": "ok",
                            "columns": ["n.text", "n.metadata"],
                            "rows": [["GDB supports Pregel BSP analytics", '{"topic": "pregel"}']],
                        }
                    )
                if "MATCH (a {id: 101})-[:RELATES_TO]->(b)" in q:
                    return QueryResult(
                        {
                            "status": "ok",
                            "columns": ["src", "dst"],
                            "rows": [[101, 102]],
                        }
                    )
                return QueryResult({"status": "ok", "columns": [], "rows": []})

            client.query = MagicMock(side_effect=fake_query)

            store = GdbVectorStore(
                client=client,
                embedding=lambda txt: [0.5, 0.5, 0.5, 0.5],
                dimension=4,
            )
            self.assertTrue(any("CREATE VECTOR INDEX ON Document(embedding);" in q for q in executed_queries))

            ids = store.add_texts(
                ["GDB supports native HNSW indexing", "GDB supports Pregel BSP analytics"],
                metadatas=[{"topic": "hnsw"}, {"topic": "pregel"}],
                ids=[101, 102],
            )
            self.assertEqual(ids, [101, 102])

            docs = store.similarity_search("HNSW vector search", k=2)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].page_content, "GDB supports native HNSW indexing")
            self.assertEqual(docs[0].metadata["topic"], "hnsw")

            graph_store = GdbGraphStore(client)
            qa_chain = GdbGraphQAChain(
                graph_store=graph_store,
                vector_store=store,
                llm=lambda prompt: f"Synthesized answer from {len(prompt)} chars",
            )
            out = qa_chain.invoke("How does GDB execute vector search?", k=2)
            self.assertIn("Synthesized answer", out["answer"])
            self.assertEqual(len(out["documents"]), 2)
            self.assertEqual(out["graph_context"], [{"src": 101, "dst": 102}])

    def test_langgraph_gdb_checkpoint_saver_and_graphrag_toolkit(self):
        with patch.object(ClusterTopology, "refresh"):
            client = GdbClient(graph="agent_memory")
            client.schema = MagicMock(return_value={"vertices": [], "edges": []})
            client.query = MagicMock(return_value=QueryResult({"status": "ok", "rows_affected": 1}))

            saver = GdbCheckpointSaver(client)
            cfg1 = saver.put(
                {"configurable": {"thread_id": "thread-42"}},
                {"id": "step-1", "messages": ["hello"]},
                metadata={"step": 1},
            )
            self.assertEqual(cfg1["configurable"]["checkpoint_id"], "step-1")

            cfg2 = saver.put(
                cfg1,
                {"id": "step-2", "messages": ["hello", "world"]},
                metadata={"step": 2},
            )
            self.assertEqual(cfg2["configurable"]["checkpoint_id"], "step-2")

            latest = saver.get(cfg2)
            self.assertEqual(latest["id"], "step-2")
            history = list(saver.list({"configurable": {"thread_id": "thread-42"}}))
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].checkpoint["id"], "step-2")
            self.assertEqual(history[1].checkpoint["id"], "step-1")


if __name__ == "__main__":
    unittest.main()

