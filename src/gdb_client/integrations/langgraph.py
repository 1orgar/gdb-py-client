"""Official LangGraph integration (`langgraph-gdb`) for GDB v0.6.0 ("Hals").

Provides:
- `GdbCheckpointSaver`: Persistent state & conversation memory checkpointer for
  stateful LangGraph agent workflows, stored natively in a GDB Graph Space.
- `GdbGraphRAGToolkit`: Multi-hop hybrid Graph + Vector + Community Detection
  toolkit for autonomous LangGraph agents.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..client import GdbClient
from .langchain import GdbGraphStore, GdbVectorStore


@dataclass
class CheckpointTuple:
    """LangGraph-compatible checkpoint tuple representation."""
    config: Dict[str, Any]
    checkpoint: Dict[str, Any]
    metadata: Dict[str, Any]
    parent_config: Optional[Dict[str, Any]] = None


class GdbCheckpointSaver:
    """Persistent LangGraph state checkpointer backed by GDB.

    Stores each agent execution step as a `LangGraphCheckpoint` vertex and links
    parent -> child steps via `:NEXT_CHECKPOINT` edges inside the target Graph Space,
    enabling time-travel debugging and branching state graphs.
    """

    def __init__(
        self,
        client: GdbClient,
        graph: Optional[str] = None,
        label: str = "LangGraphCheckpoint",
        edge_type: str = "NEXT_CHECKPOINT",
        ensure_schema: bool = True,
    ):
        self.client = client
        if graph:
            self.client.use_graph(graph)
        self.label = label
        self.edge_type = edge_type
        self._local_cache: Dict[str, List[CheckpointTuple]] = {}
        if ensure_schema:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        schema_info = self.client.schema()
        v_labels = {
            (v.get("label") or v.get("name"))
            for v in schema_info.get("vertices", [])
            if isinstance(v, dict)
        }
        e_types = {
            (e.get("edge_type_name") or e.get("name"))
            for e in schema_info.get("edges", [])
            if isinstance(e, dict)
        }
        if self.label not in v_labels:
            self.client.query(
                f"CREATE VERTEX {self.label} ("
                f"thread_id STRING, checkpoint_id STRING, parent_id STRING, "
                f"state_json STRING, metadata_json STRING, created_at INT64"
                f");"
            )
            self.client.query(f"CREATE INDEX ON {self.label}(thread_id);")
        if self.edge_type not in e_types:
            self.client.query(f"CREATE EDGE {self.edge_type} ();")

    @staticmethod
    def _vertex_id(thread_id: str, checkpoint_id: str) -> int:
        raw = f"{thread_id}::{checkpoint_id}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(raw, digest_size=7).digest(), "big") or 1

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a LangGraph checkpoint vertex and link it to its parent checkpoint."""
        configurable = config.get("configurable", {})
        thread_id = str(configurable.get("thread_id", "default_thread"))
        parent_id = str(configurable.get("checkpoint_id", ""))
        checkpoint_id = str(checkpoint.get("id") or f"ckpt_{int(time.time() * 1000)}")
        meta = metadata or {}

        vid = self._vertex_id(thread_id, checkpoint_id)
        self.client.insert_vertices(
            self.label,
            [
                {
                    "id": vid,
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "parent_id": parent_id,
                    "state_json": json.dumps(checkpoint, ensure_ascii=False),
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                    "created_at": int(time.time() * 1000),
                }
            ],
        )

        if parent_id:
            parent_vid = self._vertex_id(thread_id, parent_id)
            self.client.insert_edges(self.edge_type, [(parent_vid, vid)])

        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_cfg = (
            {"configurable": {"thread_id": thread_id, "checkpoint_id": parent_id}}
            if parent_id
            else None
        )
        tup = CheckpointTuple(
            config=next_config,
            checkpoint=checkpoint,
            metadata=meta,
            parent_config=parent_cfg,
        )
        self._local_cache.setdefault(thread_id, []).append(tup)
        return next_config

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Retrieve the latest or specific `CheckpointTuple` for a `thread_id`."""
        configurable = config.get("configurable", {})
        thread_id = str(configurable.get("thread_id", "default_thread"))
        checkpoint_id = configurable.get("checkpoint_id")

        if thread_id in self._local_cache and self._local_cache[thread_id]:
            history = self._local_cache[thread_id]
            if checkpoint_id:
                for item in reversed(history):
                    if item.config["configurable"].get("checkpoint_id") == str(checkpoint_id):
                        return item
            return history[-1]

        res = self.client.query(
            f"MATCH (c:{self.label} {{thread_id: '{thread_id}'}}) "
            f"RETURN c.checkpoint_id, c.parent_id, c.state_json, c.metadata_json, c.created_at;"
        )
        if not res.rows:
            return None

        rows = sorted(res.rows, key=lambda r: int(r[4] or 0))
        selected = rows[-1]
        if checkpoint_id:
            for r in rows:
                if str(r[0]) == str(checkpoint_id):
                    selected = r
                    break

        ckpt_id, parent_id, state_str, meta_str, _ = selected
        state = json.loads(state_str) if isinstance(state_str, str) and state_str else {}
        meta = json.loads(meta_str) if isinstance(meta_str, str) and meta_str else {}
        parent_cfg = (
            {"configurable": {"thread_id": thread_id, "checkpoint_id": str(parent_id)}}
            if parent_id
            else None
        )
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_id": str(ckpt_id)}},
            checkpoint=state,
            metadata=meta,
            parent_config=parent_cfg,
        )

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve the checkpoint dictionary for `config`."""
        tup = self.get_tuple(config)
        return tup.checkpoint if tup is not None else None

    def list(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Iterate over checkpoints for a thread in reverse chronological order."""
        configurable = config.get("configurable", {})
        thread_id = str(configurable.get("thread_id", "default_thread"))
        items = list(reversed(self._local_cache.get(thread_id, [])))
        if limit is not None:
            items = items[:limit]
        return iter(items)


class GdbGraphRAGToolkit:
    """Agentic GraphRAG Toolkit for LangGraph workflows.

    Combines GDB's HNSW vector index, multi-hop Cypher traversals, and distributed
    Pregel community/centrality analytics into ready-to-call tools for AI agents.
    """

    def __init__(
        self,
        client: GdbClient,
        vector_store: Optional[GdbVectorStore] = None,
        graph_store: Optional[GdbGraphStore] = None,
    ):
        self.client = client
        self.graph_store = graph_store or GdbGraphStore(client)
        self.vector_store = vector_store

    def hybrid_retrieve(
        self,
        query_embedding: Sequence[float],
        k: int = 4,
        edge_type: Optional[str] = None,
        max_hops: int = 2,
    ) -> Dict[str, Any]:
        """Perform HNSW vector search followed by k-hop graph neighborhood expansion."""
        if self.vector_store is None:
            raise ValueError("vector_store is required for hybrid_retrieve")

        docs_with_scores = self.vector_store.similarity_search_by_vector_with_score(
            query_embedding, k=k
        )
        seed_ids = [
            int(doc.metadata["vertex_id"])
            for doc, _ in docs_with_scores
            if "vertex_id" in doc.metadata
        ]
        subgraph_edges = self.subgraph_context(seed_ids, edge_type=edge_type, hops=max_hops)
        return {
            "seed_documents": [
                {
                    "vertex_id": doc.metadata.get("vertex_id"),
                    "score": score,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc, score in docs_with_scores
            ],
            "subgraph_edges": subgraph_edges,
        }

    def subgraph_context(
        self,
        seed_ids: Sequence[int],
        edge_type: Optional[str] = None,
        hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """Expand k-hop neighborhood around seed vertex IDs."""
        rel_pattern = f"[:{edge_type}*1..{max(1, hops)}]" if edge_type else f"[*1..{max(1, hops)}]"
        edges: List[Dict[str, Any]] = []
        for vid in seed_ids:
            q = f"MATCH (a {{id: {int(vid)}}})-{rel_pattern}->(b) RETURN id(a) AS source, id(b) AS target LIMIT 50;"
            edges.extend(self.graph_store.query(q))
        return edges

    def community_summary(self, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run Louvain community detection to group related knowledge entities."""
        arg = f"{{edge_type: '{edge_type}'}}" if edge_type else ""
        return self.graph_store.query(
            f"CALL algo.louvain({arg}) YIELD vertex_id, community;"
        )

    def pagerank_influencers(
        self, edge_type: Optional[str] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Run PageRank centrality to rank the most authoritative entities."""
        arg = f"{{edge_type: '{edge_type}'}}" if edge_type else ""
        rows = self.graph_store.query(
            f"CALL algo.pageRank({arg}) YIELD vertex_id, score;"
        )
        rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return rows[:top_k]
