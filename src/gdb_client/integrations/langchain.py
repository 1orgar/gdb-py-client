"""Official LangChain integration (`langchain-gdb`) for GDB v0.6.0 ("Hals").

Provides:
- `GdbVectorStore`: Native HNSW-backed vector store using `CALL vector.similaritySearch`
  and `CREATE VECTOR INDEX ON`.
- `GdbGraphStore`: Native Property Graph store for openCypher queries, schema inspection,
  and GraphDocument ingestion.
- `GdbGraphQAChain`: Hybrid Graph + Vector retrieval-augmented generation (GraphRAG) chain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..client import GdbClient, QueryResult

try:
    from langchain_core.documents import Document as LCDocument
except ImportError:
    @dataclass
    class LCDocument:  # type: ignore[no-redef]
        """Lightweight LangChain-compatible Document representation."""
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)


Document = LCDocument


class GdbVectorStore:
    """Native HNSW Vector Store backed by GDB's hybrid Graph + Vector engine.

    Stores documents as vertices in the active Graph Space, builds an in-memory
    HNSW index (`CREATE VECTOR INDEX ON`), and executes sub-millisecond ANN searches
    via `CALL vector.similaritySearch(...)`.
    """

    def __init__(
        self,
        client: GdbClient,
        embedding: Optional[Any] = None,
        label: str = "Document",
        text_property: str = "text",
        embedding_property: str = "embedding",
        metadata_property: str = "metadata",
        dimension: int = 128,
        metric: str = "cosine",
        graph: Optional[str] = None,
        ensure_schema: bool = True,
    ):
        self.client = client
        if graph:
            self.client.use_graph(graph)
        self.embedding = embedding
        self.label = label
        self.text_property = text_property
        self.embedding_property = embedding_property
        self.metadata_property = metadata_property
        self.dimension = dimension
        self.metric = metric

        if ensure_schema:
            self._ensure_schema_and_hnsw()

    def _ensure_schema_and_hnsw(self) -> None:
        schema_info = self.client.schema()
        existing_labels = {
            (v.get("label") or v.get("name"))
            for v in schema_info.get("vertices", [])
            if isinstance(v, dict)
        }
        if self.label not in existing_labels:
            self.client.query(
                f"CREATE VERTEX {self.label} ("
                f"{self.text_property} STRING, "
                f"{self.metadata_property} STRING, "
                f"{self.embedding_property} VECTOR({self.dimension})"
                f");"
            )
            self.client.query(
                f"CREATE VECTOR INDEX ON {self.label}({self.embedding_property});"
            )

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.embedding is None:
            raise ValueError("An embedding model or callable must be provided to embed texts.")
        if hasattr(self.embedding, "embed_documents"):
            return self.embedding.embed_documents(texts)
        if callable(self.embedding):
            return [list(self.embedding(t)) for t in texts]
        raise TypeError("Unsupported embedding object: expected .embed_documents() or callable.")

    def _embed_query(self, text: str) -> List[float]:
        if self.embedding is None:
            raise ValueError("An embedding model or callable must be provided to embed query.")
        if hasattr(self.embedding, "embed_query"):
            return list(self.embedding.embed_query(text))
        if callable(self.embedding):
            return list(self.embedding(text))
        raise TypeError("Unsupported embedding object: expected .embed_query() or callable.")

    @staticmethod
    def _stable_vertex_id(key: Union[int, str]) -> int:
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.isdigit():
            return int(key)
        digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=7).digest()
        return int.from_bytes(digest, "big") or 1

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[Union[int, str]]] = None,
    ) -> List[int]:
        """Insert pre-computed embeddings and texts into GDB HNSW index."""
        text_list = list(texts)
        emb_list = [list(map(float, e)) for e in embeddings]
        meta_list = list(metadatas) if metadatas is not None else [{} for _ in text_list]
        id_list = (
            [self._stable_vertex_id(i) for i in ids]
            if ids is not None
            else [self._stable_vertex_id(f"{idx}:{t}") for idx, t in enumerate(text_list)]
        )

        rows = []
        for vid, txt, emb, meta in zip(id_list, text_list, emb_list, meta_list):
            rows.append(
                {
                    "id": vid,
                    self.text_property: txt,
                    self.metadata_property: json.dumps(meta, ensure_ascii=False),
                    self.embedding_property: emb,
                }
            )
        if rows:
            self.client.insert_vertices(self.label, rows)
        return id_list

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[Union[int, str]]] = None,
    ) -> List[int]:
        """Embed texts and store them as HNSW-indexed graph vertices."""
        text_list = list(texts)
        embeddings = self._embed_documents(text_list)
        return self.add_embeddings(text_list, embeddings, metadatas=metadatas, ids=ids)

    def similarity_search_by_vector_with_score(
        self,
        embedding: Sequence[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Run sub-millisecond HNSW ANN vector search and return `(Document, score)`."""
        vec_str = "[" + ", ".join(f"{float(x):.6f}" for x in embedding) + "]"
        cypher = (
            f"CALL vector.similaritySearch('{self.label}', '{self.embedding_property}', "
            f"{vec_str}, {int(k)}, '{self.metric}') YIELD vertex_id, score;"
        )
        res = self.client.query(cypher)
        if not res.rows:
            return []

        results: List[Tuple[Document, float]] = []
        for row in res.rows:
            vid = int(row[0])
            score = float(row[1])
            prop_res = self.client.query(
                f"MATCH (n:{self.label} {{id: {vid}}}) "
                f"RETURN n.{self.text_property}, n.{self.metadata_property} LIMIT 1;"
            )
            page_content = ""
            meta: Dict[str, Any] = {"vertex_id": vid, "score": score}
            if prop_res.rows and prop_res.rows[0]:
                page_content = str(prop_res.rows[0][0] or "")
                raw_meta = prop_res.rows[0][1]
                if isinstance(raw_meta, str) and raw_meta:
                    try:
                        parsed = json.loads(raw_meta)
                        if isinstance(parsed, dict):
                            meta.update(parsed)
                    except Exception:
                        pass
            results.append((Document(page_content=page_content, metadata=meta), score))
        return results

    def similarity_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = 4,
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(embedding, k=k)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        vec = self._embed_query(query)
        return self.similarity_search_by_vector_with_score(vec, k=k)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        vec = self._embed_query(query)
        return self.similarity_search_by_vector(vec, k=k)

    def delete(self, ids: Iterable[Union[int, str]]) -> bool:
        for raw_id in ids:
            vid = self._stable_vertex_id(raw_id)
            self.client.query(f"MATCH (n:{self.label} {{id: {vid}}}) DETACH DELETE n;")
        return True

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Callable[[str], List[Document]]:
        kwargs = search_kwargs or {"k": 4}
        k = int(kwargs.get("k", 4))
        return lambda q: self.similarity_search(q, k=k)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        client: Optional[GdbClient] = None,
        endpoint: str = "http://localhost:8847",
        graph: str = "default",
        **kwargs: Any,
    ) -> "GdbVectorStore":
        gdb_client = client or GdbClient(endpoint=endpoint, graph=graph)
        dimension = kwargs.pop("dimension", None)
        if dimension is None:
            sample_emb = (
                embedding.embed_documents([texts[0]])[0]
                if hasattr(embedding, "embed_documents")
                else list(embedding(texts[0]))
            )
            dimension = len(sample_emb)
        store = cls(gdb_client, embedding=embedding, dimension=dimension, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store


class GdbGraphStore:
    """LangChain-compatible Property Graph Store backed by GDB."""

    def __init__(self, client: GdbClient, graph: Optional[str] = None):
        self.client = client
        if graph:
            self.client.use_graph(graph)
        self._schema_cache: Dict[str, Any] = {}
        self.refresh_schema()

    def refresh_schema(self) -> Dict[str, Any]:
        """Fetch latest schema catalog from the active GDB Graph Space."""
        self._schema_cache = self.client.schema()
        return self._schema_cache

    @property
    def get_schema(self) -> str:
        """Formatted human/LLM-readable schema description for Text2Cypher prompts."""
        v_list = self._schema_cache.get("vertices", [])
        e_list = self._schema_cache.get("edges", [])
        lines = [f"Graph Space: {self.client.graph}", "Vertex Labels:"]
        for v in v_list:
            name = v.get("label") or v.get("name")
            props = ", ".join(
                f"{p.get('name')}: {p.get('type') or p.get('data_type')}"
                for p in v.get("properties", [])
            )
            lines.append(f"  - (:{name} {{{props}}})")
        lines.append("Edge Types:")
        for e in e_list:
            name = e.get("edge_type_name") or e.get("name")
            props = ", ".join(
                f"{p.get('name')}: {p.get('type') or p.get('data_type')}"
                for p in e.get("properties", [])
            )
            lines.append(f"  - [:{name} {{{props}}}]")
        return "\n".join(lines)

    def query(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute an openCypher query and return rows as a list of dictionaries."""
        res: QueryResult = self.client.query(cypher)
        cols = res.columns
        return [dict(zip(cols, row)) for row in res.rows]

    def add_graph_documents(
        self,
        nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Ingest structured entities and relationships extracted by an LLM."""
        v_count = 0
        e_count = 0
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            lbl = n.get("label", "Entity")
            props = dict(n.get("properties", {}))
            props["id"] = GdbVectorStore._stable_vertex_id(n.get("id", len(props)))
            by_label.setdefault(lbl, []).append(props)

        for lbl, items in by_label.items():
            v_count += self.client.insert_vertices(lbl, items)

        by_edge: Dict[str, List[Tuple[int, int]]] = {}
        for rel in relationships:
            etype = rel.get("type", "RELATED_TO")
            src = GdbVectorStore._stable_vertex_id(rel["source"])
            dst = GdbVectorStore._stable_vertex_id(rel["target"])
            by_edge.setdefault(etype, []).append((src, dst))

        for etype, pairs in by_edge.items():
            e_count += self.client.insert_edges(etype, pairs)

        return {"vertices_inserted": v_count, "edges_inserted": e_count}


class GdbGraphQAChain:
    """Hybrid Graph + Vector Question-Answering Chain (GraphRAG)."""

    def __init__(
        self,
        graph_store: GdbGraphStore,
        vector_store: Optional[GdbVectorStore] = None,
        llm: Optional[Callable[[str], str]] = None,
        edge_type: str = "RELATES_TO",
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm = llm
        self.edge_type = edge_type

    def invoke(
        self,
        question: str,
        query_embedding: Optional[Sequence[float]] = None,
        k: int = 4,
    ) -> Dict[str, Any]:
        """Retrieve hybrid vector + 1-hop graph neighborhood context and synthesize answer."""
        seed_docs: List[Document] = []
        if self.vector_store is not None:
            if query_embedding is not None:
                seed_docs = self.vector_store.similarity_search_by_vector(query_embedding, k=k)
            elif self.vector_store.embedding is not None:
                seed_docs = self.vector_store.similarity_search(question, k=k)

        seed_ids = [
            int(d.metadata["vertex_id"])
            for d in seed_docs
            if "vertex_id" in d.metadata
        ]
        graph_neighbors: List[Dict[str, Any]] = []
        for vid in seed_ids:
            rows = self.graph_store.query(
                f"MATCH (a {{id: {vid}}})-[:{self.edge_type}]->(b) RETURN id(a) AS src, id(b) AS dst LIMIT 20;"
            )
            graph_neighbors.extend(rows)

        context_parts = [d.page_content for d in seed_docs if d.page_content]
        if graph_neighbors:
            context_parts.append("Graph edges: " + json.dumps(graph_neighbors))
        context_str = "\n".join(context_parts)

        answer = (
            self.llm(f"Schema:\n{self.graph_store.get_schema}\n\nContext:\n{context_str}\n\nQuestion: {question}")
            if self.llm is not None
            else context_str
        )
        return {
            "question": question,
            "answer": answer,
            "documents": seed_docs,
            "graph_context": graph_neighbors,
        }
