from dataclasses import dataclass
from typing import List, Optional
import json
import urllib.request
import urllib.error

try:
    import requests
except ImportError:
    requests = None


@dataclass
class NodeInfo:
    node_id: int
    http_url: str
    flight_port: int
    client_flight_port: int
    flight_address: str = "127.0.0.1"
    client_flight_address: str = "127.0.0.1"
    host: str = "127.0.0.1"

    @classmethod
    def from_dict(cls, data: dict, seed_host: Optional[str] = None) -> "NodeInfo":
        http_url = data.get("http_url", "http://127.0.0.1:8847")
        flight_address = data.get("flight_address") or ""
        client_flight_address = data.get("client_flight_address") or ""
        flight_port = int(data.get("flight_port", 8848))
        client_flight_port = int(data.get("client_flight_port", 8860))
        node_id = int(data.get("node_id", 1))

        # Host determination logic for external client Arrow Flight connections:
        host = client_flight_address

        # Check if announced address is loopback / unspecified
        is_loopback = host in ("127.0.0.1", "localhost", "0.0.0.0", "::1", "")

        # If loopback, but connected via explicit remote seed host, prioritize seed host
        if is_loopback and seed_host and seed_host not in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
            host = seed_host
        elif not host:
            if "//" in http_url:
                host_part = http_url.split("//")[1].split(":")[0]
                if host_part:
                    host = host_part
            if not host:
                host = seed_host or "127.0.0.1"

        return cls(
            node_id=node_id,
            http_url=http_url,
            flight_port=flight_port,
            client_flight_port=client_flight_port,
            flight_address=flight_address or host,
            client_flight_address=client_flight_address or host,
            host=host,
        )


class ClusterTopology:
    """Discovers and caches the cluster hash ring and node ports."""

    def __init__(self, seed_url: str = "http://localhost:8847"):
        self.seed_url = seed_url.rstrip("/")
        self.seed_host: Optional[str] = None
        if "//" in self.seed_url:
            host_part = self.seed_url.split("//")[1].split(":")[0]
            if host_part:
                self.seed_host = host_part
        self.nodes: List[NodeInfo] = []
        self.replication_factor: int = 1
        self.partitions: int = 4
        self.refresh()

    def refresh(self):
        url = f"{self.seed_url}/cluster"
        try:
            if requests is not None:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "gdb-client/0.5.7"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            raw_nodes = data.get("ring_nodes", [])
            if raw_nodes:
                self.nodes = [NodeInfo.from_dict(n, seed_host=self.seed_host) for n in raw_nodes]
            else:
                fallback_dict = {
                    "node_id": int(data.get("node_id", 1)),
                    "http_url": self.seed_url,
                    "flight_address": data.get("flight_address", ""),
                    "flight_port": int(data.get("flight_port", 8848)),
                    "client_flight_address": data.get("client_flight_address", ""),
                    "client_flight_port": int(data.get("client_flight_port", 8860)),
                }
                self.nodes = [NodeInfo.from_dict(fallback_dict, seed_host=self.seed_host)]
            self.replication_factor = int(data.get("replication_factor", 1))
            self.partitions = int(data.get("partitions", 4))
        except Exception:
            # Fallback single node
            fallback_host = self.seed_host or "127.0.0.1"
            self.nodes = [
                NodeInfo(
                    node_id=1,
                    http_url=self.seed_url,
                    flight_address=fallback_host,
                    flight_port=8848,
                    client_flight_address=fallback_host,
                    client_flight_port=8860,
                    host=fallback_host,
                )
            ]

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    def target_node_for_key(self, key: int) -> NodeInfo:
        idx = key % len(self.nodes)
        return self.nodes[idx]

    def target_replica_nodes_for_key(self, key: int) -> List[NodeInfo]:
        if not self.nodes:
            return []
        rf = max(1, min(self.replication_factor, len(self.nodes)))
        start_idx = key % len(self.nodes)
        return [self.nodes[(start_idx + r) % len(self.nodes)] for r in range(rf)]
