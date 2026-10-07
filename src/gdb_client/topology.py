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
    host: str = "127.0.0.1"

    @classmethod
    def from_dict(cls, data: dict) -> "NodeInfo":
        http_url = data.get("http_url", "http://127.0.0.1:8847")
        host = "127.0.0.1"
        if "//" in http_url:
            host_part = http_url.split("//")[1].split(":")[0]
            if host_part:
                host = host_part

        return cls(
            node_id=int(data.get("node_id", 1)),
            http_url=http_url,
            flight_port=int(data.get("flight_port", 8848)),
            client_flight_port=int(data.get("client_flight_port", 8860)),
            host=host,
        )


class ClusterTopology:
    """Discovers and caches the cluster hash ring and node ports."""

    def __init__(self, seed_url: str = "http://localhost:8847"):
        self.seed_url = seed_url.rstrip("/")
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
                req = urllib.request.Request(url, headers={"User-Agent": "gdb-client/0.2.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            raw_nodes = data.get("ring_nodes", [])
            if raw_nodes:
                self.nodes = [NodeInfo.from_dict(n) for n in raw_nodes]
            else:
                self.nodes = [
                    NodeInfo(
                        node_id=int(data.get("node_id", 1)),
                        http_url=self.seed_url,
                        flight_port=int(data.get("flight_port", 8848)),
                        client_flight_port=int(data.get("client_flight_port", 8860)),
                    )
                ]
            self.replication_factor = int(data.get("replication_factor", 1))
            self.partitions = int(data.get("partitions", 4))
        except Exception as e:
            # Fallback single node
            self.nodes = [
                NodeInfo(
                    node_id=1,
                    http_url=self.seed_url,
                    flight_port=8848,
                    client_flight_port=8860,
                )
            ]

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    def target_node_for_key(self, key: int) -> NodeInfo:
        idx = key % len(self.nodes)
        return self.nodes[idx]
