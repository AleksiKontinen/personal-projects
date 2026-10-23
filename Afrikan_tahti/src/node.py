from dataclasses import dataclass, field
from typing import List, Dict
 
 
@dataclass
class Node:
    id: int                 # equals the LED index on the physical board
    name: str
    token: str = ''
    type: int = 0           # 0 = ground, 1 = sea, 3 = city
    distValue: int = 0      # hops to the nearest token (pathfinding / scoring)
    heatValue: int = 0      # reflects how many players are in the vicinity
    neighbors: List[tuple[int, int, int]] = field(default_factory=list)
    distToStart: int = 0    # hops to the nearest starting city
    kind: str = ''          # city | fork | bandit | road | sea  (from map.json)
    x: float = 0.0          # position on the board image, in pixels
    y: float = 0.0
 
    def __post_init__(self) -> None:
        if self.type not in (0, 1, 3):
            raise ValueError("type must be 0 (ground), 1 (sea), or 3 (city)")
 
    def add_neighbor(self, neighbor_id: int, route: int, distance: int) -> None:
        if route not in (0, 1, 3):
            raise ValueError("route must be 0 (ground), 1 (sea), or 3 (air)")
        if distance < 0:
            raise ValueError("distance must be a non-negative integer")
        neighbor = (neighbor_id, route, distance)
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
 
    def neighbors_of_type(self, nodes: Dict[int, "Node"], type_: int) -> List[int]:
        return [nid for nid, _, _ in self.neighbors if nid in nodes and nodes[nid].type == type_]
 
    def neighbors_of_route(self, route: int) -> List[int]:
        return [nid for nid, r, _ in self.neighbors if r == route]
 
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "token": self.token,
            "type": self.type,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "neighbors": [{"id": nid, "route": route, "distance": distance} for nid, route, distance in self.neighbors]
        }
 
    def __repr__(self) -> str:
        return (f"Node(id={self.id}, name={self.name!r}, kind={self.kind!r}, token={self.token!r}, "
                f"type={self.type}, neighbors={self.neighbors})")