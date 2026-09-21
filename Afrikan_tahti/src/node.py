from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Node:
    id: int
    name: str
    token: str = ''
    type: int = 0  # 0 = ground, 1 = sea, 3 = city
    value: int = 0  # For cities this is 1 if it has a token, 0 if empty.
    nearValue: int = 0  # For cities this is the number of neighboring tokens.
    heat: int = 0  # For cities this is the sum direct heat and indirect heat from neighbors.
    neighbors: List[tuple[int, int, int]] = field(default_factory=list)

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

    def remove_neighbor(self, neighbor_id: int, route: int | None = None) -> None:
        if route is None:
            self.neighbors = [n for n in self.neighbors if n[0] != neighbor_id]
        else:
            self.neighbors = [n for n in self.neighbors if not (n[0] == neighbor_id and n[1] == route)]

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
            "neighbors": [{"id": nid, "route": route, "distance": distance} for nid, route, distance in self.neighbors]
        }

    def __repr__(self) -> str:
        return f"Node(id={self.id}, name={self.name!r}, token={self.token!r}, type={self.type}, neighbors={self.neighbors})"
