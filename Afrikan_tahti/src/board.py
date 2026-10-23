import json
import random
import warnings
from collections import deque
from pathlib import Path
from typing import List, Dict
 
from node import Node
 
 
class Board:
    STARTING_CITY_NAMES = ("Tangier", "Cairo")      # order = seat order (seat 0 starts at Tangier)
    REQUIRED_NAMES = ("Tangier", "Cairo", "Gold Coast", "Slave Coast", "Cape Town")
    NODE_KINDS = ("city", "fork", "bandit", "road", "sea")
    TOKEN_COUNTS = (
        ("Star", 1),
        ("Ruby", 2),
        ("Smaragd", 3),
        ("Topaz", 4),
        ("Horseshoe", 5),
        ("Bandit", 3),
    )
 
    def __init__(self):
        self.nodes: Dict[int, Node] = {}        # node id == index in map.json == LED index
        self.tokens = {}
        self.available_tokens: List[str] = []
        self.map_meta: dict = {}
        self.reset_tokens()
 
    # ------------------------------------------------------------ node access
    def get_node_by_id(self, node_id: int) -> Node | None:
        return self.nodes.get(node_id)
 
    def get_node_by_name(self, name: str) -> Node | None:
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None
 
    def find_nodes_by_name(self, name: str) -> List[Node]:
        return [node for node in self.nodes.values() if node.name == name]
 
    def search_nodes_by_name(self, query: str) -> List[Node]:
        query_lower = query.lower()
        return [node for node in self.nodes.values() if query_lower in node.name.lower()]
 
    def move_data(self, id: int) -> List[List[int]]:
        node = self.get_node_by_id(id)
        if node is None:
            return []
 
        choices = []
        for neighbor_id, route, _ in node.neighbors:
            neighbor = self.get_node_by_id(neighbor_id)
            if neighbor is not None:
                choices.append([
                    neighbor.id,
                    neighbor.distValue,
                    neighbor.heatValue,
                    neighbor.distToStart,
                    route,
                ])
        return choices
 
    # ---------------------------------------------------------------- map file
    def load_map(self, file_path: str | Path) -> None:
        """Load map.json produced by tools/build_map.py.
 
        Node ids are the list indices (and the LED indices on the real board).
        The neighbor order in the file is kept exactly: the observation slots
        depend on it, and the C++ firmware reads the same file.
        """
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        records = data["nodes"]
        count = len(records)
 
        self.nodes.clear()
        self.tokens.clear()
        self.map_meta = data.get("meta", {})
 
        for index, rec in enumerate(records):
            if rec["id"] != index:
                raise ValueError(f"map node at position {index} has id {rec['id']}; ids must equal positions")
            kind = rec["kind"]
            if kind not in self.NODE_KINDS:
                raise ValueError(f"node {index} has unknown kind {kind!r}")
            routes = set()
            neighbors = []
            for target, route in rec["neighbors"]:
                if not 0 <= target < count or target == index:
                    raise ValueError(f"node {index} ({rec['name']}) has an invalid neighbor {target}")
                if route not in (0, 1, 3):
                    raise ValueError(f"node {index} ({rec['name']}) has an invalid route {route}")
                neighbors.append((target, route))
                if route != 3:
                    routes.add(route)
            if kind == "city":
                node_type = 3
            elif kind == "road":
                node_type = 0
            elif kind == "sea":
                node_type = 1
            else:                                   # fork / bandit take the type of their ways
                node_type = 1 if routes == {1} else 0
            node = Node(id=index, name=rec["name"], type=node_type, kind=kind,
                        x=float(rec["x"]), y=float(rec["y"]))
            for target, route in neighbors:
                node.add_neighbor(target, route, 0)
            self.nodes[index] = node
 
        self._validate_map(data)
 
    def _validate_map(self, data: dict) -> None:
        names: Dict[str, int] = {}
        for node in self.nodes.values():
            if node.name in names:
                raise ValueError(f"duplicate node name {node.name!r}")
            names[node.name] = node.id
        for name in self.REQUIRED_NAMES:
            if name not in names:
                raise ValueError(f"map is missing the required node {name!r}")
 
        for node in self.nodes.values():            # every link must exist in both directions
            for target, route, _ in node.neighbors:
                if (node.id, route, 0) not in self.nodes[target].neighbors:
                    raise ValueError(f"link {node.name} -> {self.nodes[target].name} (route {route}) "
                                     f"has no matching link back")
 
        expected = sorted(names[n] for n in self.STARTING_CITY_NAMES)
        if "starts" in data and sorted(data["starts"]) != expected:
            raise ValueError(f"map 'starts' {sorted(data['starts'])} do not match "
                             f"{self.STARTING_CITY_NAMES} {expected}")
 
        seen = set(expected)                        # everything must be reachable without flying
        queue = deque(expected)
        while queue:
            for target, route, _ in self.nodes[queue.popleft()].neighbors:
                if route != 3 and target not in seen:
                    seen.add(target)
                    queue.append(target)
        if len(seen) != len(self.nodes):
            lost = [self.nodes[i].name for i in self.nodes if i not in seen]
            raise ValueError(f"{len(lost)} node(s) cannot be reached from the start cities: {lost[:10]}")
 
        token_total = sum(count for _, count in self.TOKEN_COUNTS)
        token_cities = sum(1 for n in self.nodes.values()
                           if n.type == 3 and n.name not in self.STARTING_CITY_NAMES)
        if token_cities < token_total:
            warnings.warn(f"map has {token_cities} token cities but {token_total} tokens; "
                          f"tokens will be cut off")
 
    def setup(self, map_file: str | Path | None = None) -> None:
        map_path = Path(map_file) if map_file is not None else Path(__file__).with_name("map.json")
        self.load_map(map_path)
        self.reset_tokens()
 
        starting_nodes = [self.get_node_by_name(name) for name in self.STARTING_CITY_NAMES]
        self.apply(self.get_nodes_with_tokens(), "distValue")
        self.apply(starting_nodes, "distToStart")
 
    # ------------------------------------------------------------------ tokens
    def reset_tokens(self):
        city_nodes = [node for node in self.nodes.values() if node.type == 3]
        starting_city_names = set(self.STARTING_CITY_NAMES)
        for node in city_nodes:
            if node.name in starting_city_names:
                node.token = ''
 
        token_nodes = [node for node in city_nodes if node.name not in starting_city_names]
        tokens: List[str] = []
        for token_name, count in self.TOKEN_COUNTS:
            tokens.extend([token_name] * count)
 
        if len(tokens) < len(token_nodes):
            tokens.extend(["Blank"] * (len(token_nodes) - len(tokens)))
        elif len(tokens) > len(token_nodes):
            tokens = tokens[: len(token_nodes)]
 
        random.shuffle(tokens)
        self.available_tokens = list(tokens)
        self.tokens = {}
 
        for node, token in zip(token_nodes, tokens):
            node.token = token
            self.tokens[node.name] = token
 
    def get_available_tokens(self) -> List[str]:
        return list(self.available_tokens)
 
    def get_nodes_with_tokens(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.type == 3 and node.token]
 
    def token_value(self) -> float:
        total = len(self.available_tokens)
        if total == 0:
            return 0.0
 
        star_found = "Star" not in self.available_tokens
        if not star_found:
            return 1.0 / total
 
        horseshoes = sum(1 for token in self.available_tokens if token == "Horseshoe")
        return horseshoes / total
 
    def give_token(self, node_id: int) -> str | None:
        node = self.get_node_by_id(node_id)
        if node is None:
            return None
 
        token = node.token
        if not token:
            return None
 
        if token not in self.available_tokens:
            return None
 
        node.token = ''
        self.available_tokens.remove(token)
 
        if node.name in self.tokens:
            del self.tokens[node.name]
 
        return token
 
    # --------------------------------------------------------------- distances
    def strip_value(self, effect_type: str) -> None:
        if effect_type not in ("heatValue", "distValue", "distToStart"):
            raise ValueError("effect_type must be 'heatValue' or 'distValue' or 'distToStart'")
 
        for node in self.nodes.values():
            setattr(node, effect_type, 0)
 
    def apply(self, start_nodes: List[Node], effect_type: str) -> int:
        if effect_type not in ("heatValue", "distValue", "distToStart"):
            raise ValueError("effect_type must be 'heatValue' or 'distValue' or 'distToStart'")
 
        self.strip_value(effect_type)
        visited = set()
        queue = deque()
        max_value = 0
 
        for start_node in start_nodes:
            node = self.nodes.get(start_node.id)
            if node is None:
                raise ValueError(f"Start node {start_node.id} is not on this board")
            if node.id not in visited:
                visited.add(node.id)
                queue.append(node)
 
        while queue:
            node = queue.popleft()
            next_value = getattr(node, effect_type) + 1
 
            for neighbor_id, route, _ in node.neighbors:
                if route == 3:
                    continue
                neighbor = self.nodes.get(neighbor_id)
                if neighbor is None or neighbor.id in visited:
                    continue
 
                setattr(neighbor, effect_type, next_value)
                max_value = max(max_value, next_value)
                visited.add(neighbor.id)
                queue.append(neighbor)
 
        return max_value