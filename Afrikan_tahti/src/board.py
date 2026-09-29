import heapq
import json
import re
from collections import deque
from pathlib import Path
import random
from typing import List, Dict

from node import Node


class Board:
    STARTING_CITY_NAMES = ("Tangier", "Cairo")

    def __init__(self):
        # Define nodes and their connections (Graph)
        self.nodes: Dict[int, Node] = {}
        self.tokens = {}
        self.available_tokens: List[str] = []
        self._next_node_id = 0
        self.reset_tokens()

    def _next_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def create_node(self, node_id: int, name: str, token: str = '', type: int = 0) -> Node:
        node = Node(id=node_id, name=name, token=token, type=type)
        self.add_node(node)
        return node

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists")
        self.nodes[node.id] = node

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

    def _is_intersection_name(self, name: str) -> bool:
        return bool(re.match(r'^(fork|bandit)\d+$', name, re.IGNORECASE))

    def _route_type_from_string(self, route_type: str) -> int:
        route_type = route_type.lower()
        if route_type == 'road':
            return 0
        if route_type == 'sea':
            return 1
        if route_type == 'air':
            return 3
        raise ValueError(f"Unknown route type: {route_type}")

    def _create_named_node(self, name: str, route: int | None = None) -> Node:
        if self._is_intersection_name(name):
            node_type = route if route in (0, 1) else 0
        else:
            node_type = 3
        return self.create_node(self._next_id(), name, type=node_type)

    def _ensure_node(self, name: str, route: int | None = None) -> Node:
        existing = self.get_node_by_name(name)
        if existing is not None:
            if self._is_intersection_name(name) and route in (0, 1) and existing.type != route:
                existing.type = route
            return existing
        return self._create_named_node(name, route)

    def load_map_from_json(self, file_path: str) -> None:
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as handle:
            edges = json.load(handle)

        self.nodes.clear()
        self.tokens.clear()
        self._next_node_id = 0

        # create all named nodes first
        for edge in edges:
            route = self._route_type_from_string(edge['type'])
            self._ensure_node(edge['from'], route if route in (0, 1) else None)
            self._ensure_node(edge['to'], route if route in (0, 1) else None)

        for edge in edges:
            source = self.get_node_by_name(edge['from'])
            destination = self.get_node_by_name(edge['to'])
            route = self._route_type_from_string(edge['type'])
            distance = int(edge['distance'])

            if source is None or destination is None:
                continue

            if route == 3:
                source.add_neighbor(destination.id, route, distance)
                destination.add_neighbor(source.id, route, distance)
            else:
                self._attach_segment_nodes(source, destination, route, distance)

    def setup(self, map_file: str | Path | None = None) -> None:
        map_path = Path(map_file) if map_file is not None else Path(__file__).with_name("mapData.json")
        self.load_map_from_json(map_path)
        self.reset_tokens()

        starting_nodes = [self.get_node_by_name(name) for name in self.STARTING_CITY_NAMES]
        if any(node is None for node in starting_nodes):
            raise ValueError("The map must contain both 'Tangier' and 'Cairo' starting nodes")

        self.apply(self.get_nodes_with_tokens(), "distValue")
        self.apply(starting_nodes, "distToStart")

    def _attach_segment_nodes(self, source: Node, destination: Node, route: int, distance: int) -> None:
        if distance <= 0:
            source.add_neighbor(destination.id, route, 0)
            destination.add_neighbor(source.id, route, 0)
            return

        segment_count = distance
        previous_node = source

        for index in range(segment_count):
            segment_name = f"{['road','sea'][route]}_segment_{self._next_node_id}"
            segment_node = self.create_node(self._next_id(), segment_name, type=route)
            previous_node.add_neighbor(segment_node.id, route, 0)
            segment_node.add_neighbor(previous_node.id, route, 0)
            previous_node = segment_node

        previous_node.add_neighbor(destination.id, route, 0)
        destination.add_neighbor(previous_node.id, route, 0)

    def reset_tokens(self):
        token_counts = [
            ("Star", 1),
            ("Ruby", 2),
            ("Smaragd", 3),
            ("Topaz", 4),
            ("Horseshoe", 5),
            ("Bandit", 3),
        ]

        city_nodes = [node for node in self.nodes.values() if node.type == 3]
        starting_city_names = set(self.STARTING_CITY_NAMES)
        for node in city_nodes:
            if node.name in starting_city_names:
                node.token = ''

        token_nodes = [node for node in city_nodes if node.name not in starting_city_names]
        tokens: List[str] = []
        for token_name, count in token_counts:
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

    def strip_value(self, effect_type: str) -> None:
        if effect_type not in ("heatValue", "distValue","distToStart"):
            raise ValueError("effect_type must be 'heatValue' or 'distValue' or 'distToStart'")

        for node in self.nodes.values():
            setattr(node, effect_type, 0)

    def apply(self, start_nodes: List[Node], effect_type: str) -> None:
        if effect_type not in ("heatValue", "distValue", "distToStart"):
            raise ValueError("effect_type must be 'heatValue' or 'distValue' or 'distToStart'")

        self.strip_value(effect_type)
        visited = set()
        queue = deque()

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
                visited.add(neighbor.id)
                queue.append(neighbor)

   

    
