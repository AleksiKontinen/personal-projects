import heapq
import json
import re
from pathlib import Path
import random
from typing import List, Dict

from node import Node


class Board:
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

    def create_node(self, node_id: int, name: str, token: str = '', type: int = 0,
                    value: int = 0, nearValue: int = 0, heat: int = 0) -> Node:
        node = Node(id=node_id, name=name, token=token, type=type,
                    value=value, nearValue=nearValue, heat=heat)
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
        tokens: List[str] = []
        for token_name, count in token_counts:
            tokens.extend([token_name] * count)

        if len(tokens) < len(city_nodes):
            tokens.extend(["Blank"] * (len(city_nodes) - len(tokens)))
        elif len(tokens) > len(city_nodes):
            tokens = tokens[: len(city_nodes)]

        random.shuffle(tokens)
        self.available_tokens = list(tokens)
        self.tokens = {}

        for node, token in zip(city_nodes, tokens):
            node.token = token
            self.tokens[node.name] = token

    def get_available_tokens(self) -> List[str]:
        return list(self.available_tokens)

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

    def heat_up(self, node_id: int) -> None:
        node = self.get_node_by_id(node_id)
        if node is None:
            return
        node.heat += 10

    def find_closest_token_paths(self, start_id: int, target_count: int) -> List[List[tuple[Node, int | None]]]:
        start = self.get_node_by_id(start_id)
        if start is None or target_count <= 0:
            return []

        def is_token_city(node: Node) -> bool:
            return node.type == 3 and node.token

        def route_weight(route: int) -> int:
            if route == 0:
                return 1
            if route == 1:
                return 2
            if route == 3:
                return 5
            return 1

        paths: List[List[tuple[Node, int | None]]] = []
        visited: Dict[int, int] = {}
        queue: List[tuple[int, int, List[tuple[Node, int | None]]]] = []

        heapq.heappush(queue, (0, start.id, [(start, None)]))
        visited[start.id] = 0

        while queue and len(paths) < target_count:
            cost, node_id, path = heapq.heappop(queue)
            if visited.get(node_id, float('inf')) < cost:
                continue

            current = self.get_node_by_id(node_id)
            if current is None:
                continue

            if node_id != start_id and is_token_city(current):
                paths.append(path)
                continue

            for neighbor_id, route, edge_distance in current.neighbors:
                if edge_distance != 0:
                    continue

                neighbor = self.get_node_by_id(neighbor_id)
                if neighbor is None:
                    continue

                new_cost = cost + route_weight(route)
                if new_cost < visited.get(neighbor_id, float('inf')):
                    visited[neighbor_id] = new_cost
                    new_path = path + [(neighbor, route)]
                    heapq.heappush(queue, (new_cost, neighbor_id, new_path))

        return paths
