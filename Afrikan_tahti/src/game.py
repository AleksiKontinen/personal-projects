from board import Board
from players import BasePlayer


class Game:
    def __init__(self, player_count = 5):
        self.board = Board()
        self.players = [] # This will be filled by player objects
        self.turn = 0
        self.star_found = False
        self.player_count = player_count
        self.startup()

    def startup(self):
        if self.player_count < 1:
            raise ValueError("player_count must be at least 1")

        self.board.setup()
        starting_nodes = [
            self.board.get_node_by_name(name)
            for name in self.board.STARTING_CITY_NAMES
        ]
        self.players = []
        for index in range(self.player_count):
            player = BasePlayer()
            player.id = index + 1
            player.location = starting_nodes[index % len(starting_nodes)].id
            self.players.append(player)
    
