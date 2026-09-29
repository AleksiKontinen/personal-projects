from board import Board


class Game:
    def __init__(self, player_count = 5):
        self.board = Board()
        self.players = [] # This will be filled by player objects
        self.turn = 0
        self.star_found = False

    