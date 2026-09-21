from board import Board


class Game:
    def __init__(self, player_count=6):
        self.board = Board()
        self.players = [] # This will be filled by player objects
        self.turn = 0
        self.star_found = False

    def handle_move(self, player, target_city):
        # 1. Check if move is legal (is it adjacent?)
        # 2. Roll dice
        # 3. Update player position
        
        pass