# players.py
import numpy as np

class BasePlayer:
    def __init__(self):
        self.location: int = 0
        self.money: int = 300
        self.available_to_win: bool = False
        self.stun: int = 0
        self.moves: int = 0
        self.id: int = 0
        self.paid_sea: bool = False
        self.free_sea: bool = False
        self.bandit_stuck: bool = False
    def take(self, token: str, is_star_found: bool) -> str:
        # takes a token and updates money and available_to_win accordingly return token if not taken, otherwise None
        
        if token == "Star":
            self.available_to_win = True
            self.is_star_found = True
            return None
        elif token == "Topaz":
            self.money += 300
            return None
        elif token == "Horseshoe":
            if is_star_found:
                self.available_to_win = True
            return None
        elif token == "Smaragd":
            self.money += 600
            return None
        elif token == "Ruby":
            self.money += 1000
            return None
        elif token == "Bandit":
            self.money = 0
            return None
        elif token == "Blank":
            return None
        return token

    def move(self, new_location: int) -> int:
        self.location = new_location
        self.moves -= 1
        return self.moves