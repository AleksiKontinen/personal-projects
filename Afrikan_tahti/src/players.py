# players.py
import token

import numpy as np

class BasePlayer:
    def __init__(self):
        self.location: int = 0
        self.money: int = 300
        self.available_to_win: bool = False
        self.is_star_found: bool = False
        self.stun: int = 0
        self.path: list = []
        self.moving: bool = False
        self.last_city: int | None = None
    def move(self, amount: int) -> None:
        # moves player by amount along the path and updates location and path accordingly
        if amount <= 0 or not self.path:
            return

        if self.location in self.path:
            current_index = self.path.index(self.location)
        else:
            current_index = 0
            self.location = self.path[0]

        remaining_distance = len(self.path) - current_index - 1
        if remaining_distance <= 0:
            return

        move_steps = min(amount, remaining_distance)
        self.last_city = self.location
        self.location = self.path[current_index + move_steps]
        self.path = self.path[current_index + move_steps:]
        self.moving = True

    def take(self, token: str) -> str:
        # takes a token and updates money and available_to_win accordingly return token if not taken, otherwise None
        
        if token == "Star":
            self.available_to_win = True
            self.is_star_found = True
            return None
        elif token == "Topaz":
            self.money += 300
            return None
        elif token == "Horseshoe":
            if self.is_star_found:
                self.available_to_win = True
            return None
        elif token == "Smaragd":
            self.money += 600
            return None
        elif token == "Topaz":
            self.money += 1000
            return None
        elif token == "Bandit":
            self.money = 0
            return None
        elif token == "Blank":
            return None
        return token

    def assign_path(self, path: list) -> None:
        # assigns a new path to the player
        self.path = path