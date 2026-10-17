import random
import numpy as np
from board import Board
from players import BasePlayer
from actions import Action

class Game:
    DICE_MIN = 1
    DICE_MAX = 6
    FREE_SEA_MOVES = 2
    CAPE_TOWN_BONUS = 500
    SLAVE_COAST_STUN = 3
    GEMS = ("Topaz", "Smaragd", "Ruby")
    def __init__(self, player_count: int = 5):
        if player_count < 1:
            raise ValueError("player_count must be at least 1")
        self.player_count = player_count
        self.board = Board()
        self.board.setup()          # load the JSON map
        self.players: list[BasePlayer] = []
        self.turn = 0               # full rounds played
        self.current = 0            # index of the player whose turn it is
        self.winner: int | None = None   # player index
        self.star_found = False
        self.maxHeatStartValue = 0
        self.maxDistValue = 0
        self.start_ids: list[int] = []
        self.cape_town_claimed = False
        self.reset()

    
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)       # Board.reset_tokens uses the global RNG
        self.board.reset_tokens()   # cheap: reshuffle tokens only
        starts = [self.board.get_node_by_name(n) for n in self.board.STARTING_CITY_NAMES]
        self.maxHeatStartValue = self.board.apply(starts, "distToStart")
        self.maxDistValue = self.board.apply(self.board.get_nodes_with_tokens(), "distValue")
        self.players = []
        for i in range(self.player_count):
            p = BasePlayer()
            p.id = i + 1
            p.location = starts[i % len(starts)].id
            self.players.append(p)
 
        self.turn = 0
        self.current = 0
        self.winner = None
        self.star_found = False
        self.start_ids = {n.id for n in starts}
        self.cape_town_claimed = False
        self.refresh_heat(self.players[self.current])
        self.start_turn()

    def refresh_heat(self, player: BasePlayer) -> None:
        other_player_nodes = [
            node
            for other_player in self.players
            if other_player is not player
            if (node := self.board.get_node_by_id(other_player.location)) is not None
        ]
        self.board.apply(other_player_nodes, "heatValue")

    def refresh_distValue(self) -> None:
        self.board.apply(self.board.get_nodes_with_tokens(), "distValue")

    def start_turn(self) -> None:
        p = self.players[self.current]
        self.refresh_heat(p)
        if p.stun > 0:
            p.stun -= 1
            self.end_turn()
            return

        if p.bandit_stuck:                  # stuck until a 1 or 2 is rolled
            roll = random.randint(self.DICE_MIN, self.DICE_MAX)
            if roll > 2:
                self.end_turn()
                return
            p.bandit_stuck = False

        if p.free_sea: 
            p.moves = self.FREE_SEA_MOVES
        else:
            p.moves = random.randint(self.DICE_MIN, self.DICE_MAX)
 
    def end_turn(self) -> None:
        p = self.players[self.current]
        node = self.board.get_node_by_id(p.location)
        if node is not None and node.name.startswith("bandit"):
            p.bandit_stuck = True           # ended the turn on a bandit node
        p.moves = 0

        self.current = (self.current + 1) % self.player_count
        if self.current == 0:
            self.turn += 1
        self.start_turn()

    def legal_actions(self, player: BasePlayer) -> list[Action]: 
        
        current_node = self.board.get_node_by_id(player.location)
        if current_node is None:
            raise ValueError(f"Player location {player.location} is not on the board")

        actions: list[Action] = [
            Action(
                slot=1,
                kind="stay",
                target=None,
                route=None,
                cost=0,
                moves_used=player.moves,
                distValue=current_node.distValue,
                heatValue=current_node.heatValue,
                distToStart=current_node.distToStart,
            )
        ]

        next_slot = 4
        if player.moves > 0:
            for target, dist_value, heat_value, dist_to_start, route in self.board.move_data(player.location):
                if route == 0:
                    cost = 0
                    moves_used = 1
                elif route == 1:
                    if player.paid_sea or player.free_sea:
                        cost = 0
                        moves_used = 1
                    elif player.money >= 100:
                        cost = 100
                        moves_used = 1
                    else:
                        cost = 0
                        moves_used = max(0,player.moves -1)
                    
                elif route == 3:
                    if player.money < 300:
                        continue
                    cost = 300
                    moves_used = player.moves
                else:
                    continue

                actions.append(
                    Action(
                        slot=next_slot,
                        kind="move",
                        target=target,
                        route=route,
                        cost=cost,
                        moves_used=moves_used,
                        distValue=dist_value,
                        heatValue=heat_value,
                        distToStart=dist_to_start,
                    )
                )
                if next_slot == 12:
                    break
                next_slot += 1

        if current_node.token:
            if player.money >= 100:
                actions.append(
                    Action(
                        slot=2,
                        kind="open_fee",
                        target=None,
                        route=None,
                        cost=100,
                        moves_used=player.moves,
                        distValue=current_node.distValue,
                        heatValue=current_node.heatValue,
                        distToStart=current_node.distToStart,
                    )
                )
            actions.append(
                Action(
                    slot=3,
                    kind="open_gamble",
                    target=None,
                    route=None,
                    cost=0,
                    moves_used=player.moves,
                    distValue=current_node.distValue,
                    heatValue=current_node.heatValue,
                    distToStart=current_node.distToStart,
                )
            )

        return sorted(
            actions,
            key=lambda action: action.slot if action is not None else 13,
        )

    def action_mask(self, actions: list[Action | None]) -> np.ndarray:
        mask = np.zeros(12, dtype=bool)
        for action in actions:
            if action is None:
                continue
            if not 1 <= action.slot <= 12:
                raise ValueError(f"Action slot {action.slot} is outside the range 1..12")
            mask[action.slot - 1] = True
        return mask

    def observe(self, player: BasePlayer, actions: list[Action | None]) -> np.ndarray:
        current_node = self.board.get_node_by_id(player.location)
        if current_node is None:
            raise ValueError(f"Player location {player.location} is not on the board")

        winning_distances = [
            node.distToStart
            for other_player in self.players
            if other_player is not player
            if other_player.available_to_win
            if (node := self.board.get_node_by_id(other_player.location)) is not None
        ]
        any_opponent_ready = len(winning_distances) > 0
        # 1.0 = nobody can win (safe), approaches 0 as a ready opponent nears a start node
        closest_winning_distance = min(winning_distances, default=self.maxHeatStartValue)

        obs = [
            np.clip(player.money / 2000, 0, 1),
            np.clip(player.moves / 6, 0, 1),
            1 - np.clip(current_node.heatValue / self.maxHeatStartValue, 0, 1),
            1 - np.clip(current_node.distValue / self.maxDistValue, 0, 1),
            1 - np.clip(current_node.distToStart / self.maxHeatStartValue, 0, 1),
            int(bool(current_node.token)),
            int(player.available_to_win),
            np.clip(closest_winning_distance / self.maxHeatStartValue, 0, 1),
            int(any_opponent_ready),
            int(self.star_found),
            self.board.token_value(),
        ]

        move_actions = sorted(
            (
                action
                for action in actions
                if action is not None and action.kind == "move"
            ),
            key=lambda action: action.slot,
        )[:9]
        goal = "distToStart" if player.available_to_win else "distValue"
        for action in move_actions:
            obs.extend([
                float(np.clip(getattr(current_node, goal) - getattr(action, goal), -1, 1)), 
                1 - np.clip(action.distToStart / self.maxHeatStartValue, 0, 1),
                1 - np.clip(action.heatValue / self.maxHeatStartValue, 0, 1),
                np.clip(action.cost / 300, 0, 1),
                np.clip(action.moves_used / 6, 0, 1),
            ])

        obs.extend([0] * ((9 - len(move_actions)) * 5))
        return np.array(obs, dtype=np.float32)

    def step(self, slot: int) -> dict:
        """Apply one action (slot 1..12) for game.current. Returns an info dict
        that the Gym wrapper turns into a reward."""
        if self.winner is not None:
            raise RuntimeError("game is over, call reset()")
 
        p = self.players[self.current]
        action = next((a for a in self.legal_actions(p) if a.slot == slot), None)
        if action is None:
            raise ValueError(f"slot {slot} is not legal for player {p.id}")
 
        info = {
            "player": self.current,
            "kind": action.kind,
            "token": None,
            "money_delta": 0,
            "won": False,
            "from": p.location,
            "to": None,
        }
        money_before = p.money
 
        if action.kind == "stay":
            ends_turn = True
        elif action.kind in ("open_fee", "open_gamble"):
            info["token"] = self._open_token(p, action)
            ends_turn = True                # any attempt ends the turn, so gambling can't loop
        else:                               # "move"
            ends_turn = self._move(p, action)
            info["to"] = p.location
 
        info["money_delta"] = p.money - money_before
 
        if p.available_to_win and p.location in self.start_ids:
            self.winner = self.current
            info["won"] = True
            return info                     # no end_turn: the game is over
 
        if ends_turn:
            self.end_turn()
        return info
 
    def _move(self, p, action) -> bool:
        """Returns True if the turn is over."""
        p.money -= action.cost
        p.location = action.target
        p.moves = max(0, p.moves - action.moves_used)
        node = self.board.get_node_by_id(p.location)
 
        if action.route == 1:               # sea
            if action.cost > 0:
                p.paid_sea = True
            elif not p.paid_sea:
                p.free_sea = True           # broke player just set sail
 
        if node.type == 3:                  # reached a city: sailing is over
            p.paid_sea = False
            if p.free_sea:
                p.free_sea = False
                p.moves = 0
            if node.name == "Cape Town" and not self.cape_town_claimed:
                self.cape_town_claimed = True
                p.money += self.CAPE_TOWN_BONUS
 
        if p.moves > 0:
            return False
        return not node.token               # out of moves: only stay if there's a token to open
 
    def _open_token(self, p, action) -> str | None:
        node = self.board.get_node_by_id(p.location)
        p.money -= action.cost              # 100 for open_fee, 0 for gamble
 
        if action.kind == "open_gamble" and random.randint(1, 6) < 4:
            return None                     # failed the 4-5-6 roll, token stays
 
        token = self.board.give_token(node.id)
        if token is None:
            return None
 
        before = p.money
        p.take(token, self.star_found)      # star_found as it was BEFORE this token
        if token == "Star":
            self.star_found = True
 
        if node.name == "Gold Coast" and token in self.GEMS:
            p.money += p.money - before     # double the gem reward
        if node.name == "Slave Coast" and token == "Blank":
            p.stun = self.SLAVE_COAST_STUN
 
        self.refresh_distValue()            # a token disappeared from the board
        return token