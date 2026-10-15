import random
import numpy as np
from board import Board
from players import BasePlayer
from actions import Action

class Game:
    
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
        self.maxHeatValue = 0
        self.maxDistValue = 0
        self.maxDistValueToStart = 0
        self.reset()

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
    
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)       # Board.reset_tokens uses the global RNG
        self.board.reset_tokens()   # cheap: reshuffle tokens only
        starts = [self.board.get_node_by_name(n) for n in self.board.STARTING_CITY_NAMES]
        self.refresh_distValue()
        self.maxDistValueToStart = self.board.apply(starts, "distToStart")
 
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
        self.refresh_heat(self.players[self.current])
        self.start_turn()

    def refresh_heat(self, player: BasePlayer) -> None:
        other_player_nodes = [
            node
            for other_player in self.players
            if other_player is not player
            if (node := self.board.get_node_by_id(other_player.location)) is not None
        ]
        self.maxHeatValue = self.board.apply(other_player_nodes, "heatValue")

    def refresh_distValue(self) -> None:
        self.maxDistValue = self.board.apply(
            self.board.get_nodes_with_tokens(), "distValue"
        )

    def start_turn(self) -> None:
        p = self.players[self.current]
        if p.stun > 0:
            p.stun -= 1
            p.moves = 0
        else:
            p.moves = random.randint(self.DICE_MIN, self.DICE_MAX)
 
    def end_turn(self) -> None:
        self.current = (self.current + 1) % self.player_count
        if self.current == 0:
            self.turn += 1
        self.start_turn()

    def legal_actions(self, player: BasePlayer) -> list[Action | None]: # list of 12 actions, some None
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
                        continue

                    if player.money >= 100:
                        cost = 100
                        moves_used = 1
                    elif player.money <= 0:
                        cost = 0
                        moves_used = max(0,player.moves -1)
                    else:
                        continue
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
        self.refresh_heat(player)
        current_node = self.board.get_node_by_id(player.location)
        if current_node is None:
            raise ValueError(f"Player location {player.location} is not on the board")

        winning_distances = [
            node.distToStart
            for other_player in self.players
            if other_player.available_to_win
            if (node := self.board.get_node_by_id(other_player.location)) is not None
        ]
        closest_winning_distance = min(winning_distances, default=0)

        obs = [
            player.money,
            player.moves,
            current_node.heatValue,
            current_node.distValue,
            current_node.distToStart,
            int(bool(current_node.token)),
            int(player.available_to_win),
            closest_winning_distance,
        ]

        move_actions = sorted(
            (
                action
                for action in actions
                if action is not None and action.kind == "move"
            ),
            key=lambda action: action.slot,
        )[:9]
        for action in move_actions:
            obs.extend([
                action.distValue,
                action.heatValue,
                action.distToStart,
                action.cost,
                action.moves_used,
            ])

        obs.extend([0] * (9 - len(move_actions)) * 5)
        return np.array(obs, dtype=np.float32)