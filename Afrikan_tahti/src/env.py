# env.py  --  put next to game.py in src/.  UNTESTED draft.
#
# One env.step() = one action by the learning seat. Opponents play inside
# step()/reset() until the turn comes back to the agent (or someone wins).
from __future__ import annotations

from logging import info
import random
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game import Game
from players import BasePlayer

from opponents import OpponentMix

from collections import deque

N_ACTIONS = 12                  # slots 1..12  ->  action index 0..11
MAX_TURN_ACTIONS = 60     # 8 at most, otherways rules are broken

OpponentPolicy = Callable[[Game, BasePlayer], int]   # returns a legal slot (1..12)


# ------------------------------------------------------------------ opponents
def random_opponent(game: Game, player: BasePlayer) -> int:
    return random.choice(game.legal_actions(player)).slot


def greedy_opponent(game: Game, player: BasePlayer) -> int:
    """Opens tokens it stands on, otherwise walks down the distance gradient
    (to tokens, or to the start nodes once it can win). Never flies."""
    legal = game.legal_actions(player)
    board = game.board
    here = board.get_node_by_id(player.location)

    if here.token:
        for kind in ("open_fee", "open_gamble"):
            for a in legal:
                if a.kind == kind:
                    return a.slot

    attr = "distToStart" if player.available_to_win else "distValue"

    def score(a) -> int:                      # raw hop counts from the board nodes
        return getattr(board.get_node_by_id(a.target), attr)

    moves = [a for a in legal if a.kind == "move" and a.route != 3]
    better = [a for a in moves if score(a) < getattr(here, attr)]
    if better:
        best = min(score(a) for a in better)
        return random.choice([a for a in better if score(a) == best]).slot
    if moves and random.random() < 0.5:       # nothing improves: wander a bit
        return random.choice(moves).slot
    return 1                                  # stay


# -------------------------------------------------------------------- rewards
@dataclass
class RewardConfig:
    open_token: float = 1.0      # a token was actually opened (content is ignored)
    win: float = 10.0
    lose: float = 0.0            # someone else won
    shape_scale: float = 0.02    # per hop closer to tokens / to the start nodes
    shape_clip: int = 3          # cap on hops counted per move (air jumps)
    illegal: float = -0.05
    ready_bonus: float = 0.0     # first time available_to_win flips to True
    return_scale: float = 0.1    # per hop closer to the start once ready
    return_clip: int = 15            # large enough that air jumps are paid in full
    ready_turn_penalty: float = 0.03 # per full round the agent spends ready but not home
# ------------------------------------------------------------------------ env
class AfrikanTahtiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        player_count: int = 5,
        opponent: OpponentPolicy = greedy_opponent,
        rewards: RewardConfig | None = None,
        max_turns: int = 60,
        randomize_seat: bool = True,
        mix: OpponentMix | None = None,
    ):
        super().__init__()
        self.game = Game(player_count)
        self.opponent = opponent
        self.rewards = rewards or RewardConfig()
        self.max_turns = max_turns
        self.randomize_seat = randomize_seat
        self.agent_idx = 0
        self._mask = np.ones(N_ACTIONS, dtype=bool)
        self.action_space = spaces.Discrete(N_ACTIONS)
        obs_dim = self._probe_obs_dim()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.mix = mix
        self.seat_policies: dict[int, OpponentPolicy] = {}
        self.seat_names: dict[int, str] = {}
        self.opponent_names: list[str] = []

    # ---- gym API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed)                 # seeds the global RNG the game uses
        if self.randomize_seat:               # seats alternate Tangier/Cairo, so mix them
            self.agent_idx = int(self.np_random.integers(self.game.player_count))
        else:
            self.agent_idx = 0
        self._assign_opponents()
        self._play_opponents()
        return self._observe(), self._info()

    def step(self, action: int):
        g = self.game
        if g.winner is not None or g.current != self.agent_idx:
            raise RuntimeError("step() called when it is not the agent's turn; call reset()")
        
        illegal = not self._mask[int(action)]
        if illegal:
            action = 0

        agent = g.players[self.agent_idx]
        was_ready = agent.available_to_win
        t0 = g.turn
        info = g.step(int(action) + 1)
        info["was_ready"] = was_ready
        info["became_ready"] = agent.available_to_win and not was_ready
        info["illegal"] = illegal
        reward = self._reward(info) 
        self._play_opponents()

        terminated = g.winner is not None
        if terminated and g.winner != self.agent_idx:
            reward += self.rewards.lose

        if agent.available_to_win and not (terminated and g.winner == self.agent_idx):
            reward -= self.rewards.ready_turn_penalty * (g.turn - t0)

        truncated = (not terminated) and g.turn >= self.max_turns
        return self._observe(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:     # picked up by sb3-contrib MaskablePPO
        return self._mask.copy()

    # ---- internals
    def _probe_obs_dim(self) -> int:
        g = self.game
        p = g.players[g.current]
        return int(g.observe(p, g.legal_actions(p)).shape[0])

    def _play_opponents(self) -> None:
        g = self.game
        recent = deque(maxlen=12)
        turn_key, in_turn = None, 0
        while g.winner is None and g.current != self.agent_idx:
            key = (g.turn, g.current)
            in_turn = in_turn + 1 if key == turn_key else 1
            turn_key = key
            if in_turn > MAX_TURN_ACTIONS:
                raise RuntimeError(self._turn_report(recent))
            p = g.players[g.current]
            slot = self.seat_policies.get(g.current, self.opponent)(g, p)
            recent.append((g.turn, p.id, slot, p.location, p.moves, p.money))
            g.step(slot)

    def _turn_report(self, recent) -> str:
        g = self.game
        p = g.players[g.current]
        name = lambda i: g.board.get_node_by_id(i).name
        lines = [f"P{p.id} ({self.seat_names.get(g.current, 'opponent')}) took more than "
                 f"{MAX_TURN_ACTIONS} actions in one turn (rule bug?)",
                 f"state: moves={p.moves} money={p.money} stun={p.stun} "
                 f"paid_sea={p.paid_sea} free_sea={p.free_sea} bandit_stuck={p.bandit_stuck}",
                 "last actions (turn, player, slot, node, moves, money):"]
        lines += [f"  t{t} P{pid} slot {s} @ {name(loc)} moves={m} money={mo}"
                  for t, pid, s, loc, m, mo in recent]
        return "\n".join(lines)

    def _assign_opponents(self) -> None:
        self.seat_policies, self.seat_names, self.opponent_names = {}, {}, []
        if self.mix is None:
            return
        for i in range(self.game.player_count):
            if i != self.agent_idx:
                self.seat_policies[i], name = self.mix.draw(self.np_random)
                self.seat_names[i] = name
                self.opponent_names.append(name)

    def _reward(self, info: dict) -> float:
        r = self.rewards
        total = 0.0
        if info["token"] is not None and not info["was_ready"]:
            total += r.open_token # tokens do not reward after ready
        if info["became_ready"]:
            total += r.ready_bonus # bonus for being able to win (first time only)
        if info["kind"] == "move":
            # assumes node.distValue / node.distToStart hold raw hop counts
            board = self.game.board
            agent = self.game.players[self.agent_idx]
            attr = "distToStart" if agent.available_to_win else "distValue"
            a = getattr(board.get_node_by_id(info["from"]), attr)
            b = getattr(board.get_node_by_id(info["to"]), attr)
            scale = r.return_scale if info["was_ready"] else r.shape_scale
            clip = r.return_clip if info["was_ready"] else r.shape_clip
            total += scale * float(np.clip(a - b, -clip, clip))
        if info["won"]:
            total += r.win
        if info["illegal"]:
            total += r.illegal
        return total

    def _observe(self) -> np.ndarray:
        g = self.game
        if g.winner is not None:              # terminal: dummy obs, stay-only mask
            self._mask = np.zeros(N_ACTIONS, dtype=bool)
            self._mask[0] = True
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        p = g.players[self.agent_idx]
        legal = g.legal_actions(p)
        self._mask = g.action_mask(legal)
        return g.observe(p, legal).astype(np.float32)

    def _info(self) -> dict:
        g = self.game
        return {
            "turn": g.turn,
            "winner": g.winner,
            "agent": self.agent_idx,
            "agent_won": g.winner == self.agent_idx,
            "opponents": tuple(self.opponent_names),
        }
