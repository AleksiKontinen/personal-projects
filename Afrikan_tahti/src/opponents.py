# opponents.py  --  put next to env.py.
#
# Opponent mixing for training: each opponent seat is filled per episode with
# either the greedy bot or an older checkpoint of the model being trained.
#
# Checkpoints are evaluated with plain numpy (the actor half of the MlpPolicy is
# just a tanh MLP), so 4 model opponents per round cost microseconds instead of
# a torch call each. NumpyActor is also the reference for the C++ port.
from __future__ import annotations

import os
import re
from typing import Callable

import numpy as np

_CKPT_RE = re.compile(r"_(\d+)_steps\.zip$")      # CheckpointCallback file names


class NumpyActor:
    """Actor of a MaskablePPO MlpPolicy: tanh hidden layers + linear action head."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers                      # [(W, b), ...], last one is the action head

    @classmethod
    def from_model(cls, model) -> "NumpyActor":
        import torch.nn as nn

        policy = model.policy
        if policy.activation_fn.__name__ != "Tanh":
            raise ValueError(f"expected Tanh activations, got {policy.activation_fn.__name__}")
        ext = policy.mlp_extractor
        modules = list(getattr(ext, "shared_net", [])) + list(ext.policy_net) + [policy.action_net]
        layers = [
            (m.weight.detach().cpu().numpy().astype(np.float32),
             m.bias.detach().cpu().numpy().astype(np.float32))
            for m in modules if isinstance(m, nn.Linear)
        ]
        return cls(layers)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        for W, b in self.layers[:-1]:
            x = np.tanh(W @ x + b)
        W, b = self.layers[-1]
        return W @ x + b

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Action index 0..11. rng=None -> argmax, otherwise sample from the masked softmax."""
        z = np.where(mask, self.logits(obs), -np.inf).astype(np.float64)
        if rng is None:
            return int(np.argmax(z))
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(rng.choice(len(p), p=p))


class ActorPolicy:
    """Adapts a NumpyActor to the env's OpponentPolicy signature: (game, player) -> slot 1..12."""

    def __init__(self, actor: NumpyActor, rng: np.random.Generator | None):
        self.actor, self.rng = actor, rng

    def __call__(self, game, player) -> int:
        legal = game.legal_actions(player)
        obs = game.observe(player, legal)
        mask = game.action_mask(legal)
        return self.actor.act(obs, mask, self.rng) + 1


class OpponentMix:
    """Draws an opponent policy per seat.

    p_greedy   chance a seat plays the greedy bot (always greedy while no checkpoint exists)
    p_latest   of the non-greedy draws, chance of taking the newest checkpoint
               (the rest is uniform over the pool, which keeps old versions around)
    window     only the newest N checkpoints are used (None = all of them)
    sample     checkpoints sample from their policy (True) or play argmax (False)
    """

    def __init__(self, directory: str, greedy: Callable, p_greedy: float = 0.4,
                 p_latest: float = 0.3, window: int | None = None, sample: bool = True):
        self.dir = directory
        self.greedy = greedy
        self.p_greedy, self.p_latest = p_greedy, p_latest
        self.window, self.sample = window, sample
        self._actors: dict[str, NumpyActor | None] = {}

    def checkpoints(self) -> list[str]:
        try:
            names = os.listdir(self.dir)
        except FileNotFoundError:
            return []
        found = sorted((int(m.group(1)), n) for n in names if (m := _CKPT_RE.search(n)))
        paths = [os.path.join(self.dir, n) for _, n in found]
        return paths[-self.window:] if self.window else paths

    def _actor(self, path: str) -> NumpyActor | None:
        if path not in self._actors:
            try:
                from sb3_contrib import MaskablePPO
                self._actors[path] = NumpyActor.from_model(MaskablePPO.load(path, device="cpu"))
            except Exception:                     # half-written file: try again next draw
                return None
        return self._actors[path]

    def draw(self, rng: np.random.Generator) -> tuple[Callable, str]:
        paths = self.checkpoints()
        if not paths or rng.random() < self.p_greedy:
            return self.greedy, "greedy"
        if rng.random() < self.p_latest:
            path = paths[-1]
        else:
            path = paths[int(rng.integers(len(paths)))]
        actor = self._actor(path)
        if actor is None:
            return self.greedy, "greedy"
        return ActorPolicy(actor, rng if self.sample else None), os.path.basename(path)[:-4]
