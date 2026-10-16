# train.py  --  put next to env.py.  UNTESTED draft.
#
#   pip install gymnasium sb3-contrib
#   python train.py check        # gym API sanity check + one random episode
#   python train.py train        # train MaskablePPO, saves models/afrikan_ppo
#   python train.py eval         # win rate of model vs random-legal and greedy baselines
import sys
import os
import numpy as np

from env import AfrikanTahtiEnv, greedy_opponent
from collections import Counter

PLAYERS = 5
MODEL_PATH = "models/afrikan_ppo"


def check() -> None:
    from gymnasium.utils.env_checker import check_env

    env = AfrikanTahtiEnv(player_count=PLAYERS)
    check_env(env, skip_render_check=True)

    obs, _ = env.reset(seed=0)
    done, total, steps = False, 0.0, 0
    while not done:
        legal = np.flatnonzero(env.action_masks())
        obs, r, terminated, truncated, info = env.step(int(np.random.choice(legal)))
        total += r
        steps += 1
        done = terminated or truncated
    print(f"random episode: {steps} agent steps, return {total:.2f}, info {info}")


def train(timesteps: int = 1_000_000) -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(AfrikanTahtiEnv, n_envs=8, env_kwargs={"player_count": PLAYERS})
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        gamma=0.99,          # long episodes, sparse win reward
        ent_coef=0.01,        # keep exploring
        verbose=1,
        tensorboard_log="runs",
    )
    model.learn(total_timesteps=timesteps)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)


def win_rate(pick_action, episodes: int = 1000) -> tuple[float, float, float]:
    env = AfrikanTahtiEnv(player_count=PLAYERS)
    wins, can_win, turns = 0, 0, []
    for i in range(episodes):
        obs, info = env.reset(seed=i)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(pick_action(env, obs))
            done = terminated or truncated
        wins += int(info["agent_won"])
        can_win += int(env.game.players[env.agent_idx].available_to_win)
        turns.append(info["turn"])
    return wins / episodes, float(np.mean(turns)), can_win / episodes


def evaluate() -> None:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(MODEL_PATH)

    def trained(env, obs):
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(action)

    def random_legal(env, obs):
        return int(np.random.choice(np.flatnonzero(env.action_masks())))

    def greedy(env, obs):
        g = env.game
        return greedy_opponent(g, g.players[env.agent_idx]) - 1

    print(f"fair share with {PLAYERS} players: {1 / PLAYERS:.2f}")
    for name, fn in (("trained", trained), ("random-legal", random_legal), ("greedy", greedy)):
        rate, turns, cw = win_rate(fn)
        print(f"{name:13s} win rate {rate:.2f}   could-win {cw:.2f}   mean turns {turns:.1f}")

def diagnose(episodes: int = 100, deterministic: bool = False) -> None:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(MODEL_PATH)
    env = AfrikanTahtiEnv(player_count=PLAYERS)
    kinds, returns = Counter(), []
    for i in range(episodes):
        obs, _ = env.reset(seed=i)
        done, total = False, 0.0
        while not done:
            g = env.game
            p = g.players[env.agent_idx]
            slot_kind = {a.slot: a.kind for a in g.legal_actions(p)}
            act, _ = model.predict(obs, action_masks=env.action_masks(),
                                   deterministic=deterministic)
            kinds[slot_kind[int(act) + 1]] += 1
            obs, r, term, trunc, _ = env.step(int(act))
            total += r
            done = term or trunc
        returns.append(total)
    print(dict(kinds), f"mean return {np.mean(returns):.2f}")

def greedy_return(episodes: int = 100) -> None:
    env = AfrikanTahtiEnv(player_count=PLAYERS)
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=i)
        done, total = False, 0.0
        while not done:
            g = env.game
            a = greedy_opponent(g, g.players[env.agent_idx]) - 1
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
    print(f"greedy mean return {np.mean(returns):.2f}")

if __name__ == "__main__":
    {"check": check, "train": train, "eval": evaluate, "diag": diagnose, "greedy": greedy_return}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
