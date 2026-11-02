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

from opponents import NumpyActor, OpponentMix
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse
import glob

POOL_DIR = "models/pool"

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


def train(timesteps=500_000, mix=True, p_greedy=0.4, window=None, save_every=25_000,
          seed=None, lr=3e-4, lr_decay=False, ent=0.01, n_steps=512, batch=256,
          net=64, lam=0.95) -> None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env

    n_envs = 8
    kwargs = {"player_count": PLAYERS}
    callback = None
    if mix:
        for f in glob.glob(os.path.join(POOL_DIR, "ckpt_*_steps.zip")):
            os.remove(f)                      # stale checkpoints from an earlier run
        kwargs["mix"] = OpponentMix(POOL_DIR, greedy=greedy_opponent,
                                    p_greedy=p_greedy, window=window)
        callback = CheckpointCallback(max(save_every // n_envs, 1), POOL_DIR, name_prefix="ckpt")
    env = make_vec_env(AfrikanTahtiEnv, n_envs=n_envs, env_kwargs=kwargs)
    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=(lambda f: lr * f) if lr_decay else lr,
        n_steps=n_steps, batch_size=batch, gamma=0.99, gae_lambda=lam, ent_coef=ent,
        policy_kwargs=dict(net_arch=dict(pi=[net, net], vf=[net, net])),
        seed=seed, verbose=1, tensorboard_log="runs",
    )
    model.learn(total_timesteps=timesteps, callback=callback)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)


def verify(episodes: int = 20) -> None:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(MODEL_PATH, device="cpu")
    actor = NumpyActor.from_model(model)
    env = AfrikanTahtiEnv(player_count=PLAYERS)
    n = bad = 0
    for i in range(episodes):
        obs, _ = env.reset(seed=i)
        done = False
        while not done:
            mask = env.action_masks()
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            n += 1
            bad += int(a) != actor.act(obs, mask)
            obs, _, term, trunc, _ = env.step(int(a))
            done = term or trunc
    print(f"{n} decisions, {bad} mismatches")


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


def evaluate(episodes: int = 1000) -> None:
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
        rate, turns, cw = win_rate(fn, episodes)
        print(f"{name:13s} win rate {rate:.2f}   could-win {cw:.2f}   mean turns {turns:.1f}")

def funnel(pick_action, episodes: int = 1000, label: str = "") -> None:
    env = AfrikanTahtiEnv(player_count=PLAYERS)
    out = Counter()
    ready_turn = {"won": [], "lost_race": [], "truncated": []}
    ready_dist = {"won": [], "lost_race": [], "truncated": []}
    end_turn = {"won": [], "lost_race": [], "truncated": []}
    kinds = Counter()
    for i in range(episodes):
        obs, info = env.reset(seed=i)
        g = env.game
        me = g.players[env.agent_idx]
        r_turn = r_dist = None
        done = False
        while not done:
            ready_before = me.available_to_win
            act = pick_action(env, obs)
            if ready_before:
                a = next((x for x in g.legal_actions(me) if x.slot == act + 1), None)
                if a is not None:
                    kinds[a.kind + (f"_r{a.route}" if a.kind == "move" else "")] += 1
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            if r_turn is None and me.available_to_win:
                r_turn = g.turn
                r_dist = g.board.get_node_by_id(me.location).distToStart
        if r_turn is None:
            out["never_ready"] += 1
            continue
        key = "won" if info["agent_won"] else ("lost_race" if info["winner"] is not None else "truncated")
        out[key] += 1
        ready_turn[key].append(r_turn)
        ready_dist[key].append(r_dist)
        end_turn[key].append(info["turn"])
    print(f"--- {label} ({episodes} eps) ---")
    print(dict(out))
    for k in ("won", "lost_race", "truncated"):
        if ready_turn[k]:
            print(f"{k:10s} ready at turn {np.mean(ready_turn[k]):.1f}, "
                  f"hops from start when ready {np.mean(ready_dist[k]):.1f}, "
                  f"game ended turn {np.mean(end_turn[k]):.1f}")
    print("actions while ready:", dict(kinds))

def funnel_cmd() -> None:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(MODEL_PATH)

    def trained(env, obs, det=True):
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=det)
        return int(a)

    def greedy(env, obs):
        g = env.game
        return greedy_opponent(g, g.players[env.agent_idx]) - 1

    def agent_then_greedy(env, obs):          # agent searches, greedy returns
        p = env.game.players[env.agent_idx]
        return greedy(env, obs) if p.available_to_win else trained(env, obs)

    def greedy_then_agent(env, obs):          # greedy searches, agent returns
        p = env.game.players[env.agent_idx]
        return trained(env, obs) if p.available_to_win else greedy(env, obs)

    for name, fn in (("trained", trained), ("greedy", greedy),
                     ("agent->greedy", agent_then_greedy), ("greedy->agent", greedy_then_agent)):
        funnel(fn, 1000, name)

def diagnose(episodes: int = 100, deterministic: bool = True) -> None:
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
    
def ready_audit(episodes: int = 1000) -> None:
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(MODEL_PATH)
    env = AfrikanTahtiEnv(player_count=PLAYERS)
    c = Counter()
    for i in range(episodes):
        obs, _ = env.reset(seed=i)
        g = env.game
        me = g.players[env.agent_idx]
        done = False
        while not done:
            if me.available_to_win:
                legal = g.legal_actions(me)
                here = g.board.get_node_by_id(me.location).distToStart
                act, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                a = next(x for x in legal if x.slot == int(act) + 1)
                walk = [x for x in legal if x.kind == "move" and x.route != 3]
                improving = [x for x in walk if g.board.get_node_by_id(x.target).distToStart < here]
                if me.moves > 0:
                    c["decisions_with_moves"] += 1
                    c["improving_available"] += bool(improving)
                    air = [x for x in legal if x.kind == "move" and x.route == 3]
                    best_air = max((here - g.board.get_node_by_id(x.target).distToStart for x in air), default=0)
                    if best_air > me.moves:                        # flying beats walking all remaining moves
                        c["air_beats_walk"] += 1
                        c["air_beats_walk_taken"] += bool(a.kind == "move" and a.route == 3)
                    if a.kind == "move" and a.route != 3:
                        d = here - g.board.get_node_by_id(a.target).distToStart
                        c["walk_improve" if d > 0 else "walk_flat" if d == 0 else "walk_worse"] += 1
                        c["walk_improve_but_cost"] += (d > 0 and a.cost > 0)
                    elif a.kind == "move":
                        gain = here - g.board.get_node_by_id(a.target).distToStart
                        c["air"] += 1
                        c["air_gain_lt_moves"] += gain < me.moves
                        c["air_gain_sum"] += gain
                        c["air_moves_sum"] += me.moves
                    else:
                        c[f"non_move_{a.kind}_with_moves_left"] += 1
                else:
                    c["decisions_no_moves"] += 1
            obs, _, term, trunc, _ = env.step(int(act) if False else int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]))
            done = term or trunc
    print(dict(c))



if __name__ == "__main__":
    cmds = ["check", "train", "eval", "diag", "greedy", "funnel", "audit", "verify"]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="check", choices=cmds)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--no-mix", action="store_true")
    ap.add_argument("--p-greedy", type=float, default=0.4)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=25_000)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-decay", action="store_true")
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--net", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.95)
    a = ap.parse_args()
    MODEL_PATH = a.model
    {
        "check": check, "eval": lambda: evaluate(a.episodes), "diag": diagnose,
        "greedy": greedy_return, "funnel": funnel_cmd, "audit": ready_audit, "verify": verify,
        "train": lambda: train(a.steps, not a.no_mix, a.p_greedy, a.window, a.save_every, a.seed, a.lr, a.lr_decay, a.ent, a.n_steps, a.batch, a.net, a.lam),
    }[a.cmd]()