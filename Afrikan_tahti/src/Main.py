# main.py
import random, sys
from game import Game
from env import greedy_opponent
from collections import Counter



def play_one(verbose=True, seed=None):
    g = Game(5)
    g.reset(seed)
    for _ in range(5000):
        if g.winner is not None:
            break
        p = g.players[g.current]
        legal = g.legal_actions(p)
        a = random.choice(legal)
        if verbose:
            print(f"t{g.turn} P{p.id} money={p.money} moves={p.moves} -> {a.kind} {a.target}")
        g.step(a.slot)
    return g

def batch(n=200):
    turns, stars, wins = [], 0, 0
    for i in range(n):
        g = play_one(verbose=False, seed=i)
        turns.append(g.turn)
        stars += g.star_found
        wins += g.winner is not None
    print(f"{n} games: finished {wins}, star found {stars}, mean turns {sum(turns)/n:.1f}")

def greedy_batch(n=200):
    wins, turns = Counter(), []
    for i in range(n):
        g = Game(5)
        g.reset(i)
        for _ in range(5000):
            if g.winner is not None:
                break
            p = g.players[g.current]
            g.step(greedy_opponent(g, p))
        if g.winner is not None:
            wins[g.winner] += 1
            turns.append(g.turn)
    turns.sort()
    print(f"finished {len(turns)}/{n}")
    print(f"wins per seat {dict(sorted(wins.items()))}  (seats 0,2,4 start Tangier; 1,3 Cairo)")
    if turns:
        print(f"turns: median {turns[len(turns)//2]}, 90th pct {turns[int(len(turns)*0.9)]}, max {turns[-1]}")

if __name__ == "__main__":
    #batch() if len(sys.argv) > 1 and sys.argv[1] == "batch" else play_one()
    if len(sys.argv) > 1:
        run = sys.argv[1]
        if run == "batch":
            batch()
        elif run == "greedy":
            greedy_batch()
        else:
            play_one()

            