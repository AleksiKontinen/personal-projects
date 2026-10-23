"""Plan the LED strip: one chain through every node, shown on the board image.

    pip install pygame numpy
    python plan_strip.py board_nodes.json [order.json] [options]

Writes order.json: tracer ids in strip order (LED 0 first). Feed it to the map builder:
    python build_map.py board_nodes.json map.json --order order.json

A chain that only ever steps between neighbouring nodes (a true Hamiltonian
path) usually does not exist, because the road network branches. So the
planner finds the SHORTEST chain through all nodes in which a step may also be a
wire jump between non-neighbouring nodes. Steps along real road/sea
connections are drawn in colour (a gradient from LED 0 to the last LED), wire
jumps in red. Air connections are never used (they are not physical).

Options
  --start NAME|ID    node the chain must start at (where the ESP32 sits). Default: free.
  --seconds 10       search time; more time gives a (slightly) shorter chain
  --jump-cost PX     extra cost per jump, in image pixels. Default = median spacing of
                     connected nodes. 0 = minimise pure wire length; large = fewest cuts.
  --width-cm 30      real board width, to report wire lengths in cm
  --image board.jpg  board image (default: the path stored in the nodes file)
  --scale 1.5        initial zoom (the window is resizable and fits your screen)
  --view             do not solve, just show the existing order.json

Keys: space animate a light along the strip   [ ] animation speed   r search again
      l toggle labels   + / - zoom   hover a node to see its LED index
"""
import colorsys
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame


# ------------------------------------------------------------------- data
def load_board(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    adj = set()
    for e in data["edges"]:
        if e["route"] in (0, 1):                  # physical connections only
            adj.add(frozenset((e["a"], e["b"])))
    return data, nodes, adj


def label(nodes, nid):
    n = nodes[nid]
    return n["name"] or f"segment@({n['x']:.0f},{n['y']:.0f})"


def default_jump(nodes, adj):
    lengths = []
    for pair in adj:
        a, b = tuple(pair)
        lengths.append(math.hypot(nodes[a]["x"] - nodes[b]["x"], nodes[a]["y"] - nodes[b]["y"]))
    return float(np.median(lengths)) if lengths else 10.0


def build_costs(ids, nodes, adj, jump):
    """D[0] / D[:,0] is a dummy node with cost 0 that turns the open chain into a
    closed tour; node k (0-based in ids) has index k+1 in D."""
    n = len(ids)
    xy = np.array([[nodes[i]["x"], nodes[i]["y"]] for i in ids], dtype=float)
    euclid = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    penalty = np.full((n, n), float(jump))
    pos = {nid: k for k, nid in enumerate(ids)}
    for pair in adj:
        a, b = tuple(pair)
        penalty[pos[a], pos[b]] = penalty[pos[b], pos[a]] = 0.0
    np.fill_diagonal(penalty, 0.0)
    D = np.zeros((n + 1, n + 1))
    D[1:, 1:] = euclid + penalty
    return D


# ----------------------------------------------------------------- solver
def tour_len(p, D):
    return float(D[p, np.roll(p, -1)].sum())


def local_search(p, D, lo):
    """2-opt + Or-opt on the closed tour p (p[0] is the dummy). Positions < lo never move."""
    m = len(p)
    improved = True
    while improved:
        improved = False
        for i in range(lo, m - 1):                              # 2-opt
            pn = np.roll(p, -1)
            a, b = p[i - 1], p[i]
            c, d = p[i + 1:], pn[i + 1:]
            delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = i + 1 + k
                p[i:j + 1] = p[i:j + 1][::-1].copy()
                improved = True
        for L in (1, 2, 3):                                     # Or-opt
            i = lo
            while i + L <= m:
                seg = p[i:i + L].copy()
                left, right = p[i - 1], p[(i + L) % m]
                gain = D[left, seg[0]] + D[seg[-1], right] - D[left, right]
                q = np.concatenate((p[:i], p[i + L:]))
                qn = np.roll(q, -1)
                base = D[q, qn]
                fwd = D[q, seg[0]] + D[seg[-1], qn] - base
                rev = D[q, seg[-1]] + D[seg[0], qn] - base
                if lo == 2:                                     # keep the dummy-start link
                    fwd[0] = rev[0] = np.inf
                kf, kr = int(np.argmin(fwd)), int(np.argmin(rev))
                if fwd[kf] <= rev[kr]:
                    k, s, cost = kf, seg, fwd[kf]
                else:
                    k, s, cost = kr, seg[::-1], rev[kr]
                if cost - gain < -1e-9:
                    p = np.concatenate((q[:k + 1], s, q[k + 1:]))
                    improved = True
                i += 1
    return p


def perturb(p, lo, rng):
    m = len(p)
    if m < lo + 8:
        return p
    i = rng.randrange(lo, m - 4)
    j = min(m - 1, i + rng.randint(2, 15))
    k = min(m, j + rng.randint(2, 15))
    if not i < j < k:
        return p
    return np.concatenate((p[:i], p[j:k], p[i:j], p[k:]))


def solve(D, start, seconds, rng):
    """start: D-index of the fixed first node, or None. Returns 0-based node indices."""
    m = D.shape[0]
    n = m - 1

    def nearest_neighbour(s):
        left, path = set(range(1, m)), [s]
        left.discard(s)
        while left:
            c = path[-1]
            nxt = min(left, key=lambda k: D[c, k])
            path.append(nxt)
            left.discard(nxt)
        return path

    lo = 2 if start is not None else 1
    starts = [start] if start is not None else rng.sample(range(1, m), min(8, n))
    best = None
    for s in starts:
        p = local_search(np.array([0] + nearest_neighbour(s)), D, lo)
        if best is None or tour_len(p, D) < tour_len(best, D):
            best = p
    cur, cur_len = best, tour_len(best, D)
    t_end = time.time() + seconds
    while time.time() < t_end:
        cand = local_search(perturb(cur.copy(), lo, rng), D, lo)
        cand_len = tour_len(cand, D)
        if cand_len < cur_len - 1e-9:
            cur, cur_len = cand, cand_len
    return [int(k) - 1 for k in cur[1:]]


# --------------------------------------------------------------- reporting
def analyse(order, nodes, adj):
    total, jumps = 0.0, []
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        d = math.hypot(nodes[a]["x"] - nodes[b]["x"], nodes[a]["y"] - nodes[b]["y"])
        total += d
        if frozenset((a, b)) not in adj:
            jumps.append((k, a, b, d))
    return total, jumps


def report(order, nodes, adj, cm_per_px):
    total, jumps = analyse(order, nodes, adj)
    unit = (lambda d: f"{d * cm_per_px:.1f} cm") if cm_per_px else (lambda d: f"{d:.0f} px")
    print(f"\nLED 0 = {label(nodes, order[0])}   LED {len(order) - 1} = {label(nodes, order[-1])}")
    print(f"{len(order)} LEDs, {len(jumps)} wire jump(s), "
          f"total link length {unit(total)}, jumps total {unit(sum(j[3] for j in jumps))}")
    if len(jumps) == 0:
        print("no jumps: the whole strip follows real connections")
    for k, a, b, d in jumps:
        print(f"  jump LED {k} ({label(nodes, a)}) -> LED {k + 1} ({label(nodes, b)})  {unit(d)}")
    return total, jumps


# ---------------------------------------------------------------- viewer
def hue(t):
    r, g, b = colorsys.hsv_to_rgb(0.75 * (1 - t), 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw(screen, bg, font, scale, nodes, order, adj, hover, head, labels, hud):
    screen.fill((40, 40, 40))
    screen.blit(bg, (0, 0))
    n = len(order)
    pts = [(int(nodes[i]["x"] * scale), int(nodes[i]["y"] * scale)) for i in order]
    jump_ends = set()
    for k in range(n - 1):
        if frozenset((order[k], order[k + 1])) in adj:
            pygame.draw.line(screen, hue(k / n), pts[k], pts[k + 1], 3)
        else:
            pygame.draw.line(screen, (255, 40, 40), pts[k], pts[k + 1], 1)
            jump_ends.update((k, k + 1))
    for k, p in enumerate(pts):
        color = (255, 40, 40) if k in jump_ends else hue(k / n)
        pygame.draw.circle(screen, color, p, 4)
        pygame.draw.circle(screen, (0, 0, 0), p, 4, 1)
        if labels and (k % 10 == 0 or k in jump_ends or k == n - 1):
            screen.blit(font.render(str(k), True, (0, 0, 0), (255, 255, 200)), (p[0] + 5, p[1] - 12))
    if head is not None:
        for t in range(8):
            k = (head - t) % n
            c = 255 - t * 28
            pygame.draw.circle(screen, (c, c, c), pts[k], 7 - t // 2)
    if hover is not None:
        text = f"LED {hover}: {label(nodes, order[hover])}"
        screen.blit(font.render(text, True, (255, 255, 255), (0, 0, 0)), (4, screen.get_height() - 20))
    for i, line in enumerate(hud):
        screen.blit(font.render(line, True, (255, 255, 255), (0, 0, 0)), (4, 4 + i * 18))


def run_viewer(data, nodes, adj, order, out_path, args):
    image_path = args.get("--image", data.get("image"))
    if not image_path or not Path(image_path).exists():
        raise SystemExit(f"board image not found ({image_path!r}); pass --image board.jpg")
    pygame.init()
    img = pygame.image.load(image_path)
    w, h = img.get_size()
    cm_per_px = float(args["--width-cm"]) / w if "--width-cm" in args else None
    desk = pygame.display.Info()
    scale = min(float(args.get("--scale", 1.5)), (desk.current_h - 140) / h, (desk.current_w - 40) / w)
    screen = pygame.display.set_mode((int(w * scale), int(h * scale)), pygame.RESIZABLE)
    pygame.display.set_caption("Afrikan tahti strip planner")
    img = img.convert()
    font = pygame.font.SysFont(None, 18)

    def make_bg():
        return pygame.transform.smoothscale(img, (max(1, int(w * scale)), max(1, int(h * scale))))

    bg = make_bg()
    ids = sorted(nodes)
    seconds = float(args.get("--seconds", 10))
    jump = float(args["--jump-cost"]) if "--jump-cost" in args else default_jump(nodes, adj)
    D = build_costs(ids, nodes, adj, jump)
    seed = 0
    animate, head, speed, labels = False, 0.0, 30.0, True
    clock = pygame.time.Clock()
    running = True
    stats = analyse(order, nodes, adj)
    while running:
        mouse = pygame.mouse.get_pos()
        hover = None
        for k, nid in enumerate(order):
            if math.hypot(nodes[nid]["x"] * scale - mouse[0], nodes[nid]["y"] * scale - mouse[1]) < 8:
                hover = k
                break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                scale = max(0.3, min(event.w / w, event.h / h))
                bg = make_bg()
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_SPACE:
                    animate = not animate
                elif k == pygame.K_l:
                    labels = not labels
                elif k == pygame.K_LEFTBRACKET:
                    speed = max(2.0, speed / 1.5)
                elif k == pygame.K_RIGHTBRACKET:
                    speed = min(600.0, speed * 1.5)
                elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
                           pygame.K_MINUS, pygame.K_KP_MINUS):
                    scale = max(0.3, min(6.0, scale * (1.1 if k in (pygame.K_EQUALS, pygame.K_PLUS,
                                                                      pygame.K_KP_PLUS) else 1 / 1.1)))
                    screen = pygame.display.set_mode((int(w * scale), int(h * scale)), pygame.RESIZABLE)
                    bg = make_bg()
                elif k == pygame.K_r and "--view" not in args:
                    seed += 1
                    draw(screen, bg, font, scale, nodes, order, adj, None, None, labels,
                         [f"searching {seconds:.0f} s ..."])
                    pygame.display.flip()
                    start_idx = start_index(args, nodes, ids)
                    cand = [ids[k2] for k2 in solve(D, start_idx, seconds, random.Random(seed))]
                    cand_stats = analyse(cand, nodes, adj)
                    print(f"\nsearch {seed}: total {cand_stats[0]:.0f} px, {len(cand_stats[1])} jumps "
                          f"(current {stats[0]:.0f} px, {len(stats[1])} jumps)")
                    order, stats = cand, cand_stats
                    Path(out_path).write_text(json.dumps(order), encoding="utf-8")
                    report(order, nodes, adj, cm_per_px)
        if animate:
            head = (head + speed / 60.0) % len(order)
        unit = (lambda d: f"{d * cm_per_px:.0f} cm") if cm_per_px else (lambda d: f"{d:.0f} px")
        longest = max((j[3] for j in stats[1]), default=0)
        hud = [f"{len(order)} LEDs   jumps {len(stats[1])}   links {unit(stats[0])}   "
               f"longest jump {unit(longest)}",
               "space animate   [ ] speed   r search again   l labels   + - zoom"]
        draw(screen, bg, font, scale, nodes, order, adj, hover, int(head) if animate else None,
             labels, hud)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def start_index(args, nodes, ids):
    if "--start" not in args:
        return None
    want = args["--start"]
    for nid in ids:
        if nodes[nid]["name"] == want or str(nid) == want:
            return ids.index(nid) + 1
    raise SystemExit(f"--start {want!r}: no node with that name or id")


def main():
    argv = sys.argv[1:]
    args, positional, i = {}, [], 0
    while i < len(argv):
        if argv[i] == "--view":
            args["--view"] = True
        elif argv[i].startswith("--"):
            args[argv[i]] = argv[i + 1]
            i += 1
        else:
            positional.append(argv[i])
        i += 1
    if not positional:
        print(__doc__)
        return
    data, nodes, adj = load_board(positional[0])
    out_path = positional[1] if len(positional) > 1 else "order.json"
    ids = sorted(nodes)
    cm_per_px = None
    if "--width-cm" in args:
        width = data.get("size", [0])[0]
        cm_per_px = float(args["--width-cm"]) / width if width else None

    if "--view" in args:
        order = json.loads(Path(out_path).read_text(encoding="utf-8"))
        if sorted(order) != ids:
            raise SystemExit(f"{out_path} does not list every node id exactly once")
    else:
        jump = float(args["--jump-cost"]) if "--jump-cost" in args else default_jump(nodes, adj)
        seconds = float(args.get("--seconds", 10))
        print(f"{len(ids)} nodes, jump cost {jump:.1f} px, searching {seconds:.0f} s ...")
        D = build_costs(ids, nodes, adj, jump)
        result = solve(D, start_index(args, nodes, ids), seconds, random.Random(0))
        order = [ids[k] for k in result]
        Path(out_path).write_text(json.dumps(order), encoding="utf-8")
        print(f"wrote {out_path}")
    report(order, nodes, adj, cm_per_px)
    run_viewer(data, nodes, adj, order, out_path, args)


if __name__ == "__main__":
    main()
