"""Virtual board: watch Afrikan tahti games on the traced board image.

    pip install pygame numpy
    python viewer.py                                  # 5 greedy seats
    python viewer.py --model models/afrikan_ppo.zip   # seat 0 = trained model, others greedy
    python viewer.py --seats model,greedy,random,greedy,greedy --seed 7
    python viewer.py --check 300                      # no window: play 300 games, stop at the
                                                      # first rule/invariant violation

The window shows what the NEXT action of the current player will be (chosen
before it is executed, so what you see is what happens), with the model's
probability for every legal move.

Keys
  space run / pause        n or Right: one action        [ ]  slower / faster
  r next seed, R same seed v debug view <-> LED view (exact colours of the real LEDs)
  t reveal tokens          1 2 3 overlay distValue / distToStart / heatValue, 0 off
  a air links              l names                       + -  zoom   (window is resizable)
  Esc quit.  Hover a node to see its values.
"""
import argparse
import colorsys
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pygame

from game import Game
from led_frame import PLAYER_COLORS, render_frame

PANEL_W = 360
TOKEN_LETTER = {"Star": "S", "Ruby": "R", "Smaragd": "E", "Topaz": "T",
                "Horseshoe": "H", "Bandit": "B", "Blank": "-"}
TOKEN_COLOR = {"Star": (255, 230, 0), "Ruby": (255, 60, 60), "Smaragd": (60, 220, 90),
               "Topaz": (255, 150, 30), "Horseshoe": (190, 190, 190), "Bandit": (150, 60, 200),
               "Blank": (255, 255, 255)}
NODE_COLOR = {"road": (200, 120, 40), "sea": (60, 140, 255), "city": (255, 230, 60),
              "fork": (60, 230, 230), "bandit": (255, 70, 70)}
NODE_RADIUS = {"road": 3, "sea": 3, "city": 7, "fork": 5, "bandit": 6}


def short(node):
    return node.name if node.kind in ("city", "fork", "bandit") else f"{node.kind[0]}{node.id}"


# ----------------------------------------------------------------- policies
class Decision:
    def __init__(self, slot, action, legal, probs):
        self.slot, self.action, self.legal, self.probs = slot, action, legal, probs


class RandomPolicy:
    name = "random"

    def choose(self, g, p, legal):
        return random.choice(legal).slot, None


class GreedyPolicy:
    name = "greedy"

    def __init__(self):
        try:
            from env import greedy_opponent
        except ImportError as exc:
            raise SystemExit(f"greedy needs env.py (and gymnasium): {exc}")
        self.fn = greedy_opponent

    def choose(self, g, p, legal):
        return self.fn(g, p), None


class ModelPolicy:
    name = "model"

    def __init__(self, path, sample=False):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(path)
        self.deterministic = not sample

    def choose(self, g, p, legal):
        obs = g.observe(p, legal)
        mask = g.action_mask(legal)
        probs = None
        try:                                    # probability of every legal slot
            tensor, _ = self.model.policy.obs_to_tensor(obs)
            dist = self.model.policy.get_distribution(tensor, action_masks=mask)
            pr = dist.distribution.probs[0].detach().cpu().numpy()
            probs = {i + 1: float(v) for i, v in enumerate(pr) if mask[i]}
        except Exception:
            pass
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action) + 1, probs


def build_policies(args):
    seats = args.seats.split(",") if args.seats else (
        ["model"] + ["greedy"] * (args.players - 1) if args.model else ["greedy"] * args.players)
    seats = (seats + ["greedy"] * args.players)[:args.players]
    cache, policies = {}, []
    for s in seats:
        if s not in cache:
            if s == "model":
                if not args.model:
                    raise SystemExit("seat 'model' needs --model path")
                cache[s] = ModelPolicy(args.model, args.sample)
            elif s == "greedy":
                cache[s] = GreedyPolicy()
            elif s == "random":
                cache[s] = RandomPolicy()
            else:
                raise SystemExit(f"unknown seat type {s!r} (model, greedy, random)")
        policies.append(cache[s])
    return policies


# --------------------------------------------------------------- invariants
def check_invariants(g):
    """Things that must always be true. Returns a list of problem descriptions."""
    problems = []
    b = g.board
    on_board = Counter(n.token for n in b.get_nodes_with_tokens())
    if on_board != Counter(b.available_tokens):
        problems.append(f"tokens on the board {dict(on_board)} != available_tokens "
                        f"{dict(Counter(b.available_tokens))}")
    if ("Star" in b.available_tokens) == g.star_found:
        problems.append(f"star_found={g.star_found} but Star in available_tokens="
                        f"{'Star' in b.available_tokens}")
    for p in g.players:
        node = b.get_node_by_id(p.location)
        if node is None:
            problems.append(f"P{p.id} is on a missing node {p.location}")
            continue
        if p.money < 0:
            problems.append(f"P{p.id} has negative money {p.money}")
        if not 0 <= p.moves <= 6:
            problems.append(f"P{p.id} has moves={p.moves}")
        if p.stun < 0:
            problems.append(f"P{p.id} has stun={p.stun}")
        if p.available_to_win and not g.star_found:
            problems.append(f"P{p.id} can win but the star has not been found")
        if p.paid_sea and p.free_sea:
            problems.append(f"P{p.id} is both paid_sea and free_sea")
        if node.type == 3 and (p.paid_sea or p.free_sea):
            problems.append(f"P{p.id} is still sailing while in city {node.name}")
    if g.winner is not None:
        w = g.players[g.winner]
        if not (w.available_to_win and w.location in g.start_ids):
            problems.append(f"winner P{w.id} is not ready or not at a start city")
    return problems


# ------------------------------------------------------------------ helpers
def heat_color(t):
    r, g, b = colorsys.hsv_to_rgb(0.66 * (1 - max(0.0, min(1.0, t))), 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def arc_points(a, b, steps=16):
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    cx, cy = mx - dy * 0.25, my + dx * 0.25
    return [((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * cx + t ** 2 * b[0],
             (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * cy + t ** 2 * b[1])
            for t in (i / steps for i in range(steps + 1))]


def lerp(a, b, t):
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


# ---------------------------------------------------------------------- app
class App:
    def __init__(self, args):
        self.args = args
        self.game = Game(args.players)
        if args.map:
            self.game.board.setup(args.map)
        self.policies = build_policies(args)
        self.seed = args.seed
        self.speed = args.speed
        self.view = "debug"
        self.reveal, self.overlay, self.show_air, self.labels = True, 0, False, True
        self.reset(self.seed)

    # ---- game flow
    def reset(self, seed):
        self.seed = seed
        self.game.reset(seed)
        self.log = []
        self.popups = []
        self.pending = None
        self.anim = None
        self.running = False
        self.problems = []
        self.next_time = 0.0
        self.note(f"seed {seed} started")

    def note(self, text):
        self.log.append(f"t{self.game.turn:<3d} {text}")
        del self.log[:-200]
        if self.args.verbose:
            print(self.log[-1])

    def over(self):
        return self.game.winner is not None or self.game.turn >= self.args.max_turns

    def make_pending(self):
        g = self.game
        p = g.players[g.current]
        legal = g.legal_actions(p)
        slot, probs = self.policies[g.current].choose(g, p, legal)
        action = next(a for a in legal if a.slot == slot)
        self.pending = Decision(slot, action, legal, probs)

    def apply_pending(self):
        g, pend = self.game, self.pending
        actor = g.current
        p = g.players[actor]
        action = pend.action
        turn0, stun0 = g.turn, [q.stun for q in g.players]
        info = g.step(pend.slot)
        self.pending = None
        board = g.board
        who = f"P{p.id}"
        if info["kind"] == "move":
            a, b = board.nodes[info["from"]], board.nodes[info["to"]]
            route = {0: "road", 1: "sea", 3: "AIR"}[action.route]
            extra = f" (-{action.cost})" if action.cost else ""
            self.note(f"{who} {route} {short(a)} -> {short(b)}{extra}")
            dur = min(0.3, 0.6 / self.speed)
            self.anim = {"player": actor, "a": info["from"], "b": info["to"],
                         "t0": time.time(), "dur": dur, "air": action.route == 3}
        elif info["kind"] == "stay":
            self.note(f"{who} stays")
        else:
            node = board.nodes[p.location]
            if info["token"] is None:
                self.note(f"{who} {info['kind']} at {short(node)}: no luck")
                self.popup(node, "no luck", (200, 200, 200))
            else:
                self.note(f"{who} {info['kind']} {short(node)}: {info['token']} ({info['money_delta']:+d})")
                self.popup(node, f"{info['token']} {info['money_delta']:+d}", TOKEN_COLOR[info["token"]])
        if info["won"]:
            self.note(f"*** {who} WINS ***")
        else:                                           # who got skipped when the turn passed on
            n = len(g.players)
            if g.current == actor:
                skipped = [i for i in range(n) if i != actor] if g.turn != turn0 else []
            else:
                skipped = [(actor + k) % n for k in range(1, (g.current - actor) % n)]
            for i in skipped:
                q = g.players[i]
                self.note(f"P{q.id} skipped ({'stunned' if stun0[i] > q.stun else 'stuck at a bandit'})")
        self.problems = [] if self.args.no_checks else check_invariants(g)
        for text in self.problems:
            self.note(f"!! {text}")
        if self.problems:
            self.running = False
        if self.over():
            self.running = False

    def popup(self, node, text, color):
        self.popups.append({"node": node.id, "text": text, "color": color, "t0": time.time()})

    # ---- drawing
    def node_pos(self, node_id, scale):
        n = self.game.board.nodes[node_id]
        return n.x * scale, n.y * scale

    def draw_board(self, screen, bg, font, scale, mouse, now):
        g = self.game
        board = g.board
        screen.blit(bg, (0, 0))
        led_view = self.view == "led"
        if led_view:
            dark = pygame.Surface(bg.get_size())
            dark.set_alpha(190)
            screen.blit(dark, (0, 0))
        for n in board.nodes.values():                                  # links
            for t, route, _ in n.neighbors:
                if t < n.id or (route == 3 and not self.show_air):
                    continue
                col = {0: (90, 70, 40), 1: (40, 80, 140), 3: (170, 60, 170)}[route]
                pygame.draw.line(screen, col, self.node_pos(n.id, scale), self.node_pos(t, scale), 1)
        if led_view:
            frame = render_frame(g, now, 1.0)
            glow = pygame.Surface(bg.get_size(), pygame.SRCALPHA)
            for n in board.nodes.values():
                c = frame[n.id]
                x, y = self.node_pos(n.id, scale)
                if max(c) > 20:
                    pygame.draw.circle(glow, (*c, 70), (x, y), 9)
                pygame.draw.circle(screen, c, (x, y), 4)
            screen.blit(glow, (0, 0))
            return
        values = None
        if self.overlay:
            attr = {1: "distValue", 2: "distToStart", 3: "heatValue"}[self.overlay]
            values = {i: getattr(n, attr) for i, n in board.nodes.items()}
            top = max(max(values.values()), 1)
        for n in board.nodes.values():
            x, y = self.node_pos(n.id, scale)
            r = max(2, int(NODE_RADIUS[n.kind] * min(1.5, scale)))
            color = heat_color(values[n.id] / top) if values else NODE_COLOR[n.kind]
            pygame.draw.circle(screen, color, (x, y), r)
            pygame.draw.circle(screen, (0, 0, 0), (x, y), r, 1)
            if n.id in g.start_ids:
                pygame.draw.circle(screen, (60, 255, 60), (x, y), r + 4, 2)
            if n.token:
                if self.reveal:
                    pygame.draw.rect(screen, TOKEN_COLOR[n.token], (x + r, y - r - 12, 12, 12))
                    screen.blit(font.render(TOKEN_LETTER[n.token], True, (0, 0, 0)), (x + r + 2, y - r - 12))
                else:
                    pygame.draw.circle(screen, (255, 0, 255), (x, y), r + 3, 2)
            if self.labels and n.kind in ("city", "fork", "bandit"):
                screen.blit(font.render(n.name, True, (0, 0, 0), (255, 255, 210)), (x + r + 2, y + 2))
        self.draw_decision(screen, font, scale)
        self.draw_players(screen, font, scale, now)

    def draw_decision(self, screen, font, scale):
        pend = self.pending
        if pend is None or self.anim is not None:
            return
        g = self.game
        here = self.node_pos(g.players[g.current].location, scale)
        for a in pend.legal:
            if a.kind != "move":
                continue
            pos = self.node_pos(a.target, scale)
            chosen = a.slot == pend.slot
            prob = pend.probs.get(a.slot) if pend.probs else None
            radius = 8 + (int(16 * prob) if prob is not None else 0)
            color = (255, 255, 255) if chosen else (120, 200, 255)
            pygame.draw.circle(screen, color, pos, radius, 3 if chosen else 1)
            if prob is not None and (prob >= 0.03 or chosen):
                screen.blit(font.render(f"{prob:.2f}", True, (0, 0, 0), (255, 255, 255)),
                            (pos[0] + radius, pos[1] - radius - 8))
            if chosen:
                if a.route == 3:
                    pygame.draw.lines(screen, color, False, arc_points(here, pos), 2)
                else:
                    pygame.draw.line(screen, color, here, pos, 2)

    def draw_players(self, screen, font, scale, now):
        g = self.game
        anim = self.anim
        shown = {}
        for i, p in enumerate(g.players):
            if anim and anim["player"] == i:
                t = min(1.0, (now - anim["t0"]) / anim["dur"])
                a, b = self.node_pos(anim["a"], scale), self.node_pos(anim["b"], scale)
                if anim["air"]:
                    pts = arc_points(a, b)
                    pos = pts[min(len(pts) - 1, int(t * (len(pts) - 1)))]
                else:
                    pos = lerp(a, b, t)
                shown[i] = (pos, None)
            else:
                shown.setdefault(("n", p.location), []).append(i)
        for key, val in list(shown.items()):
            if isinstance(key, tuple):
                base = self.node_pos(key[1], scale)
                for k, i in enumerate(val):
                    ang = 2 * math.pi * k / len(val)
                    off = 0 if len(val) == 1 else 8
                    self.draw_player(screen, font, i, (base[0] + off * math.cos(ang), base[1] + off * math.sin(ang)))
            else:
                self.draw_player(screen, font, key, val[0])
        for pop in list(self.popups):
            age = now - pop["t0"]
            if age > 1.6:
                self.popups.remove(pop)
                continue
            x, y = self.node_pos(pop["node"], scale)
            surf = font.render(pop["text"], True, pop["color"], (0, 0, 0))
            surf.set_alpha(int(255 * (1 - age / 1.6)))
            screen.blit(surf, (x + 6, y - 30 - 24 * age))

    def draw_player(self, screen, font, i, pos):
        g = self.game
        p = g.players[i]
        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        pos = (int(pos[0]), int(pos[1]))
        if i == g.current:
            pygame.draw.circle(screen, (255, 255, 255), pos, 11, 2)
        if p.available_to_win:
            pygame.draw.circle(screen, (255, 215, 0), pos, 9, 2)
        pygame.draw.circle(screen, tuple(c // 3 for c in color) if p.stun else color, pos, 7)
        pygame.draw.circle(screen, (0, 0, 0), pos, 7, 1)
        screen.blit(font.render(str(p.id), True, (0, 0, 0)), (pos[0] - 3, pos[1] - 6))

    def draw_panel(self, screen, font, x0, height, hover):
        g = self.game
        board = g.board
        pygame.draw.rect(screen, (24, 24, 28), (x0, 0, PANEL_W, height))
        y = 6
        line_h = font.get_linesize()

        def put(text, color=(220, 220, 220), bg=None):
            nonlocal y
            screen.blit(font.render(text, True, color, bg), (x0 + 8, y))
            y += line_h

        state = "RUN" if self.running else "pause"
        put(f"seed {self.seed}  turn {g.turn}  {state}  {self.speed:.1f}/s  [{self.view}]")
        put(f"star {'FOUND' if g.star_found else 'hidden'}   cape town {'taken' if g.cape_town_claimed else 'open'}")
        left = Counter(board.available_tokens)
        put("tokens " + " ".join(f"{TOKEN_LETTER[k]}{v}" for k, v in left.items()) + f"  ({sum(left.values())})")
        y += 4
        for i, p in enumerate(g.players):
            node = board.nodes[p.location]
            flags = ("R" if p.available_to_win else "") + (f"Z{p.stun}" if p.stun else "") + \
                    ("B" if p.bandit_stuck else "") + ("$" if p.paid_sea else "") + ("~" if p.free_sea else "")
            mark = ">" if i == g.current else " "
            put(f"{mark}P{p.id} {self.policies[i].name:6s} ${p.money:<5d} mv{p.moves} {flags:<4s} {short(node)}",
                PLAYER_COLORS[i % len(PLAYER_COLORS)], (60, 60, 70) if i == g.current else None)
        y += 4
        for text in self.problems:
            put("!! " + text[:46], (255, 80, 80))
        if g.winner is not None:
            put(f"*** P{g.players[g.winner].id} WINS (turn {g.turn}) ***", (120, 255, 120))
        elif g.turn >= self.args.max_turns:
            put(f"turn limit {self.args.max_turns} reached", (255, 200, 80))
        elif self.pending is not None:
            p = g.players[g.current]
            here = board.nodes[p.location]
            goal = "distToStart" if p.available_to_win else "distValue"
            put(f"next: P{p.id}  moves {p.moves}  goal {goal}", (255, 255, 255))
            for a in self.pending.legal:
                target = short(board.nodes[a.target]) if a.target is not None else ""
                delta = f"d{getattr(here, goal) - getattr(a, goal):+d}" if a.kind == "move" else ""
                prob = f"{self.pending.probs[a.slot]:.2f}" if self.pending.probs and a.slot in self.pending.probs else ""
                chosen = a.slot == self.pending.slot
                put(f"{'>' if chosen else ' '}{a.slot:2d} {a.kind[:9]:9s} {target[:11]:11s} "
                    f"r{a.route if a.route is not None else '-'} ${a.cost:<3d} {delta:>3s} {prob}",
                    (255, 255, 255) if chosen else (170, 170, 170))
        y += 4
        room = (height - y) // line_h - 3
        for text in self.log[-max(room, 0):]:
            put(text[:52], (150, 200, 150))
        yb = height - 2 * line_h - 4
        if hover is not None:
            n = board.nodes[hover]
            screen.blit(font.render(f"#{n.id} {short(n)} {n.kind} dV{n.distValue} dS{n.distToStart} "
                                    f"heat{n.heatValue} {n.token}", True, (255, 255, 120)), (x0 + 8, yb))
        screen.blit(font.render("space run  n step  [ ] speed  r seed  v view  t tokens", True,
                                (120, 120, 120)), (x0 + 8, yb + line_h))

    # ---- main loop
    def run(self):
        pygame.init()
        board = self.game.board
        meta = board.map_meta
        image = self.load_image(meta)
        w, h = image.get_size()
        desk = pygame.display.Info()
        scale = min(self.args.scale, (desk.current_w - 40 - PANEL_W) / w, (desk.current_h - 120) / h)

        def make_window(sc):
            return pygame.display.set_mode((int(w * sc) + PANEL_W, max(int(h * sc), 600)), pygame.RESIZABLE)

        screen = make_window(scale)
        pygame.display.set_caption("Afrikan tahti viewer")
        font = pygame.font.SysFont("dejavusansmono,consolas,menlo,couriernew,monospace", 13)
        image = image.convert()

        def make_bg():
            return pygame.transform.smoothscale(image, (max(1, int(w * scale)), max(1, int(h * scale))))

        bg = make_bg()
        clock = pygame.time.Clock()
        running = True
        while running:
            now = time.time()
            mouse = pygame.mouse.get_pos()
            hover = None
            for n in board.nodes.values():
                if math.hypot(n.x * scale - mouse[0], n.y * scale - mouse[1]) < 8:
                    hover = n.id
                    break
            step_once = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    scale = max(0.3, min((ev.w - PANEL_W) / w, ev.h / h))
                    bg = make_bg()
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_SPACE:
                        self.running = not self.running and not self.over()
                    elif k in (pygame.K_n, pygame.K_RIGHT):
                        step_once = True
                    elif k == pygame.K_LEFTBRACKET:
                        self.speed = max(0.5, self.speed / 1.5)
                    elif k == pygame.K_RIGHTBRACKET:
                        self.speed = min(300.0, self.speed * 1.5)
                    elif k == pygame.K_r:
                        self.reset(self.seed if ev.mod & pygame.KMOD_SHIFT else self.seed + 1)
                    elif k == pygame.K_v:
                        self.view = "led" if self.view == "debug" else "debug"
                    elif k == pygame.K_t:
                        self.reveal = not self.reveal
                    elif k == pygame.K_a:
                        self.show_air = not self.show_air
                    elif k == pygame.K_l:
                        self.labels = not self.labels
                    elif k in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3):
                        self.overlay = k - pygame.K_0
                    elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
                               pygame.K_MINUS, pygame.K_KP_MINUS):
                        up = k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS)
                        scale = max(0.3, min(6.0, scale * (1.1 if up else 1 / 1.1)))
                        screen = make_window(scale)
                        bg = make_bg()
            if self.anim and now - self.anim["t0"] >= self.anim["dur"]:
                self.anim = None
            if not self.over() and self.pending is None:
                self.make_pending()
            go = self.pending is not None and not self.over()
            if go and step_once:
                self.anim = None
                self.apply_pending()
            elif go and self.running and self.anim is None and now >= self.next_time:
                self.apply_pending()
                self.next_time = now + 1.0 / self.speed
            screen.fill((24, 24, 28))
            self.draw_board(screen, bg, font, scale, mouse, now)
            self.draw_panel(screen, font, int(w * scale), screen.get_height(), hover)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

    def load_image(self, meta):
        candidates = [self.args.image, meta.get("image")]
        for c in candidates:
            if c and Path(c).exists():
                return pygame.image.load(c)
        nodes = self.game.board.nodes.values()
        size = meta.get("size") or [int(max(n.x for n in nodes)) + 30, int(max(n.y for n in nodes)) + 30]
        print("board image not found (use --image); drawing on a blank board")
        surf = pygame.Surface(tuple(size))
        surf.fill((70, 70, 78))
        return surf


# -------------------------------------------------------------- headless check
def check_mode(args):
    g = Game(args.players)
    if args.map:
        g.board.setup(args.map)
    policies = build_policies(args)
    wins, turns, unfinished = Counter(), [], 0
    for i in range(args.check):
        seed = args.seed + i
        g.reset(seed)
        actions = 0
        while g.winner is None and g.turn < args.max_turns:
            p = g.players[g.current]
            legal = g.legal_actions(p)
            slot, _ = policies[g.current].choose(g, p, legal)
            g.step(slot)
            actions += 1
            problems = check_invariants(g)
            if problems:
                print(f"VIOLATION seed {seed}, action {actions}, turn {g.turn}:")
                for text in problems:
                    print("  -", text)
                print(f"replay: python viewer.py --seed {seed} (pause, step with n)")
                return 1
        if g.winner is None:
            unfinished += 1
        else:
            wins[g.winner] += 1
            turns.append(g.turn)
    mean = sum(turns) / len(turns) if turns else 0
    print(f"{args.check} games, no violations. finished {len(turns)}, hit turn limit {unfinished}, "
          f"mean turns {mean:.1f}, wins per seat {dict(sorted(wins.items()))}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, default=5)
    ap.add_argument("--seats", help="comma list of model/greedy/random, one per seat")
    ap.add_argument("--model", help="path to a MaskablePPO .zip")
    ap.add_argument("--sample", action="store_true", help="model samples instead of argmax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", help="map.json (default: next to board.py)")
    ap.add_argument("--image", help="board image (default: the one stored in map.json)")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--speed", type=float, default=4.0, help="actions per second")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--check", type=int, default=0, help="headless: play N games and check invariants")
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="also print the event log")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    if args.check:
        sys.exit(check_mode(args))
    App(args).run()
