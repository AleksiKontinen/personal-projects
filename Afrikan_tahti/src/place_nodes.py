"""Board tracer: click on the board image to build the node graph.

    pip install pygame
    python place_nodes.py board.jpg [board_nodes.json] [--scale 1.5] [--target 225]

Everything is saved to the JSON after every change, and the file is reloaded
on start, so you can quit and continue any time.

Nodes
  left click on empty space  place a node of the current kind and connect it
                             to the active node (the chain) with the current route
  left click on a node       connect the active node to it (or select it)
  space / Esc                break the chain (next click starts a new one)
  right-drag                 move a node
  Delete                     delete the node under the mouse
  x                          remove the edge between active node and hovered node
  u                          undo
Kind of the NEXT placed node (falls back to segment after each placement)
  c city (asks for a name, type it in the window, Enter)   f fork   b bandit
  s segment (default, one LED between two named nodes)
Route of new edges
  1 road   2 sea   3 air (air only links existing nodes: click two cities)
Other
  n  rename hovered node      h  toggle "start city" on hovered node
  l  toggle name labels
"""
import json
import math
import sys
from pathlib import Path

import pygame

ROUTE_NAMES = {0: "road", 1: "sea", 3: "air"}
ROUTE_COLORS = {0: (230, 140, 40), 1: (70, 150, 255), 3: (230, 70, 230)}
KIND_COLORS = {"segment": (255, 255, 255), "city": (255, 230, 60),
               "fork": (60, 230, 230), "bandit": (255, 70, 70)}
KIND_RADIUS = {"segment": 4, "city": 8, "fork": 6, "bandit": 6}
HIT_RADIUS = 9


class Tracer:
    def __init__(self, image_path, out_path, scale=1.5, target=225):
        self.image_path = image_path
        self.out_path = Path(out_path)
        self.scale = scale
        self.target = target
        self.nodes = {}      # id -> {"name", "kind", "x", "y", "start"}
        self.edges = []      # {"a", "b", "route"}  with a < b
        self.next_id = 0
        self.active = None
        self.kind = "segment"
        self.route = 0
        self.typing = None   # {"id", "buf"} while a name is being typed
        self.show_labels = True
        self.history = []
        self.size = (0, 0)
        if self.out_path.exists():
            self.load()

    # ---------------------------------------------------------- data model
    def state(self):
        return {"image": str(self.image_path), "size": list(self.size),
                "next_id": self.next_id,
                "nodes": [dict(id=i, **n) for i, n in sorted(self.nodes.items())],
                "edges": list(self.edges)}

    def load(self):
        data = json.loads(self.out_path.read_text(encoding="utf-8"))
        self.nodes = {n["id"]: {k: v for k, v in n.items() if k != "id"} for n in data["nodes"]}
        self.edges = data["edges"]
        self.next_id = data.get("next_id", max(self.nodes, default=-1) + 1)

    def save(self):
        self.out_path.write_text(json.dumps(self.state(), indent=1), encoding="utf-8")

    def push_history(self):
        self.history.append(json.dumps({"nodes": self.nodes, "edges": self.edges,
                                        "next_id": self.next_id}))
        del self.history[:-200]

    def undo(self):
        if not self.history:
            return
        snap = json.loads(self.history.pop())
        self.nodes = {int(k): v for k, v in snap["nodes"].items()}
        self.edges, self.next_id = snap["edges"], snap["next_id"]
        self.active = None
        self.save()

    def count_kind(self, kind):
        return sum(1 for n in self.nodes.values() if n["kind"] == kind)

    def auto_name(self, kind):
        k = 1
        names = {n["name"] for n in self.nodes.values()}
        while f"{kind}{k}" in names:
            k += 1
        return f"{kind}{k}"

    def add_node(self, x, y, kind):
        self.push_history()
        nid = self.next_id
        self.next_id += 1
        name = "" if kind in ("segment", "city") else self.auto_name(kind)
        self.nodes[nid] = {"name": name, "kind": kind, "x": round(x, 1), "y": round(y, 1),
                           "start": False}
        return nid

    def add_edge(self, a, b):
        if a == b:
            return
        a, b = min(a, b), max(a, b)
        edge = {"a": a, "b": b, "route": self.route}
        if edge not in self.edges:
            self.push_history()
            self.edges.append(edge)

    def remove_edge(self, a, b):
        a, b = min(a, b), max(a, b)
        keep = [e for e in self.edges if not (e["a"] == a and e["b"] == b)]
        if len(keep) != len(self.edges):
            self.push_history()
            self.edges = keep

    def delete_node(self, nid):
        self.push_history()
        del self.nodes[nid]
        self.edges = [e for e in self.edges if nid not in (e["a"], e["b"])]
        if self.active == nid:
            self.active = None

    # ------------------------------------------------------------ geometry
    def to_screen(self, n):
        return int(n["x"] * self.scale), int(n["y"] * self.scale)

    def hit(self, mx, my):
        best, best_d = None, HIT_RADIUS
        for nid, n in self.nodes.items():
            sx, sy = self.to_screen(n)
            d = math.hypot(sx - mx, sy - my)
            if d <= best_d:
                best, best_d = nid, d
        return best

    # --------------------------------------------------------------- input
    def click(self, mx, my):
        hit = self.hit(mx, my)
        if hit is not None:
            if self.active is not None and hit != self.active:
                self.add_edge(self.active, hit)
            self.active = hit
            self.save()
            return
        if self.route == 3:
            return                          # air links only existing nodes
        nid = self.add_node(mx / self.scale, my / self.scale, self.kind)
        if self.active is not None:
            self.add_edge(self.active, nid)
        self.active = nid
        if self.kind == "city":
            self.typing = {"id": nid, "buf": ""}
        self.kind = "segment"
        self.save()

    def key(self, event, mouse):
        if self.typing is not None:
            self.key_typing(event)
            return
        k = event.key
        hover = self.hit(*mouse)
        if k in (pygame.K_SPACE, pygame.K_ESCAPE):
            self.active = None
        elif k == pygame.K_c:
            self.kind = "city"
        elif k == pygame.K_f:
            self.kind = "fork"
        elif k == pygame.K_b:
            self.kind = "bandit"
        elif k == pygame.K_s:
            self.kind = "segment"
        elif k == pygame.K_1:
            self.route = 0
        elif k == pygame.K_2:
            self.route = 1
        elif k == pygame.K_3:
            self.route = 3
        elif k == pygame.K_l:
            self.show_labels = not self.show_labels
        elif k == pygame.K_u:
            self.undo()
        elif k == pygame.K_x and hover is not None and self.active is not None:
            self.remove_edge(self.active, hover)
            self.save()
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE) and hover is not None:
            self.delete_node(hover)
            self.save()
        elif k == pygame.K_n and hover is not None:
            self.typing = {"id": hover, "buf": self.nodes[hover]["name"]}
        elif k == pygame.K_h and hover is not None:
            self.push_history()
            self.nodes[hover]["start"] = not self.nodes[hover]["start"]
            self.save()

    def key_typing(self, event):
        t = self.typing
        if event.key == pygame.K_RETURN:
            name = t["buf"].strip() or self.auto_name("city")
            if name in {n["name"] for i, n in self.nodes.items() if i != t["id"]}:
                name += "_dup"
            self.nodes[t["id"]]["name"] = name
            self.typing = None
            self.save()
        elif event.key == pygame.K_ESCAPE:
            self.typing = None
        elif event.key == pygame.K_BACKSPACE:
            t["buf"] = t["buf"][:-1]
        elif event.unicode and event.unicode.isprintable():
            t["buf"] += event.unicode

    # ------------------------------------------------------------- drawing
    def draw(self, screen, bg, font, mouse):
        screen.blit(bg, (0, 0))
        for e in self.edges:
            pa, pb = self.to_screen(self.nodes[e["a"]]), self.to_screen(self.nodes[e["b"]])
            pygame.draw.line(screen, ROUTE_COLORS[e["route"]], pa, pb, 2 if e["route"] != 3 else 1)
        hover = self.hit(*mouse)
        for nid, n in self.nodes.items():
            p = self.to_screen(n)
            pygame.draw.circle(screen, KIND_COLORS[n["kind"]], p, KIND_RADIUS[n["kind"]])
            pygame.draw.circle(screen, (0, 0, 0), p, KIND_RADIUS[n["kind"]], 1)
            if n["start"]:
                pygame.draw.circle(screen, (60, 255, 60), p, KIND_RADIUS[n["kind"]] + 4, 2)
            if nid == self.active:
                pygame.draw.circle(screen, (255, 255, 255), p, KIND_RADIUS[n["kind"]] + 6, 2)
            if nid == hover:
                pygame.draw.circle(screen, (255, 0, 255), p, KIND_RADIUS[n["kind"]] + 2, 1)
            if n["name"] and (self.show_labels or nid == hover):
                screen.blit(font.render(n["name"], True, (0, 0, 0), (255, 255, 200)),
                            (p[0] + 7, p[1] - 14))
        total = len(self.nodes)
        lines = [f"nodes {total}/{self.target}  cities {self.count_kind('city')}  "
                 f"forks {self.count_kind('fork')}  bandits {self.count_kind('bandit')}  "
                 f"segments {self.count_kind('segment')}",
                 f"next kind: {self.kind}   route: {ROUTE_NAMES[self.route]}   "
                 f"starts: {sum(n['start'] for n in self.nodes.values())}"]
        if self.typing is not None:
            lines.append(f"NAME: {self.typing['buf']}_   (Enter = ok, blank = auto)")
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, (255, 255, 255), (0, 0, 0)), (4, 4 + i * 18))


def main():
    args = sys.argv[1:]
    scale, target = 1.5, 225
    for flag in ("--scale", "--target"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            if flag == "--scale":
                scale = float(val)
            else:
                target = int(val)
    if not args:
        print(__doc__)
        return
    image = args[0]
    out = args[1] if len(args) > 1 else "board_nodes.json"

    pygame.init()
    img = pygame.image.load(image)
    w, h = img.get_size()
    tracer = Tracer(image, out, scale, target)
    tracer.size = (w, h)
    screen = pygame.display.set_mode((int(w * scale), int(h * scale)))
    pygame.display.set_caption("Afrikan tahti board tracer")
    bg = pygame.transform.smoothscale(img, screen.get_size()).convert()
    font = pygame.font.SysFont(None, 18)
    drag = None
    clock = pygame.time.Clock()
    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                tracer.key(event, mouse)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and tracer.typing is None:
                    tracer.click(*event.pos)
                elif event.button == 3:
                    drag = tracer.hit(*event.pos)
                    if drag is not None:
                        tracer.push_history()
            elif event.type == pygame.MOUSEMOTION and drag is not None:
                tracer.nodes[drag]["x"] = round(event.pos[0] / scale, 1)
                tracer.nodes[drag]["y"] = round(event.pos[1] / scale, 1)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3 and drag is not None:
                drag = None
                tracer.save()
        tracer.draw(screen, bg, font, mouse)
        pygame.display.flip()
        clock.tick(60)
    tracer.save()
    pygame.quit()


if __name__ == "__main__":
    main()
