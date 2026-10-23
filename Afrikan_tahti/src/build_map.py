"""Build map.json (single source of truth for Python AND the C++ firmware)
from the traced board.

    python build_map.py board_nodes.json map.json [--old mapData.json] [--order order.json]

  --old    the previous mapData.json; prints which connections (city to city,
           route, number of steps) differ between the old map and the traced one
  --order  JSON list of tracer ids in LED-strip order. Without it, ids follow
           tracer creation order. Re-run with --order once the wiring is decided:
           node id == LED index, and the model is unaffected because neighbor
           order does not depend on ids.

Neighbor order (this is what the observation slots are based on):
  sorted by (route: road, sea, air), then by direction clockwise from "up"
  on the board image. It depends only on the geometry, not on ids or wiring.
"""
import json
import math
import sys
from collections import defaultdict

ROUTE_IDS = {"road": 0, "sea": 1, "air": 3}
ROUTE_NAMES = {v: k for k, v in ROUTE_IDS.items()}
REQUIRED_NAMES = ("Tangier", "Cairo", "Gold Coast", "Slave Coast", "Cape Town")
MAX_MOVE_SLOTS = 9          # the observation keeps 9 move slots


def build(traced, order=None):
    problems = []
    nodes = {n["id"]: n for n in traced["nodes"]}
    adj = defaultdict(list)
    for e in traced["edges"]:
        adj[e["a"]].append((e["b"], e["route"]))
        adj[e["b"]].append((e["a"], e["route"]))

    # ---- validation
    names = {}
    for nid, n in nodes.items():
        if n["kind"] != "segment":
            if not n["name"]:
                problems.append(f"node {nid} ({n['kind']}) at ({n['x']}, {n['y']}) has no name")
            elif n["name"] in names:
                problems.append(f"duplicate name {n['name']!r}")
            names[n["name"]] = nid
        if not adj[nid]:
            problems.append(f"{n['kind']} {n['name'] or nid} at ({n['x']}, {n['y']}) has no edges")
    for nid, n in nodes.items():
        if n["kind"] != "segment":
            continue
        routes = {r for _, r in adj[nid]}
        if len(adj[nid]) != 2 or len(routes) != 1 or 3 in routes:
            problems.append(f"segment {nid} at ({n['x']}, {n['y']}) must have exactly 2 edges "
                            f"of one road/sea route (has {[(o, ROUTE_NAMES[r]) for o, r in adj[nid]]})")
    for e in traced["edges"]:
        if e["route"] == 3 and (nodes[e["a"]]["kind"] == "segment" or nodes[e["b"]]["kind"] == "segment"):
            problems.append(f"air edge {e['a']}-{e['b']} touches a segment node")
    for name in REQUIRED_NAMES:
        if name not in names:
            problems.append(f"required node {name!r} is missing")
    starts = [nid for nid, n in nodes.items() if n.get("start")]
    if len(starts) != 2:
        problems.append(f"expected 2 start cities, found {len(starts)}")

    # ---- ids
    if order is None:
        order = sorted(nodes)
    if sorted(order) != sorted(nodes):
        raise SystemExit("--order must list every traced node id exactly once")
    new_id = {old: i for i, old in enumerate(order)}

    # ---- output nodes with canonical neighbor order
    out = []
    for old in order:
        n = nodes[old]
        kind = n["kind"]
        if kind == "segment":
            kind = ROUTE_NAMES[adj[old][0][1]] if adj[old] else "road"

        def key(item):
            other, route = item
            o = nodes[other]
            angle = math.atan2(o["x"] - n["x"], -(o["y"] - n["y"])) % (2 * math.pi)
            return (route, angle, new_id[other])

        neighbors = [[new_id[o], r] for o, r in sorted(adj[old], key=key)]
        move_slots = len(neighbors)
        if move_slots > MAX_MOVE_SLOTS:
            problems.append(f"{n['name'] or old} has {move_slots} neighbors (observation keeps {MAX_MOVE_SLOTS})")
        out.append({"id": new_id[old],
                    "name": n["name"] or f"{kind}_seg_{new_id[old]}",
                    "kind": kind, "x": n["x"], "y": n["y"], "neighbors": neighbors})

    result = {"meta": {"image": traced.get("image"), "size": traced.get("size"),
                       "nodes": len(out), "route_ids": ROUTE_IDS},
              "starts": sorted(new_id[s] for s in starts),
              "nodes": out}
    return result, problems, nodes, adj


def chains(nodes, adj):
    """Set of (nameA, nameB, route, steps): every city-to-city connection."""
    found = set()
    for a, n in nodes.items():
        if n["kind"] == "segment":
            continue
        for first, route in adj[a]:
            prev, cur, steps = a, first, 0
            while nodes[cur]["kind"] == "segment" and steps < 1000:
                steps += 1
                nxt = [o for o, _ in adj[cur] if o != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if nodes[cur]["kind"] == "segment":
                continue
            pair = tuple(sorted((n["name"], nodes[cur]["name"])))
            found.add((pair[0], pair[1], route, steps))
    return found


def old_chains(path):
    edges = json.load(open(path, encoding="utf-8"))
    found = set()
    for e in edges:
        pair = tuple(sorted((e["from"], e["to"])))
        found.add((pair[0], pair[1], ROUTE_IDS[e["type"].lower()],
                   0 if e["type"].lower() == "air" else int(e["distance"])))
    return found


def show_diff(new, old):
    only_old, only_new = sorted(old - new), sorted(new - old)
    fmt = lambda c: f"{c[0]} - {c[1]}  {ROUTE_NAMES[c[2]]}  steps={c[3]}"
    print(f"\nconnections: {len(new)} traced, {len(old)} in old map")
    if not only_old and not only_new:
        print("traced map matches the old map exactly")
        return
    for c in only_old:
        print("  only in OLD:", fmt(c))
    for c in only_new:
        print("  only in NEW:", fmt(c))


def main():
    args = sys.argv[1:]
    opts = {}
    for flag in ("--old", "--order"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i:i + 2]
    if len(args) < 2:
        print(__doc__)
        return
    traced = json.load(open(args[0], encoding="utf-8"))
    order = json.load(open(opts["--order"])) if "--order" in opts else None
    result, problems, nodes, adj = build(traced, order)

    kinds = defaultdict(int)
    for n in result["nodes"]:
        kinds[n["kind"]] += 1
    print(f"{len(result['nodes'])} nodes: {dict(kinds)}")
    degs = [len(n["neighbors"]) for n in result["nodes"]]
    print(f"max neighbors per node: {max(degs)}")

    if "--old" in opts:
        show_diff(chains(nodes, adj), old_chains(opts["--old"]))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        print("map.json NOT written")
        sys.exit(1)
    with open(args[1], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {args[1]}")


if __name__ == "__main__":
    main()
