from dataclasses import dataclass

@dataclass(frozen = False)
class Action:
    slot: int          # index in the fixed action space (0..12)
    kind: str          # "stay" | "open_fee" | "open_gamble" | "move"
    target: int | None # node id for moves
    route: int | None  # 0 road, 1 sea, 3 air
    cost: int          # money
    moves_used: int    # moves consumed (air: all remaining)
    distValue: int      # distance value of the target node
    heatValue: int      # heat value of the target node
    distToStart: int    # distance to start of the target node
