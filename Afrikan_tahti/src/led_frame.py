"""State -> LED colours. This is the logic the ESP32 renderer will implement.

render_frame(game, t) returns one (r, g, b) per node; index == node id == LED
index. The viewer shows exactly these colours in its "LED view", so what you
tune here is what the physical board will show.
"""
PLAYER_COLORS = [(255, 60, 60), (60, 160, 255), (60, 255, 90), (255, 200, 40), (210, 80, 255)]

TERRAIN = {                    # dim on purpose: LEDs are bright at full value
    "road": (45, 26, 6),
    "sea": (0, 18, 55),
    "city": (80, 58, 0),
    "fork": (0, 50, 50),
    "bandit": (85, 0, 0),
}
START_CITY = (0, 90, 0)
TOKEN_PRESENT = (130, 130, 130)      # a closed token is shown, never its content


def _scale(color, k):
    return tuple(int(c * k) for c in color)


def render_frame(game, t: float = 0.0, brightness: float = 1.0):
    """t: seconds, only used for blinking. brightness: 0..1 global cap."""
    board = game.board
    frame = []
    for node in board.nodes.values():
        if node.id in game.start_ids:
            color = START_CITY
        elif node.token:
            color = TOKEN_PRESENT
        else:
            color = TERRAIN[node.kind]
        frame.append(color)

    at_node = {}
    for p in game.players:
        at_node.setdefault(p.location, []).append(p)
    current = game.players[game.current]
    for location, players in at_node.items():
        p = players[int(t * 2) % len(players)]          # players sharing a node take turns
        color = PLAYER_COLORS[(p.id - 1) % len(PLAYER_COLORS)]
        if p.stun > 0:
            color = _scale(color, 0.35)
        elif p.available_to_win and int(t * 3) % 2 == 0:
            color = (255, 255, 255)                     # can win: flashes white
        if p is current and int(t * 4) % 2 == 1:
            color = _scale(color, 0.3)                  # whose turn it is: blinks
        frame[location] = color

    return [_scale(c, brightness) for c in frame]
