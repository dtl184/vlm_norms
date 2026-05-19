"""
MazeNamo grid-world physics.

A Sokoban-style environment where the agent can push boxes.  The norm we
bake in is: the agent must push the REQUIRED box at least once before
reaching the goal.  Norm-violating agents ignore the box and go straight
to the goal (shorter path); norm-following agents always detour to push the
box first.

Coordinate convention
---------------------
(x, y) where x is column and y is row.
y increases **downward** (0 = top row).
Actions: 0=NORTH(y-1), 1=SOUTH(y+1), 2=EAST(x+1), 3=WEST(x-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
ACTION_NAMES = {NORTH: "NORTH", SOUTH: "SOUTH", EAST: "EAST", WEST: "WEST"}
DELTAS = {NORTH: (0, -1), SOUTH: (0, 1), EAST: (1, 0), WEST: (-1, 0)}


# ---------------------------------------------------------------------------
# Layout string → structured grid
# ---------------------------------------------------------------------------

DEFAULT_LAYOUT = [
    "########",
    "#......#",
    "#....G.#",   # goal      at (5, 2)
    "#......#",
    "#.B....#",   # req. box  at (2, 4)
    "#......#",
    "#....A.#",   # agent     at (5, 6)  [default start, varies at run-time]
    "########",
]

# Several starting positions used to generate a varied training set.
TRAINING_STARTS = [
    (5, 6), (4, 6), (6, 6), (3, 6),
    (5, 5), (4, 5), (6, 5), (3, 5),
    (6, 4), (5, 4),
]


@dataclass
class GridConfig:
    width: int
    height: int
    walls: frozenset
    goal: tuple[int, int]
    required_box_start: tuple[int, int]
    default_agent_start: tuple[int, int]

    @staticmethod
    def from_layout(rows: list[str]) -> "GridConfig":
        walls = set()
        goal = (0, 0)
        req_box = (0, 0)
        agent = (0, 0)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "#":
                    walls.add((x, y))
                elif ch == "G":
                    goal = (x, y)
                elif ch == "B":
                    req_box = (x, y)
                elif ch == "A":
                    agent = (x, y)
        height = len(rows)
        width = max(len(r) for r in rows)
        return GridConfig(
            width=width,
            height=height,
            walls=frozenset(walls),
            goal=goal,
            required_box_start=req_box,
            default_agent_start=agent,
        )


# ---------------------------------------------------------------------------
# Runtime simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    agent: tuple[int, int]
    box_pos: tuple[int, int]       # required box current position
    box_pushed: bool = False       # has box been moved from its start?

    def copy(self) -> "SimState":
        return SimState(self.agent, self.box_pos, self.box_pushed)


def step(cfg: GridConfig, sim: SimState, action: int) -> Optional[SimState]:
    """
    Apply *action* to *sim* and return the resulting SimState, or None if
    the move is illegal (hits a wall or unmovable box).
    """
    dx, dy = DELTAS[action]
    nx, ny = sim.agent[0] + dx, sim.agent[1] + dy

    if (nx, ny) in cfg.walls or not (0 <= nx < cfg.width and 0 <= ny < cfg.height):
        return None

    if (nx, ny) == sim.box_pos:
        # Attempt push
        bx, by = nx + dx, ny + dy
        if (bx, by) in cfg.walls or not (0 <= bx < cfg.width and 0 <= by < cfg.height):
            return None  # box blocked by wall
        # Successful push
        return SimState(
            agent=(nx, ny),
            box_pos=(bx, by),
            box_pushed=True,
        )

    return SimState(agent=(nx, ny), box_pos=sim.box_pos, box_pushed=sim.box_pushed)


def at_goal(cfg: GridConfig, sim: SimState) -> bool:
    return sim.agent == cfg.goal


def render(cfg: GridConfig, sim: SimState) -> str:
    rows = []
    for y in range(cfg.height):
        row = []
        for x in range(cfg.width):
            if (x, y) in cfg.walls:
                row.append("#")
            elif (x, y) == sim.agent:
                row.append("A")
            elif (x, y) == sim.box_pos:
                row.append("B")
            elif (x, y) == cfg.goal:
                row.append("G")
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows)
