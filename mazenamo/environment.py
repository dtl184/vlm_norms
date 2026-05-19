"""
MazeNamoEnvironment — EnvironmentInterface adapter.

State encoding
--------------
State = y * width + x  (agent cell index, integer)

The planner is NORM-UNAWARE: it plans on agent position only, treating walls
as hard obstacles and the box as transparent (ignores it).  This means it
will find the direct shortest path to any goal cell — including paths that
skip the required-box detour — which is exactly the "shortcut" the norm
learner needs to detect.
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from norm_learner.environment import EnvironmentInterface
from norm_learner.types import Action, State, Trajectory, WorldState

from .env import (
    ACTION_NAMES,
    DELTAS,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    GridConfig,
)


class MazeNamoEnvironment(EnvironmentInterface):
    """
    MazeNamo environment wrapper.

    The norm baked into the training trajectories is:
        *The agent must push the required box before reaching the goal.*

    Parameters
    ----------
    cfg : GridConfig
        Grid layout (walls, goal, box start).
    """

    def __init__(self, cfg: GridConfig) -> None:
        self.cfg = cfg
        self._wall_set = cfg.walls  # cached for fast membership test

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def xy_to_state(self, x: int, y: int) -> State:
        return y * self.cfg.width + x

    def state_to_xy(self, state: State) -> tuple[int, int]:
        return state % self.cfg.width, state // self.cfg.width

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.cfg.width and 0 <= y < self.cfg.height

    def is_passable(self, x: int, y: int) -> bool:
        """For the norm-unaware planner: walls are blocked, everything else free."""
        return self.in_bounds(x, y) and (x, y) not in self._wall_set

    # ------------------------------------------------------------------
    # Planner (norm-unaware A*)
    # ------------------------------------------------------------------

    def _heuristic(self, s: State, g: State) -> int:
        sx, sy = self.state_to_xy(s)
        gx, gy = self.state_to_xy(g)
        return abs(sx - gx) + abs(sy - gy)

    def _neighbors(self, state: State) -> list[tuple[State, Action]]:
        x, y = self.state_to_xy(state)
        out = []
        for action, (dx, dy) in DELTAS.items():
            nx, ny = x + dx, y + dy
            if self.is_passable(nx, ny):
                out.append((self.xy_to_state(nx, ny), action))
        return out

    def _astar(self, start: State, goal: State) -> tuple[Trajectory, float]:
        if start == goal:
            return [], 0.0
        queue: list[tuple[float, float, State, Trajectory]] = []
        heapq.heappush(queue, (self._heuristic(start, goal), 0.0, start, []))
        best_g: dict[State, float] = {start: 0.0}

        while queue:
            _, g, cur, path = heapq.heappop(queue)
            if cur == goal:
                return path, g
            if g > best_g.get(cur, float("inf")):
                continue
            for nxt, action in self._neighbors(cur):
                ng = g + 1.0
                if ng < best_g.get(nxt, float("inf")):
                    best_g[nxt] = ng
                    f = ng + self._heuristic(nxt, goal)
                    heapq.heappush(queue, (f, ng, nxt, path + [(cur, action)]))
        return [], float("inf")

    # ------------------------------------------------------------------
    # EnvironmentInterface — planning
    # ------------------------------------------------------------------

    def plan(self, start: State, goal: State) -> list[Trajectory]:
        path, cost = self._astar(start, goal)
        return [] if cost == float("inf") else [path]

    def shortest_path_cost(self, start: State, goal: State) -> float:
        _, cost = self._astar(start, goal)
        return cost

    def trajectory_cost(self, trajectory: Trajectory) -> float:
        return float(len(trajectory))

    def successor(self, state: State, action: Action) -> State | None:
        x, y = self.state_to_xy(state)
        dx, dy = DELTAS[action]
        nx, ny = x + dx, y + dy
        if not self.is_passable(nx, ny):
            return None
        return self.xy_to_state(nx, ny)

    # ------------------------------------------------------------------
    # EnvironmentInterface — semantic descriptions
    # ------------------------------------------------------------------

    def describe_state(self, state: State) -> str:
        x, y = self.state_to_xy(state)
        gx, gy = self.cfg.goal
        bx, by = self.cfg.required_box_start
        d_goal = abs(x - gx) + abs(y - gy)
        d_box = abs(x - bx) + abs(y - by)
        zone = self._zone(x, y)
        return f"cell ({x},{y}) [{zone}, d_goal={d_goal}, d_box={d_box}]"

    def describe_action(self, action: Action) -> str:
        return ACTION_NAMES.get(action, f"action_{action}")

    def extract_semantic_features(self, state: State) -> WorldState:
        x, y = self.state_to_xy(state)
        gx, gy = self.cfg.goal
        bx, by = self.cfg.required_box_start
        d_goal = abs(x - gx) + abs(y - gy)
        d_box = abs(x - bx) + abs(y - by)
        return {
            "x": x,
            "y": y,
            "state": state,
            "zone": self._zone(x, y),
            "dist_to_goal": d_goal,
            "dist_to_req_box": d_box,
            "near_goal": d_goal <= 1,
            "near_req_box": d_box <= 1,
            "at_push_position": (x == bx and y == by + 1),  # one south of box
        }

    def get_environment_description(self) -> str:
        gx, gy = self.cfg.goal
        bx, by = self.cfg.required_box_start
        return (
            f"A {self.cfg.width}×{self.cfg.height} grid maze with box-pushing "
            f"mechanics (MazeNamo).  "
            f"The agent must reach the goal at ({gx},{gy}).  "
            f"There is a REQUIRED BOX at ({bx},{by}) that the agent must push "
            f"(by moving into it) before reaching the goal.  "
            f"Actions: NORTH (y−1), SOUTH (y+1), EAST (x+1), WEST (x−1).  "
            f"Walls block movement; pushing a box moves it one cell in the "
            f"direction of travel."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zone(self, x: int, y: int) -> str:
        gx, gy = self.cfg.goal
        bx, by = self.cfg.required_box_start
        if (x, y) == self.cfg.goal:
            return "goal"
        if abs(x - gx) + abs(y - gy) <= 2:
            return "near_goal"
        if abs(x - bx) + abs(y - by) <= 1:
            return "near_req_box"
        if y >= self.cfg.height - 3:
            return "start_area"
        return "open"

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def goal_state(self) -> State:
        return self.xy_to_state(*self.cfg.goal)

    def box_push_state(self) -> State:
        """The state SOUTH of the required box — the push position."""
        bx, by = self.cfg.required_box_start
        return self.xy_to_state(bx, by + 1)
