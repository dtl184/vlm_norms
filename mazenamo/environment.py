"""
MazeNamoEnvironment — EnvironmentInterface adapter.

State encoding
--------------
State = y * width + x  (agent cell index, integer)

The planner is NORM-UNAWARE: it plans on agent position only, treating walls
as hard obstacles and everything else (forbidden cells, boxes) as transparent.
This means it finds the direct shortest path to any goal — including paths
through spills or restricted zones — which is exactly the "shortcut" the norm
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
    GREET,
    NORTH,
    PUSH,
    SOUTH,
    WEST,
    GridConfig,
)


class MazeNamoEnvironment(EnvironmentInterface):
    """
    MazeNamo environment wrapper.

    The planner is norm-unaware (avoids walls only).  Norm-aware trajectory
    generators in trajectory_gen.py handle scenario-specific constraints.

    Parameters
    ----------
    cfg : GridConfig
        Grid layout (walls, goal, box start, forbidden cells, guard, etc.).
    """

    def __init__(self, cfg: GridConfig) -> None:
        self.cfg = cfg
        self._wall_set = cfg.walls

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
        """Norm-unaware: walls are blocked, everything else is free."""
        return self.in_bounds(x, y) and (x, y) not in self._wall_set

    def is_passable_norm_aware(self, x: int, y: int) -> bool:
        """Norm-aware: also blocks forbidden cells (spills, restricted zones)."""
        return self.is_passable(x, y) and (x, y) not in self.cfg.forbidden_cells

    # ------------------------------------------------------------------
    # Planner (norm-unaware A*)
    # ------------------------------------------------------------------

    def _heuristic(self, s: State, g: State) -> int:
        sx, sy = self.state_to_xy(s)
        gx, gy = self.state_to_xy(g)
        return abs(sx - gx) + abs(sy - gy)

    def _neighbors(self, state: State, passable_fn=None) -> list[tuple[State, Action]]:
        if passable_fn is None:
            passable_fn = self.is_passable
        x, y = self.state_to_xy(state)
        out = []
        for action, (dx, dy) in DELTAS.items():
            nx, ny = x + dx, y + dy
            if passable_fn(nx, ny):
                out.append((self.xy_to_state(nx, ny), action))
        return out

    def _astar(
        self,
        start: State,
        goal: State,
        passable_fn=None,
    ) -> tuple[Trajectory, float]:
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
            for nxt, action in self._neighbors(cur, passable_fn):
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
        """Norm-unaware plan (avoids walls only)."""
        path, cost = self._astar(start, goal)
        return [] if cost == float("inf") else [path]

    def plan_norm_aware(self, start: State, goal: State) -> list[Trajectory]:
        """Norm-aware plan (also avoids forbidden cells)."""
        path, cost = self._astar(start, goal, self.is_passable_norm_aware)
        return [] if cost == float("inf") else [path]

    def shortest_path_cost(self, start: State, goal: State) -> float:
        _, cost = self._astar(start, goal)
        return cost

    def trajectory_cost(self, trajectory: Trajectory) -> float:
        return float(len(trajectory))

    def successor(self, state: State, action: Action) -> State | None:
        if action == GREET:
            return state  # GREET: agent stays in place
        x, y = self.state_to_xy(state)
        # PUSH acts as NORTH (agent moves into box, box displaced one cell north)
        move = NORTH if action == PUSH else action
        if move not in DELTAS:
            return None
        dx, dy = DELTAS[move]
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
        d_goal = abs(x - gx) + abs(y - gy)
        parts = [f"cell ({x},{y})", f"zone={self._zone(x, y)}", f"d_goal={d_goal}"]

        if self.cfg.required_box_start is not None:
            bx, by = self.cfg.required_box_start
            d_box = abs(x - bx) + abs(y - by)
            parts.append(f"d_box={d_box}")

        label = self.cfg.cell_labels.get((x, y))
        if label:
            parts.append(label)

        if self.cfg.guard_pos is not None:
            gux, guy = self.cfg.guard_pos
            d_guard = abs(x - gux) + abs(y - guy)
            parts.append(f"d_guard={d_guard}")

        return " ".join(f"[{p}]" if i == 0 else p for i, p in enumerate(parts))

    def describe_action(self, action: Action) -> str:
        return ACTION_NAMES.get(action, f"action_{action}")

    def extract_semantic_features(self, state: State) -> WorldState:
        x, y = self.state_to_xy(state)
        gx, gy = self.cfg.goal
        d_goal = abs(x - gx) + abs(y - gy)
        feats: WorldState = {
            "x": x,
            "y": y,
            "state": state,
            "zone": self._zone(x, y),
            "dist_to_goal": d_goal,
            "near_goal": d_goal <= 1,
        }

        if self.cfg.required_box_start is not None:
            bx, by = self.cfg.required_box_start
            d_box = abs(x - bx) + abs(y - by)
            feats["dist_to_box"] = d_box
            feats["near_box"] = d_box <= 1

        label = self.cfg.cell_labels.get((x, y))
        if label:
            feats["cell_type"] = label

        near_forbidden = any(
            abs(x - fx) + abs(y - fy) <= 1
            for (fx, fy) in self.cfg.forbidden_cells
        )
        if near_forbidden:
            feats["near_forbidden"] = True

        if self.cfg.guard_pos is not None:
            gux, guy = self.cfg.guard_pos
            d_guard = abs(x - gux) + abs(y - guy)
            feats["dist_to_guard"] = d_guard
            feats["near_guard"] = d_guard <= 1

        return feats

    def get_environment_description(self) -> str:
        gx, gy = self.cfg.goal
        parts = [
            f"A {self.cfg.width}×{self.cfg.height} grid maze (MazeNamo). "
            f"The agent must reach the goal at ({gx},{gy}). "
            f"Actions: NORTH (y−1), SOUTH (y+1), EAST (x+1), WEST (x−1). "
            f"Walls block movement."
        ]

        if self.cfg.required_box_start is not None:
            bx, by = self.cfg.required_box_start
            parts.append(
                f"There is a REQUIRED BOX at ({bx},{by}) that the agent must push "
                f"(by moving into it) before reaching the goal. "
                f"Pushing a box moves it one cell in the direction of travel."
            )

        spills = [(x, y) for (x, y), lbl in self.cfg.cell_labels.items() if lbl == "spill"]
        if spills:
            coords = ", ".join(f"({x},{y})" for x, y in spills)
            parts.append(f"There are SPILL cells at {coords}. Agents must not step on spills.")

        restricted = [(x, y) for (x, y), lbl in self.cfg.cell_labels.items() if lbl == "restricted"]
        if restricted:
            coords = ", ".join(f"({x},{y})" for x, y in restricted)
            parts.append(f"There is a RESTRICTED ZONE at {coords}. Agents must not enter these cells.")

        if self.cfg.guard_pos is not None:
            gux, guy = self.cfg.guard_pos
            parts.append(
                f"There is a GUARD at ({gux},{guy}). "
                f"Agents must GREET the guard before passing through the checkpoint."
            )

        return "  ".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zone(self, x: int, y: int) -> str:
        gx, gy = self.cfg.goal
        if (x, y) == self.cfg.goal:
            return "goal"
        if abs(x - gx) + abs(y - gy) <= 2:
            return "near_goal"
        if self.cfg.required_box_start is not None:
            bx, by = self.cfg.required_box_start
            if abs(x - bx) + abs(y - by) <= 1:
                return "near_box"
        if self.cfg.guard_pos is not None:
            gux, guy = self.cfg.guard_pos
            if abs(x - gux) + abs(y - guy) <= 1:
                return "near_guard"
        label = self.cfg.cell_labels.get((x, y))
        if label:
            return label
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
        assert self.cfg.required_box_start is not None
        bx, by = self.cfg.required_box_start
        return self.xy_to_state(bx, by + 1)
