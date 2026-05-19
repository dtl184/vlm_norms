"""
SupermarketEnvironment — EnvironmentInterface adapter for the 19×23 grid
supermarket used in the original norm-discovery paper.

Shelf layout is taken from ``socket_env.obj_state_index_dict["shelves"]``
when available, or can be supplied explicitly via the ``shelf_states``
constructor argument (useful when ``socket_env`` is not installed).

Usage
-----
    from norm_learner.environments.supermarket import SupermarketEnvironment

    # with socket_env on the path:
    env = SupermarketEnvironment()

    # without socket_env — pass shelf state indices manually:
    env = SupermarketEnvironment(shelf_states={42, 61, 80, ...})

Extending for evaluation
------------------------
The ground-truth prohibited states (TRUE_PROHIBITED_STATES) are kept in this
file for evaluation only.  They are NOT used by the symbolic algorithm —
use ``eval_sanitize`` to filter learned norms against ground truth.
"""

from __future__ import annotations

import heapq

from ..environment import EnvironmentInterface
from ..types import Action, State, Trajectory, WorldState

# ---------------------------------------------------------------------------
# Ground-truth prohibited states (evaluation / oracle use only)
# ---------------------------------------------------------------------------

TRUE_PROHIBITED_STATES: frozenset[int] = frozenset({
    319, 320, 321, 340, 359, 378, 377, 347, 346, 345, 344, 363,
    382, 192, 211, 212, 230, 231, 232, 100, 101, 99, 98, 79,
    148, 129, 130, 131,
})

# Rough zone boundaries for the 19×23 supermarket layout
_ZONES = [
    ("entrance",   0,  2, 20, 23),
    ("checkout",  16, 20,  0,  6),
    ("aisle_1",    3,  7,  0, 23),
    ("aisle_2",    7, 11,  0, 23),
    ("aisle_3",   11, 15,  0, 23),
    ("aisle_4",   15, 19,  0, 23),
]


def _zone_for(x: int, y: int) -> str:
    for name, x0, x1, y0, y1 in _ZONES:
        if x0 <= x < x1 and y0 <= y < y1:
            return name
    return "floor"


# ---------------------------------------------------------------------------
# SupermarketEnvironment
# ---------------------------------------------------------------------------

class SupermarketEnvironment(EnvironmentInterface):
    """
    A* planner + semantic interface for the 19×23 supermarket grid.

    Parameters
    ----------
    grid_width, grid_height : int
        Dimensions of the grid.
    shelf_states : set[int] | None
        Set of grid indices that are shelves (obstacles).  When *None* the
        class tries to import ``socket_env``; if that fails it defaults to an
        empty set (no obstacles — planning will still work, just unconstrained).
    """

    ACTION_NAMES: dict[int, str] = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST"}

    def __init__(
        self,
        grid_width: int = 19,
        grid_height: int = 23,
        shelf_states: set[int] | None = None,
    ) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height

        if shelf_states is not None:
            self.invalid_states: set[int] = set(shelf_states)
        else:
            try:
                from socket_env import obj_state_index_dict  # type: ignore[import]
                self.invalid_states = set(obj_state_index_dict["shelves"])
            except ImportError:
                self.invalid_states = set()
                import warnings
                warnings.warn(
                    "socket_env not found; SupermarketEnvironment has no "
                    "shelf obstacles.  Pass shelf_states= explicitly to "
                    "suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )

        self._action_map: dict[int, tuple[int, int]] = {
            0: (0, -1),   # NORTH
            1: (0, 1),    # SOUTH
            2: (1, 0),    # EAST
            3: (-1, 0),   # WEST
        }

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def idx_to_xy(self, idx: int) -> tuple[int, int]:
        return idx % self.grid_width, idx // self.grid_width

    def xy_to_idx(self, x: int, y: int) -> int:
        return y * self.grid_width + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def successor(self, state: State, action: Action) -> State | None:
        x, y = self.idx_to_xy(state)
        dx, dy = self._action_map[action]
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny):
            return None
        nidx = self.xy_to_idx(nx, ny)
        return None if nidx in self.invalid_states else nidx

    def _neighbors(self, state: int) -> list[tuple[int, int]]:
        x, y = self.idx_to_xy(state)
        out: list[tuple[int, int]] = []
        for action, (dx, dy) in self._action_map.items():
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                nidx = self.xy_to_idx(nx, ny)
                if nidx not in self.invalid_states:
                    out.append((nidx, action))
        return out

    def _heuristic(self, state: int, goal: int) -> int:
        x1, y1 = self.idx_to_xy(state)
        x2, y2 = self.idx_to_xy(goal)
        return abs(x1 - x2) + abs(y1 - y2)

    # ------------------------------------------------------------------
    # EnvironmentInterface — planning
    # ------------------------------------------------------------------

    def _astar(self, start: int, goal: int) -> tuple[Trajectory, float]:
        if start == goal:
            return [], 0.0
        queue: list[tuple[float, float, int, Trajectory]] = []
        heapq.heappush(queue, (self._heuristic(start, goal), 0.0, start, []))
        best_g: dict[int, float] = {start: 0.0}

        while queue:
            _f, g, current, path = heapq.heappop(queue)
            if current == goal:
                return path, g
            if g > best_g.get(current, float("inf")):
                continue
            for nxt, action in self._neighbors(current):
                new_g = g + 1.0
                if new_g < best_g.get(nxt, float("inf")):
                    best_g[nxt] = new_g
                    heapq.heappush(
                        queue,
                        (new_g + self._heuristic(nxt, goal), new_g, nxt,
                         path + [(current, action)]),
                    )
        return [], float("inf")

    def plan(self, start: State, goal: State) -> list[Trajectory]:
        path, cost = self._astar(start, goal)
        return [] if cost == float("inf") else [path]

    def shortest_path_cost(self, start: State, goal: State) -> float:
        _, cost = self._astar(start, goal)
        return cost

    def trajectory_cost(self, trajectory: Trajectory) -> float:
        return float(len(trajectory))

    # ------------------------------------------------------------------
    # EnvironmentInterface — semantic descriptions
    # ------------------------------------------------------------------

    def describe_state(self, state: State) -> str:
        x, y = self.idx_to_xy(int(state))
        zone = _zone_for(x, y)
        return f"state {state} @ ({x},{y}) [{zone}]"

    def describe_action(self, action: Action) -> str:
        return self.ACTION_NAMES.get(int(action), f"action_{action}")

    def extract_semantic_features(self, state: State) -> WorldState:
        x, y = self.idx_to_xy(int(state))
        return {
            "grid_index": int(state),
            "x": x,
            "y": y,
            "zone": _zone_for(x, y),
            "near_shelf": any(
                self.xy_to_idx(x + dx, y + dy) in self.invalid_states
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if self.in_bounds(x + dx, y + dy)
            ),
        }

    def get_environment_description(self) -> str:
        return (
            f"A {self.grid_width}×{self.grid_height} grid-world supermarket "
            f"with {len(self.invalid_states)} shelf obstacles.  "
            "An agent navigates aisles between shelves to reach shopping goals.  "
            "Actions: NORTH (y-1), SOUTH (y+1), EAST (x+1), WEST (x-1).  "
            "Some areas may be normatively restricted (e.g. staff-only zones, "
            "wet-floor areas)."
        )

    # ------------------------------------------------------------------
    # Evaluation helpers (not part of EnvironmentInterface)
    # ------------------------------------------------------------------

    def eval_sanitize_prohibitions(
        self, prohibitions: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """
        Keep only state-action pairs whose successor is a TRUE_PROHIBITED_STATE.
        Use this for evaluation against the oracle — never inside the learner.
        """
        return {
            sa for sa in prohibitions
            if self.successor(sa[0], sa[1]) in TRUE_PROHIBITED_STATES
        }

    def print_eval_summary(self, prohibitions: set[tuple[int, int]]) -> None:
        """Print a quick oracle-vs-learned comparison."""
        filtered = self.eval_sanitize_prohibitions(prohibitions)
        total_gt = sum(
            1 for s in range(self.grid_width * self.grid_height)
            if s not in self.invalid_states
            for a in range(4)
            if self.successor(s, a) in TRUE_PROHIBITED_STATES
        )
        print(f"Learned prohibitions (oracle-filtered): {len(filtered)}")
        print(f"Ground-truth prohibited pairs          : {total_gt}")
        for sa in sorted(filtered):
            ns = self.successor(sa[0], sa[1])
            print(f"  {sa} -> {ns}")
