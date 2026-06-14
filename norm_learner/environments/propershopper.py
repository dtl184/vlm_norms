"""
ProperShopperEnvironment — EnvironmentInterface adapter for the ProperShopper
supermarket simulation.

State representation
--------------------
State = (coarse_gx, coarse_gy, has_basket: bool, has_items: bool)

Position is quantised to a coarse grid (CELL_SIZE = 0.3 sim-units = 2 raw movement
steps). This is coarse enough that both trajectories' checkout INTERACTs map to the
same abstract state — which lets hyp_obligations correctly intersect them and surface
the "pay before leaving" obligation.

Planning
--------
plan() returns [] (no prohibition hypothesis trees are needed for this environment).
The payment norm is an obligation detected via hyp_obligations, not a prohibition.
shortest_path_cost() returns coarse-Manhattan × 2 (converting to raw-step units) and
float('inf') for impossible semantic transitions (e.g. norm-naive planner cannot pay).

Loading trajectories
--------------------
    from norm_learner.environments.propershopper import ProperShopperEnvironment
    import json

    env = ProperShopperEnvironment()
    with open("propershopper/new_trajectories.json") as f:
        raw = json.load(f)
    trajectories = env.load_trajectories(raw)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..environment import EnvironmentInterface
from ..types import Action, State, Trajectory, WorldState


# ---------------------------------------------------------------------------
# Zone classification (for human-readable descriptions only)
# ---------------------------------------------------------------------------

# Ordered list of (zone_name, x_min, x_max, y_min, y_max)
# y increases southward in ProperShopper: exit/checkout at low y, shelves at high y.
_ZONES = [
    ("exit",     0.0,  2.0,  0.0,  5.5),
    ("checkout", 2.0,  5.5,  4.0,  8.5),
    ("basket",   0.0,  4.5, 16.0, 21.0),
    ("shelf",    3.0, 21.0,  8.0, 21.0),
]


def _zone_for(x: float, y: float) -> str:
    for name, x0, x1, y0, y1 in _ZONES:
        if x0 <= x < x1 and y0 <= y < y1:
            return name
    return "floor"


# ---------------------------------------------------------------------------
# ProperShopperEnvironment
# ---------------------------------------------------------------------------

class ProperShopperEnvironment(EnvironmentInterface):
    """
    Norm-learner adapter for the ProperShopper supermarket simulation.

    Parameters
    ----------
    cell_size : float
        Coarse grid cell size in simulation units (default 0.3, which equals
        two raw movement steps of 0.15 each).
    """

    STEP_SIZE: float = 0.15   # raw movement step in simulation units
    STEPS_PER_CELL: int = 2   # raw steps per coarse cell (CELL_SIZE / STEP_SIZE)

    def __init__(self, cell_size: float = 0.3) -> None:
        self.cell_size = cell_size

    # ------------------------------------------------------------------
    # Abstraction helpers
    # ------------------------------------------------------------------

    def _coarse(self, x: float, y: float) -> tuple[int, int]:
        return round(x / self.cell_size), round(y / self.cell_size)

    def abstract_state(self, raw_state: dict) -> tuple:
        """
        Convert a raw ProperShopper state dict to a hashable abstract state.

        Abstract state = (coarse_gx, coarse_gy, has_basket, has_items)
        where has_items = len(items_in_basket) > 0.
        """
        gx, gy = self._coarse(raw_state["x"], raw_state["y"])
        has_basket = bool(raw_state["has_basket"])
        has_items = len(raw_state["items_in_basket"]) > 0
        return (gx, gy, has_basket, has_items)

    def load_trajectories(
        self, raw: list[list[dict]] | str | Path
    ) -> list[Trajectory]:
        """
        Convert raw ProperShopper trajectory data to abstract Trajectory lists.

        Parameters
        ----------
        raw : list[list[dict]] | str | Path
            Either the already-parsed JSON structure (list of trajectories, each
            a list of {"state": ..., "action": ...} dicts), or a path to the
            JSON file.

        Returns
        -------
        list[Trajectory]
            Each Trajectory is a list of (abstract_State, action_str) pairs,
            one per raw step — ready to pass to NormLearner.process_trajectory().
        """
        if isinstance(raw, (str, Path)):
            with open(raw) as f:
                raw = json.load(f)

        trajectories: list[Trajectory] = []

        for raw_traj in raw:
            traj: Trajectory = []

            for i, step in enumerate(raw_traj):
                action = step["action"]

                if i == 0:
                    state_for_action = step["state"]
                else:
                    state_for_action = raw_traj[i - 1]["state"]

                traj.append((self.abstract_state(state_for_action), action))

            trajectories.append(traj)

        return trajectories

    # ------------------------------------------------------------------
    # EnvironmentInterface — planning
    # ------------------------------------------------------------------

    def plan(self, start: State, goal: State) -> list[Trajectory]:
        """
        Return [] — no prohibition hypothesis trees are generated.

        The "must pay" norm is an obligation (detected via hyp_obligations as the
        intersection of INTERACT-at-checkout across all trajectories), not a
        prohibition.  Generating prohibition trees for navigation paths would only
        produce noise.
        """
        return []

    def shortest_path_cost(self, start: State, goal: State) -> float:
        """
        Lower-bound on raw steps needed to reach *goal* from *start*.

        Norm-naive: the planner does not model the payment transition, so any
        goal that requires items to disappear from the basket (has_items True→False)
        is unreachable (returns inf).

        Cost units match trajectory_cost (raw steps):
            coarse cells × STEPS_PER_CELL + required INTERACT steps.
        """
        gx1, gy1, hb1, hi1 = start
        gx2, gy2, hb2, hi2 = goal

        # Irreversible semantic transitions the norm-naive planner cannot perform
        if hb1 and not hb2:
            return float("inf")  # basket cannot be un-obtained
        if hi1 and not hi2:
            return float("inf")  # planner does not model payment

        extra = 0
        if not hb1 and hb2:
            extra += 1  # one INTERACT to pick up basket
        if not hi1 and hi2:
            extra += 1  # one INTERACT to pick up item

        manhattan = abs(gx1 - gx2) + abs(gy1 - gy2)
        return manhattan * self.STEPS_PER_CELL + extra

    def trajectory_cost(self, trajectory: Trajectory) -> float:
        """Each raw step in the trajectory costs 1."""
        return float(len(trajectory))

    def successor(self, state: State, action: Action) -> State | None:
        """Model state transitions for the five ProperShopper actions."""
        gx, gy, hb, hi = state
        if action == "NORTH":
            return (gx, gy - 1, hb, hi)
        if action == "SOUTH":
            return (gx, gy + 1, hb, hi)
        if action == "EAST":
            return (gx + 1, gy, hb, hi)
        if action == "WEST":
            return (gx - 1, gy, hb, hi)
        if action == "INTERACT":
            x = gx * self.cell_size
            y = gy * self.cell_size
            zone = _zone_for(x, y)
            if zone == "basket" and not hb:
                return (gx, gy, True, hi)
            if zone == "shelf" and hb and not hi:
                return (gx, gy, hb, True)
            if zone == "checkout" and hi:
                return (gx, gy, hb, False)   # payment: items cleared
            return state  # INTERACT with no effect
        return None

    # ------------------------------------------------------------------
    # EnvironmentInterface — semantic descriptions
    # ------------------------------------------------------------------

    def describe_state(self, state: State) -> str:
        gx, gy, hb, hi = state
        x = round(gx * self.cell_size, 2)
        y = round(gy * self.cell_size, 2)
        zone = _zone_for(x, y)
        basket_s = "basket" if hb else "no-basket"
        items_s = "has-items" if hi else "no-items"
        return f"({x:.1f},{y:.1f}) [{zone}, {basket_s}, {items_s}]"

    def describe_action(self, action: Action) -> str:
        return str(action)

    def extract_semantic_features(self, state: State) -> WorldState:
        gx, gy, hb, hi = state
        x = round(gx * self.cell_size, 2)
        y = round(gy * self.cell_size, 2)
        return {
            "x": x,
            "y": y,
            "zone": _zone_for(x, y),
            "has_basket": hb,
            "has_items": hi,
        }

    def get_environment_description(self) -> str:
        return (
            "ProperShopper is a supermarket simulation where an agent completes a "
            "shopping trip. The store layout (y increases southward): exit area "
            "(y<5.5), checkout counters (y≈5–8), shopping aisles and shelves (y>8), "
            "basket pickup area (x<4.5, y>16). "
            "Actions: NORTH/SOUTH/EAST/WEST for navigation; INTERACT to pick up basket "
            "(at basket area), pick up item (at shelf), or pay (at checkout). "
            "A norm-compliant trip: get basket → pick item(s) → pay at checkout → exit."
        )
