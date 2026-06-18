"""
ProperShopperEnvironment — EnvironmentInterface adapter for the ProperShopper
supermarket simulation.

State representation
--------------------
State = (has_basket: bool, has_items: bool)

Position is intentionally omitted. Checkouts appear at different y-coordinates
across trajectories, so grid-position would prevent obligation candidates from
intersecting. The two boolean flags are sufficient to distinguish every
norm-relevant transition:
  (False, False) → INTERACT  — pick up basket
  (True,  False) → INTERACT  — pick up item
  (True,  True)  → INTERACT  — pay at checkout  ← the obligatory action

Planning
--------
plan() returns [] (no prohibition hypothesis trees are needed for this environment).
The payment norm is an obligation detected via hyp_obligations, not a prohibition.
shortest_path_cost() always returns inf so shortcut detection is skipped entirely
(consistent with plan() == []).

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
# ProperShopperEnvironment
# ---------------------------------------------------------------------------

class ProperShopperEnvironment(EnvironmentInterface):
    """Norm-learner adapter for the ProperShopper supermarket simulation."""

    def abstract_state(self, raw_state: dict) -> tuple:
        """
        Convert a raw ProperShopper state dict to a hashable abstract state.

        Abstract state = (has_basket, has_items)
        where has_items = len(items_in_basket) > 0.
        """
        has_basket = bool(raw_state["has_basket"])
        has_items = len(raw_state["items_in_basket"]) > 0
        return (has_basket, has_items)

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
        """Return [] — payment norm is an obligation, not a prohibition."""
        return []

    def shortest_path_cost(self, start: State, goal: State) -> float:
        """Always inf — shortcut detection is disabled (plan() == [])."""
        return float("inf")

    def trajectory_cost(self, trajectory: Trajectory) -> float:
        """Each raw step in the trajectory costs 1."""
        return float(len(trajectory))

    def successor(self, state: State, action: Action) -> State | None:
        """Model semantic state transitions for the five ProperShopper actions."""
        hb, hi = state
        if action in ("NORTH", "SOUTH", "EAST", "WEST"):
            return state  # movement doesn't change semantic state
        if action == "INTERACT":
            if not hb:
                return (True, hi)   # pick up basket
            if hb and not hi:
                return (hb, True)   # pick up item
            if hb and hi:
                return (hb, False)  # pay: items cleared
        return None

    # ------------------------------------------------------------------
    # EnvironmentInterface — semantic descriptions
    # ------------------------------------------------------------------

    def describe_state(self, state: State) -> str:
        hb, hi = state
        basket_s = "basket" if hb else "no-basket"
        items_s = "has-items" if hi else "no-items"
        return f"[{basket_s}, {items_s}]"

    def describe_action(self, action: Action) -> str:
        return str(action)

    def extract_semantic_features(self, state: State) -> WorldState:
        hb, hi = state
        return {
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
