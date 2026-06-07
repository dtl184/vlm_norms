"""
MazeNamo norm scenarios.

Each scenario packages a grid layout, ground-truth norm description, a
trajectory generator, and a realistic MockLLM response so any scenario can
be run without a real model.

Usage
-----
    from mazenamo.scenarios import SCENARIOS
    scenario = SCENARIOS["avoid_spills"]
    cfg = GridConfig.from_layout(scenario.layout)
    env = MazeNamoEnvironment(cfg)
    trajectories = scenario.generate(env)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from norm_learner.types import Trajectory

from .env import GridConfig
from .environment import MazeNamoEnvironment
from .trajectory_gen import (
    avoid_forbidden_trajectory,
    generate_from_all_starts,
    generate_from_all_starts_fn,
    greet_before_door_trajectory,
    norm_following_trajectory,
)


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    norm_description: str    # ground-truth norm shown in output header
    layout: list[str]
    mock_llm_response: str   # JSON string used by MockLLM

    # Callable: (env: MazeNamoEnvironment) -> list[Trajectory]
    _gen: Callable

    def generate(self, env: MazeNamoEnvironment) -> list[Trajectory]:
        return self._gen(env)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

_PUSH_BEFORE_GOAL_LAYOUT = [
    "########",
    "#......#",
    "#....G.#",   # goal at (5, 2)
    "#......#",
    "#.B....#",   # required box at (2, 4)
    "#......#",
    "#....A.#",   # agent default at (5, 6)
    "########",
]

# Spill cells at x=2,3,4,5 in row y=3 form a barrier.  The only non-wall
# crossing points are (1,3) and (6,3), forcing a wide detour from any start
# in columns 2–5.  Goal is at (3,2); default agent start at (3,6).
_AVOID_SPILLS_LAYOUT = [
    "########",
    "#......#",
    "#..G...#",   # goal at (3, 2)
    "#.ssss.#",   # spills at (2,3)(3,3)(4,3)(5,3); open at (1,3) and (6,3)
    "#......#",
    "#......#",
    "#..A...#",   # agent default at (3, 6)
    "########",
]

# Wall at y=3 with a single door at x=4 separates the start area from the
# goal area.  A guard at (6, 5) must be greeted before the agent crosses.
_GREET_BEFORE_DOOR_LAYOUT = [
    "########",
    "#..G...#",   # goal at (3, 1)
    "#......#",
    "####.###",   # wall with door at (4, 3)
    "#......#",
    "#.....g#",   # guard at (6, 5)
    "#....A.#",   # agent default at (5, 6)
    "########",
]

# Restricted-zone cells at x=2,3,4,5 in row y=3 form a barrier; only (1,3)
# and (6,3) are non-wall crossing points.  Same structure as avoid_spills but
# labeled as an access-control zone (safety norm rather than cleanliness).
# Goal at (3,2), agent at (3,6) — same as avoid_spills; the semantic
# difference is what the LLM must distinguish.
_NO_RESTRICTED_ZONE_LAYOUT = [
    "########",
    "#......#",
    "#..G...#",   # goal at (3, 2)
    "#.RRRR.#",   # restricted zone at (2,3)(3,3)(4,3)(5,3)
    "#......#",
    "#......#",
    "#..A...#",   # agent default at (3, 6)
    "########",
]


# ---------------------------------------------------------------------------
# Mock LLM responses (realistic stubs for each scenario)
# ---------------------------------------------------------------------------

_PUSH_MOCK = json.dumps([{
    "context": ["setting(grid_maze)", "task(reach_goal)", "box_present(B)"],
    "modality": "obligatory",
    "action": "push(agent, box_B)",
    "norm_type": "politeness",
    "description": (
        "The agent must push the required box (B) before reaching the goal (G). "
        "All observed agents detour to interact with the box prior to moving toward "
        "the goal, even though a shorter direct route exists."
    ),
    "reasoning": (
        "The PUSH action appears in every observed trajectory. Agents consistently "
        "take longer paths that pass through the box location before heading to the "
        "goal — a classic prerequisite-action obligation norm."
    ),
}])

_SPILLS_MOCK = json.dumps([{
    "context": ["setting(grid_maze)", "task(reach_goal)", "spill_present(s)"],
    "modality": "forbidden",
    "action": "step_on(agent, spill_cell)",
    "norm_type": "cleanliness",
    "description": (
        "Agents are forbidden from stepping on spill cells. All observed agents "
        "route around the spill at (3,3) even though passing through it would be "
        "the shortest path to the goal."
    ),
    "reasoning": (
        "No trajectory ever includes a move into the spill cell. Agents consistently "
        "take a longer detour around it — strong evidence for a cleanliness-based "
        "prohibition on entering contaminated cells."
    ),
}])

_GREET_MOCK = json.dumps([{
    "context": ["setting(grid_maze)", "task(reach_goal)", "guard_present(g)", "door_ahead"],
    "modality": "obligatory",
    "action": "greet(agent, guard)",
    "norm_type": "politeness",
    "description": (
        "The agent must greet the guard before passing through the checkpoint door. "
        "All observed agents detour to the guard position and perform GREET prior to "
        "crossing the wall door, even when a shorter direct route through the door exists."
    ),
    "reasoning": (
        "GREET is the only non-movement action and appears in every trajectory. It "
        "consistently occurs early — in the prerequisite phase — before the agent "
        "crosses the door at (4,3). This temporal pattern strongly indicates a "
        "greeting obligation that must be satisfied before passing the checkpoint."
    ),
}])

_RESTRICTED_MOCK = json.dumps([{
    "context": ["setting(grid_maze)", "task(reach_goal)", "restricted_zone_present"],
    "modality": "forbidden",
    "action": "enter(agent, restricted_cell)",
    "norm_type": "safety",
    "description": (
        "Agents are forbidden from entering the restricted zone cells at rows y=3 "
        "(columns 2–5). All observed agents route around this area via the edge "
        "corridors even though the direct path to the goal passes through the zone."
    ),
    "reasoning": (
        "No trajectory ever includes movement into the restricted cells. Agents "
        "consistently use the (1,3) or (6,3) corridor to cross row y=3 — strong "
        "evidence for an access-control prohibition on the restricted zone."
    ),
}])


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

def _gen_push(env: MazeNamoEnvironment) -> list[Trajectory]:
    return generate_from_all_starts(env)


def _gen_spills(env: MazeNamoEnvironment) -> list[Trajectory]:
    return generate_from_all_starts_fn(env, avoid_forbidden_trajectory)


def _gen_greet(env: MazeNamoEnvironment) -> list[Trajectory]:
    # Only generate from the lower half (below the wall at y=3)
    return generate_from_all_starts_fn(env, greet_before_door_trajectory, min_y=4)


def _gen_restricted(env: MazeNamoEnvironment) -> list[Trajectory]:
    return generate_from_all_starts_fn(env, avoid_forbidden_trajectory)


SCENARIOS: dict[str, Scenario] = {
    "push_before_goal": Scenario(
        name="push_before_goal",
        norm_description="OBLIGATION (politeness): agent must push box B before reaching goal G",
        layout=_PUSH_BEFORE_GOAL_LAYOUT,
        mock_llm_response=_PUSH_MOCK,
        _gen=_gen_push,
    ),
    "avoid_spills": Scenario(
        name="avoid_spills",
        norm_description="PROHIBITION (cleanliness): agent must not step on spill cells",
        layout=_AVOID_SPILLS_LAYOUT,
        mock_llm_response=_SPILLS_MOCK,
        _gen=_gen_spills,
    ),
    "greet_before_door": Scenario(
        name="greet_before_door",
        norm_description="OBLIGATION (politeness, temporal): agent must GREET guard before crossing door",
        layout=_GREET_BEFORE_DOOR_LAYOUT,
        mock_llm_response=_GREET_MOCK,
        _gen=_gen_greet,
    ),
    "no_restricted_zone": Scenario(
        name="no_restricted_zone",
        norm_description="PROHIBITION (safety): agent must not enter the restricted zone",
        layout=_NO_RESTRICTED_ZONE_LAYOUT,
        mock_llm_response=_RESTRICTED_MOCK,
        _gen=_gen_restricted,
    ),
}
