"""
Trajectory generation for MazeNamo.

Two generators are provided:

norm_following_trajectory
    The agent ALWAYS pushes the required box before reaching the goal.
    Implemented as three-phase A*:
      Phase 1: start  → push position (cell directly south of box)
      Phase 2: push   → one NORTH step that moves the box
      Phase 3: post-push agent position → goal

norm_violating_trajectory
    The agent goes straight to the goal, ignoring the box.
    (Used for comparison / sanity-check only — the norm learner sees only
    norm-following trajectories.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from norm_learner.types import Trajectory

from .env import NORTH, PUSH, GridConfig
from .environment import MazeNamoEnvironment


def norm_following_trajectory(
    env: MazeNamoEnvironment,
    start_xy: tuple[int, int],
) -> Trajectory | None:
    """
    Build a norm-following trajectory from *start_xy* to goal via box push.

    Returns None if any phase is unreachable.
    """
    cfg = env.cfg
    bx, by = cfg.required_box_start
    push_pos_xy = (bx, by + 1)   # one step SOUTH of box
    post_push_xy = (bx, by)      # where agent lands after pushing box north

    s_start = env.xy_to_state(*start_xy)
    s_push_pos = env.xy_to_state(*push_pos_xy)
    s_post_push = env.xy_to_state(*post_push_xy)
    s_goal = env.goal_state()

    # Phase 1: start → push position
    plans1 = env.plan(s_start, s_push_pos)
    if not plans1:
        return None
    phase1 = plans1[0]

    # Phase 2: the actual push — recorded as PUSH action, not NORTH
    push_step = [(s_push_pos, PUSH)]

    # Phase 3: post-push position → goal
    plans3 = env.plan(s_post_push, s_goal)
    if not plans3:
        return None
    phase3 = plans3[0]

    return phase1 + push_step + phase3


def norm_violating_trajectory(
    env: MazeNamoEnvironment,
    start_xy: tuple[int, int],
) -> Trajectory | None:
    """Direct path from *start_xy* to goal, ignoring the box."""
    s_start = env.xy_to_state(*start_xy)
    s_goal = env.goal_state()
    plans = env.plan(s_start, s_goal)
    return plans[0] if plans else None


def generate_training_set(
    env: MazeNamoEnvironment,
    start_positions: list[tuple[int, int]],
    repeat: int = 1,
) -> list[Trajectory]:
    """
    Generate *repeat* rounds of norm-following trajectories from each start
    position.  Skips any start from which the box push is unreachable.
    """
    trajectories: list[Trajectory] = []
    for _ in range(repeat):
        for start in start_positions:
            tau = norm_following_trajectory(env, start)
            if tau:
                trajectories.append(tau)
    return trajectories


def generate_from_all_starts(
    env: MazeNamoEnvironment,
    exclude: set[tuple[int, int]] | None = None,
) -> list[Trajectory]:
    """
    Generate one norm-following trajectory from every passable, non-goal,
    non-box cell in the grid.  Cells in *exclude* are skipped (use this to
    avoid re-generating starts already in the training set).
    """
    cfg = env.cfg
    skip = (exclude or set()) | {cfg.goal, cfg.required_box_start}
    trajectories: list[Trajectory] = []
    for y in range(cfg.height):
        for x in range(cfg.width):
            if (x, y) in cfg.walls or (x, y) in skip:
                continue
            tau = norm_following_trajectory(env, (x, y))
            if tau:
                trajectories.append(tau)
    return trajectories
