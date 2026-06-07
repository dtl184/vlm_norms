"""
Trajectory generation for MazeNamo.

Generators
----------
norm_following_trajectory        — push box before reaching goal (original norm)
norm_violating_trajectory        — direct path ignoring the box
avoid_forbidden_trajectory       — route around spills / restricted zones
greet_before_door_trajectory     — go to guard → GREET → continue to goal
generate_training_set            — multiple starts, one generator function
generate_from_all_starts         — all reachable cells, original norm
generate_from_all_starts_fn      — all reachable cells, custom generator
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from norm_learner.types import Trajectory

from .env import GREET, NORTH, PUSH, GridConfig
from .environment import MazeNamoEnvironment


# ---------------------------------------------------------------------------
# Original norm: push box before goal
# ---------------------------------------------------------------------------

def norm_following_trajectory(
    env: MazeNamoEnvironment,
    start_xy: tuple[int, int],
) -> Trajectory | None:
    """
    Build a norm-following trajectory from *start_xy* to goal via box push.

    Returns None if any phase is unreachable.
    """
    cfg = env.cfg
    assert cfg.required_box_start is not None, "push_before_goal needs a box"
    bx, by = cfg.required_box_start
    push_pos_xy = (bx, by + 1)   # one step SOUTH of box
    post_push_xy = (bx, by)      # where agent lands after pushing box north

    s_start = env.xy_to_state(*start_xy)
    s_push_pos = env.xy_to_state(*push_pos_xy)
    s_post_push = env.xy_to_state(*post_push_xy)
    s_goal = env.goal_state()

    plans1 = env.plan(s_start, s_push_pos)
    if not plans1:
        return None
    phase1 = plans1[0]

    push_step = [(s_push_pos, PUSH)]

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


# ---------------------------------------------------------------------------
# Norm: avoid spills / restricted zones
# ---------------------------------------------------------------------------

def avoid_forbidden_trajectory(
    env: MazeNamoEnvironment,
    start_xy: tuple[int, int],
) -> Trajectory | None:
    """
    Route from *start_xy* to the goal while treating forbidden cells (spills,
    restricted zones) as walls.  The norm-unaware planner finds a shorter path
    that cuts through the forbidden cells — the difference is what the norm
    learner detects.

    Returns None if no norm-aware path exists.
    """
    s_start = env.xy_to_state(*start_xy)
    s_goal = env.goal_state()
    plans = env.plan_norm_aware(s_start, s_goal)
    if not plans:
        return None
    norm_path = plans[0]
    # Only useful if the norm-aware path is actually longer than the direct path
    direct_cost = env.shortest_path_cost(s_start, s_goal)
    if len(norm_path) <= direct_cost:
        return None  # no detour needed — start is already past the obstacle
    return norm_path


# ---------------------------------------------------------------------------
# Norm: greet guard before passing through checkpoint
# ---------------------------------------------------------------------------

def greet_before_door_trajectory(
    env: MazeNamoEnvironment,
    start_xy: tuple[int, int],
) -> Trajectory | None:
    """
    Route from *start_xy* to guard, perform GREET, then continue to goal.

    The guard position is taken from ``env.cfg.guard_pos``.
    Returns None if either leg is unreachable.
    """
    cfg = env.cfg
    assert cfg.guard_pos is not None, "greet_before_door needs a guard position"

    s_start = env.xy_to_state(*start_xy)
    s_guard = env.xy_to_state(*cfg.guard_pos)
    s_goal = env.goal_state()

    # Only generate this trajectory if visiting guard costs extra steps
    direct_cost = env.shortest_path_cost(s_start, s_goal)

    plans1 = env.plan(s_start, s_guard)
    if not plans1:
        return None
    phase1 = plans1[0]

    greet_step = [(s_guard, GREET)]

    plans3 = env.plan(s_guard, s_goal)
    if not plans3:
        return None
    phase3 = plans3[0]

    full = phase1 + greet_step + phase3
    # Skip trajectories where the guard detour is free (start already at guard)
    if len(full) <= direct_cost:
        return None
    return full


# ---------------------------------------------------------------------------
# Bulk generators
# ---------------------------------------------------------------------------

def generate_training_set(
    env: MazeNamoEnvironment,
    start_positions: list[tuple[int, int]],
    repeat: int = 1,
) -> list[Trajectory]:
    """
    Generate *repeat* rounds of norm-following (push-before-goal) trajectories
    from each start position.
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
    Generate one norm-following (push-before-goal) trajectory from every
    passable, non-goal, non-box cell in the grid.
    """
    cfg = env.cfg
    skip: set = (exclude or set()) | {cfg.goal}
    if cfg.required_box_start is not None:
        skip.add(cfg.required_box_start)
    trajectories: list[Trajectory] = []
    for y in range(cfg.height):
        for x in range(cfg.width):
            if (x, y) in cfg.walls or (x, y) in skip:
                continue
            tau = norm_following_trajectory(env, (x, y))
            if tau:
                trajectories.append(tau)
    return trajectories


def generate_from_all_starts_fn(
    env: MazeNamoEnvironment,
    gen_fn,
    exclude: set[tuple[int, int]] | None = None,
    min_y: int = 0,
) -> list[Trajectory]:
    """
    Generate one trajectory from every eligible starting cell using *gen_fn*.

    Parameters
    ----------
    gen_fn   : callable(env, start_xy) -> Trajectory | None
    exclude  : extra cells to skip (goal and forbidden_cells are always skipped)
    min_y    : only start from rows y >= min_y (useful for greet_before_door)
    """
    cfg = env.cfg
    skip: set = (exclude or set()) | {cfg.goal} | set(cfg.forbidden_cells)
    if cfg.required_box_start is not None:
        skip.add(cfg.required_box_start)
    trajectories: list[Trajectory] = []
    for y in range(min_y, cfg.height):
        for x in range(cfg.width):
            if (x, y) in cfg.walls or (x, y) in skip:
                continue
            tau = gen_fn(env, (x, y))
            if tau:
                trajectories.append(tau)
    return trajectories
