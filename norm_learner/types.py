from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Core primitives — generic across all environments.
# State and Action must be hashable; int for grid worlds, tuple or str allowed.
# ---------------------------------------------------------------------------

State = Any
Action = Any
StateActionPair = tuple[State, Action]
Trajectory = list[StateActionPair]
WorldState = dict[str, Any]  # semantic feature dict extracted by the environment


# ---------------------------------------------------------------------------
# Abstract norm hypothesis — symbolic + contextual, not yet VLM-grounded
# ---------------------------------------------------------------------------

@dataclass
class AbstractNormHypothesis:
    """
    A norm candidate derived from the symbolic shortcut analysis, annotated
    with trajectory context (pre/post conditions).  The ``grounded_norm``
    field is filled in after a VLM query.
    """
    norm_type: Literal[
        "prohibition",
        "obligation",
        "disjunctive_prohibition",
        "disjunctive_obligation",
    ]
    # The state-action pairs the norm concerns
    symbolic_pairs: list[StateActionPair]
    # Indices of demonstrations that generated this hypothesis
    trajectory_ids: list[int]
    # Semantic features at the start / end of the relevant trajectory
    pre_conditions: WorldState
    post_conditions: WorldState
    # Keys whose value changed between pre and post
    state_delta: WorldState
    # Human-readable text built from env.describe_* — what the VLM sees
    symbolic_summary: str
    # Filled by VLM grounding; None until queried
    grounded_norm: str | None = None
    grounded_confidence: float = 0.0
    # How many distinct trajectories support this hypothesis
    supporting_count: int = 1


# ---------------------------------------------------------------------------
# Grounded norm — output of a VLM query
# ---------------------------------------------------------------------------

@dataclass
class GroundedNorm:
    """Natural-language norm produced by grounding abstract hypotheses via VLM."""
    description: str
    norm_type: str          # "prohibition" | "obligation" | "convention" | ...
    confidence: float       # 0.0 – 1.0 as reported by VLM
    reasoning: str
    # Indices into the abstract_hypotheses list that this norm was derived from
    source_hypothesis_ids: list[int]
    # Which VLM query round produced this (0-indexed)
    iteration: int


# ---------------------------------------------------------------------------
# Trajectory metadata record — stored alongside the symbolic demonstrations
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryRecord:
    """Per-trajectory semantic context used for grounding."""
    trajectory: Trajectory
    pre_conditions: WorldState
    post_conditions: WorldState
    # Keys that changed: {"key": {"before": v1, "after": v2}}
    delta: WorldState
