from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

# ---------------------------------------------------------------------------
# Core primitives — generic across all environments.
# State and Action must be hashable; int for grid worlds, tuple or str allowed.
# ---------------------------------------------------------------------------

State = Any
Action = Any
StateActionPair = Tuple[State, Action]
Trajectory = List[StateActionPair]
WorldState = Dict[str, Any]  # semantic feature dict extracted by the environment


# ---------------------------------------------------------------------------
# Paper norm representation — N = ⟨C_N, M_N, α_N, Type_N⟩
# "A Framework for Norm-Based Reference Resolution"
# ---------------------------------------------------------------------------

class DeonticModality(str, Enum):
    """M_N: the deontic modality of a norm."""
    OBLIGATORY = "obligatory"
    FORBIDDEN = "forbidden"
    PERMISSIBLE = "permissible"


class NormTypeCategory(str, Enum):
    """Type_N: norm category with priority SAFETY ≻ CLEANLINESS ≻ POLITENESS ≻ CONVENIENCE."""
    SAFETY = "safety"
    CLEANLINESS = "cleanliness"
    POLITENESS = "politeness"
    CONVENIENCE = "convenience"


# Priority: higher value = higher priority
NORM_TYPE_PRIORITY: Dict[NormTypeCategory, int] = {
    NormTypeCategory.SAFETY: 3,
    NormTypeCategory.CLEANLINESS: 2,
    NormTypeCategory.POLITENESS: 1,
    NormTypeCategory.CONVENIENCE: 0,
}


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
    # How many distinct trajectories support this hypothesis
    supporting_count: int = 1


# ---------------------------------------------------------------------------
# Grounded norm — output of a VLM query
# ---------------------------------------------------------------------------

@dataclass
class GroundedNorm:
    """
    LLM-grounded norm in the paper's 4-tuple representation.

    N = ⟨C_N, M_N, α_N, Type_N⟩  (see "A Framework for Norm-Based
    Reference Resolution")

    Fields
    ------
    context   : C_N — list of contextual conditions, e.g. ["setting(grid)", "task(navigate)"]
    modality  : M_N — deontic modality string ("obligatory" | "forbidden" | "permissible")
    action    : α_N — action schema string, e.g. "push(agent, box)"
    norm_type : Type_N — norm category ("safety" | "cleanliness" | "politeness" | "convenience")
    """
    # Paper 4-tuple
    context: List[str]       # C_N
    modality: str            # M_N — use DeonticModality values
    action: str              # α_N
    norm_type: str           # Type_N — use NormTypeCategory values
    # Human-readable
    description: str
    reasoning: str
    # Learner bookkeeping
    source_hypothesis_ids: List[int]
    iteration: int
    snapshot_pairs: List[StateActionPair] = field(default_factory=list)


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
