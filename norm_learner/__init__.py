"""
norm_learner — incremental symbolic + LLM norm discovery.

Quick-start
-----------
    from norm_learner import NormLearner, MockLLM
    from norm_learner.environments.supermarket import SupermarketEnvironment

    env = SupermarketEnvironment(shelf_states=my_shelf_set)
    llm = MockLLM()   # replace with OpenAICompatibleLLM or LocalTransformersLLM
    learner = NormLearner(env, llm, llm_query_interval=20)

    for tau in trajectories:
        learner.process_trajectory(tau)

    for norm in learner.get_grounded_norms():
        print(norm.description)
"""

from .environment import EnvironmentInterface
from .learner import NormLearner
from .llm import LocalTransformersLLM, MockLLM, OpenAICompatibleLLM, LLMInterface
from .grounding import build_naturalization_prompt, build_norm_prompt
from .symbolic import NormLearnerState, norm_discovery
from .types import (
    AbstractNormHypothesis,
    DeonticModality,
    GroundedNorm,
    NORM_TYPE_PRIORITY,
    NormTypeCategory,
    Trajectory,
    TrajectoryRecord,
    WorldState,
)

__all__ = [
    # Core API
    "NormLearner",
    "EnvironmentInterface",
    # LLM backends
    "LLMInterface",
    "MockLLM",
    "OpenAICompatibleLLM",
    "LocalTransformersLLM",
    # Grounding helpers (for advanced use)
    "build_naturalization_prompt",
    "build_norm_prompt",
    # Symbolic primitives (for advanced use)
    "NormLearnerState",
    "norm_discovery",
    # Types
    "Trajectory",
    "WorldState",
    "AbstractNormHypothesis",
    "DeonticModality",
    "GroundedNorm",
    "NORM_TYPE_PRIORITY",
    "NormTypeCategory",
    "TrajectoryRecord",
]
