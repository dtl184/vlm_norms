"""
norm_learner — incremental symbolic + VLM norm discovery.

Quick-start
-----------
    from norm_learner import NormLearner, MockVLM
    from norm_learner.environments.supermarket import SupermarketEnvironment

    env = SupermarketEnvironment(shelf_states=my_shelf_set)
    vlm = MockVLM()   # replace with OpenAICompatibleVLM or LocalTransformersVLM
    learner = NormLearner(env, vlm, vlm_query_interval=20)

    for tau in trajectories:
        learner.process_trajectory(tau)

    for norm in learner.get_grounded_norms():
        print(norm.description)
"""

from .environment import EnvironmentInterface
from .learner import NormLearner
from .symbolic import NormLearnerState, norm_discovery
from .types import (
    AbstractNormHypothesis,
    GroundedNorm,
    Trajectory,
    TrajectoryRecord,
    WorldState,
)
from .vlm import LocalTransformersVLM, MockVLM, OpenAICompatibleVLM, VLMInterface

__all__ = [
    # Core API
    "NormLearner",
    "EnvironmentInterface",
    # VLM backends
    "VLMInterface",
    "MockVLM",
    "OpenAICompatibleVLM",
    "LocalTransformersVLM",
    # Symbolic primitives (for advanced use)
    "NormLearnerState",
    "norm_discovery",
    # Types
    "Trajectory",
    "WorldState",
    "AbstractNormHypothesis",
    "GroundedNorm",
    "TrajectoryRecord",
]
