"""
NormLearner — the top-level orchestrator.

Ties together the symbolic algorithm (norm_discovery), pre/post condition
extraction (extract_pre_post), abstract-hypothesis building
(build_abstract_hypotheses), and VLM grounding (build_vlm_prompt /
parse_vlm_response) into a single, importable class.

Typical usage
-------------
    from norm_learner import NormLearner, MockVLM
    from norm_learner.environments.supermarket import SupermarketEnvironment

    env = SupermarketEnvironment()
    vlm = MockVLM()                          # swap for Qwen in production
    learner = NormLearner(env, vlm, vlm_query_interval=20)

    for tau in trajectories:
        learner.process_trajectory(tau)

    print(learner.get_grounded_norms())
    print(learner.get_symbolic_state().prohibitions)
"""

from __future__ import annotations

import logging
from typing import Callable

from .environment import EnvironmentInterface
from .grounding import (
    build_abstract_hypotheses,
    build_vlm_prompt,
    extract_pre_post,
    parse_vlm_response,
)
from .symbolic import NormLearnerState, norm_discovery
from .types import (
    AbstractNormHypothesis,
    GroundedNorm,
    Trajectory,
    TrajectoryRecord,
)
from .vlm import VLMInterface

logger = logging.getLogger(__name__)


class NormLearner:
    """
    Incremental norm learner with VLM grounding.

    Parameters
    ----------
    env : EnvironmentInterface
        The environment adapter (planner + semantic descriptions).
    vlm : VLMInterface
        The VLM backend to use for grounding symbolic hypotheses.
    vlm_query_interval : int
        Query the VLM every N processed trajectories.  Set to 0 to disable
        automatic queries (use ``force_vlm_query()`` instead).
    on_vlm_result : callable | None
        Optional callback invoked after each VLM query with the list of new
        GroundedNorm objects.  Useful for logging or downstream consumers.
    """

    def __init__(
        self,
        env: EnvironmentInterface,
        vlm: VLMInterface,
        vlm_query_interval: int = 10,
        on_vlm_result: Callable[[list[GroundedNorm]], None] | None = None,
    ) -> None:
        self.env = env
        self.vlm = vlm
        self.vlm_query_interval = vlm_query_interval
        self.on_vlm_result = on_vlm_result

        self._symbolic_state = NormLearnerState()
        self._trajectory_records: list[TrajectoryRecord] = []
        self._grounded_norms: list[GroundedNorm] = []
        # Full history of grounded norms across all VLM query rounds
        self._grounded_norm_history: list[list[GroundedNorm]] = []
        self._vlm_query_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_trajectory(self, tau: Trajectory) -> None:
        """
        Ingest one new trajectory.

        Steps
        -----
        1. Extract pre/post conditions from the environment.
        2. Run the symbolic norm-discovery update.
        3. Query the VLM if the configured interval is reached.
        """
        record = extract_pre_post(tau, self.env)
        self._trajectory_records.append(record)

        self._symbolic_state = norm_discovery(tau, self.env, self._symbolic_state)

        n = len(self._trajectory_records)
        if self.vlm_query_interval > 0 and n % self.vlm_query_interval == 0:
            logger.info("Trajectory %d: triggering VLM query.", n)
            self._run_vlm_query()

    def force_vlm_query(self) -> list[GroundedNorm]:
        """
        Immediately run a VLM query regardless of the interval counter and
        return the resulting grounded norms.
        """
        self._run_vlm_query()
        return list(self._grounded_norms)

    def get_grounded_norms(self) -> list[GroundedNorm]:
        """Return the most-recent list of VLM-grounded norms."""
        return list(self._grounded_norms)

    def get_grounded_norm_history(self) -> list[list[GroundedNorm]]:
        """Return grounded norms from every VLM query round, oldest first."""
        return list(self._grounded_norm_history)

    def get_symbolic_state(self) -> NormLearnerState:
        """Return the current symbolic learner state (mutable reference)."""
        return self._symbolic_state

    def get_abstract_hypotheses(self) -> list[AbstractNormHypothesis]:
        """Build and return the current abstract hypotheses without querying the VLM."""
        return build_abstract_hypotheses(
            self._symbolic_state, self.env, self._trajectory_records
        )

    def get_trajectory_records(self) -> list[TrajectoryRecord]:
        """Return per-trajectory semantic records."""
        return list(self._trajectory_records)

    @property
    def n_trajectories(self) -> int:
        return len(self._trajectory_records)

    @property
    def vlm_query_count(self) -> int:
        return self._vlm_query_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_vlm_query(self) -> None:
        abstract_hyps = build_abstract_hypotheses(
            self._symbolic_state, self.env, self._trajectory_records
        )
        if not abstract_hyps:
            logger.debug("No hypotheses to ground yet; skipping VLM query.")
            return

        prompt = build_vlm_prompt(
            abstract_hyps,
            self._grounded_norms,
            len(self._trajectory_records),
        )
        logger.debug("VLM prompt (first 500 chars):\n%s", prompt[:500])

        try:
            response = self.vlm.query(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("VLM query failed: %s", exc)
            return

        logger.debug("VLM response (first 500 chars):\n%s", response[:500])

        new_norms = parse_vlm_response(
            response, abstract_hyps, self._vlm_query_count
        )
        self._grounded_norms = new_norms
        self._grounded_norm_history.append(list(new_norms))
        self._vlm_query_count += 1

        if self.on_vlm_result is not None:
            self.on_vlm_result(new_norms)

        logger.info(
            "VLM query #%d complete: %d grounded norms.",
            self._vlm_query_count,
            len(new_norms),
        )
