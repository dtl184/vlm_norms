"""
NormLearner — the top-level orchestrator.

Ties together the symbolic algorithm (norm_discovery), pre/post condition
extraction (extract_pre_post), abstract-hypothesis building
(build_abstract_hypotheses), and LLM grounding (build_norm_prompt /
parse_vlm_response) into a single, importable class.

Two-step LLM grounding
----------------------
1. Naturalization: convert the current trajectory's pre/post conditions to
   natural language via ``build_naturalization_prompt``.
2. Norm discovery: feed the NL description into the norm-discovery prompt.

Typical usage
-------------
    from norm_learner import NormLearner, MockLLM
    from norm_learner.environments.supermarket import SupermarketEnvironment

    env = SupermarketEnvironment()
    llm = MockLLM()                          # swap for Qwen in production
    learner = NormLearner(env, llm, llm_query_interval=20)

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
    build_naturalization_prompt,
    build_norm_prompt,
    extract_pre_post,
    parse_vlm_response,
)
from .llm import LLMInterface
from .symbolic import NormLearnerState, norm_discovery
from .types import (
    AbstractNormHypothesis,
    GroundedNorm,
    Trajectory,
    TrajectoryRecord,
)

logger = logging.getLogger(__name__)


class NormLearner:
    """
    Incremental norm learner with LLM grounding.

    Parameters
    ----------
    env : EnvironmentInterface
        The environment adapter (planner + semantic descriptions).
    llm : LLMInterface
        The LLM backend to use for grounding symbolic hypotheses.
    llm_query_interval : int
        Query the LLM every N processed trajectories.  Set to 0 to disable
        automatic queries (use ``force_llm_query()`` instead).
    on_llm_result : callable | None
        Optional callback invoked after each LLM query with the list of new
        GroundedNorm objects.  Useful for logging or downstream consumers.
    """

    def __init__(
        self,
        env: EnvironmentInterface,
        llm: LLMInterface,
        llm_query_interval: int = 1,
        on_llm_result: Callable[[list[GroundedNorm]], None] | None = None,
    ) -> None:
        self.env = env
        self.llm = llm
        self.llm_query_interval = llm_query_interval
        self.on_llm_result = on_llm_result

        self._symbolic_state = NormLearnerState()
        self._trajectory_records: list[TrajectoryRecord] = []
        self._grounded_norms: list[GroundedNorm] = []
        self._rejected_norms: list[GroundedNorm] = []
        # Full history of grounded norms across all LLM query rounds
        self._grounded_norm_history: list[list[GroundedNorm]] = []
        self._llm_query_count: int = 0
        self._last_prompt: str | None = None

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
        3. Query the LLM if the configured interval is reached.
        """
        record = extract_pre_post(tau, self.env)
        self._trajectory_records.append(record)

        self._symbolic_state = norm_discovery(tau, self.env, self._symbolic_state)
        self._verify_grounded_norms()

        n = len(self._trajectory_records)
        if self.llm_query_interval > 0 and n % self.llm_query_interval == 0:
            logger.info("Trajectory %d: triggering LLM query.", n)
            self._run_llm_query()

    def force_llm_query(self) -> list[GroundedNorm]:
        """
        Immediately run an LLM query regardless of the interval counter and
        return the resulting grounded norms.
        """
        self._run_llm_query()
        return list(self._grounded_norms)

    def get_grounded_norms(self) -> list[GroundedNorm]:
        """Return the most-recent list of LLM-grounded norms."""
        return list(self._grounded_norms)

    def get_rejected_norms(self) -> list[GroundedNorm]:
        """Return norms that were rejected because later trajectories contradicted them."""
        return list(self._rejected_norms)

    @property
    def last_prompt(self) -> str | None:
        """The LLM norm-discovery prompt used in the most recent query, or None if no query yet."""
        return self._last_prompt

    def get_grounded_norm_history(self) -> list[list[GroundedNorm]]:
        """Return grounded norms from every LLM query round, oldest first."""
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
    def llm_query_count(self) -> int:
        return self._llm_query_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _verify_grounded_norms(self) -> None:
        """Drop grounded norms whose symbolic basis has been pruned away."""
        current_leaf_pairs: set = set()
        for h in self._symbolic_state.hyp_prohibitions:
            for leaf in h.leaf_nodes():
                if leaf.sequence:
                    current_leaf_pairs.update(leaf.sequence)

        surviving: list[GroundedNorm] = []
        for norm in self._grounded_norms:
            if norm.snapshot_pairs and not any(
                p in current_leaf_pairs for p in norm.snapshot_pairs
            ):
                logger.info("Norm contradicted by new trajectory, rejecting: %s", norm.description)
                self._rejected_norms.append(norm)
            else:
                surviving.append(norm)
        self._grounded_norms = surviving

    def _run_llm_query(self) -> None:
        abstract_hyps = build_abstract_hypotheses(
            self._symbolic_state, self.env, self._trajectory_records
        )
        if not abstract_hyps:
            logger.debug("No hypotheses to ground yet; skipping LLM query.")
            return

        # --- Step 1: naturalise the latest trajectory's pre/post conditions --
        nl_context: str | None = None
        if self._trajectory_records:
            rec = self._trajectory_records[-1]
            current_steps = [
                f"{self.env.describe_action(a)} from {self.env.describe_state(s)}"
                for s, a in rec.trajectory
            ]
            nat_prompt = build_naturalization_prompt(rec, current_steps)
            logger.debug("Naturalization prompt:\n%s", nat_prompt)
            try:
                nl_context = self.llm.query(nat_prompt)
                logger.debug("Naturalization response:\n%s", nl_context)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Naturalization LLM call failed: %s", exc)

        # --- Step 2: norm-discovery prompt with NL context -------------------
        prompt = build_norm_prompt(
            abstract_hyps,
            len(self._trajectory_records),
            nl_context=nl_context,
            existing_norms=self._grounded_norms or None,
        )
        self._last_prompt = prompt
        logger.debug("Norm-discovery prompt (first 500 chars):\n%s", prompt[:500])

        try:
            response = self.llm.query(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM norm-discovery query failed: %s", exc)
            return

        logger.debug("LLM response (first 500 chars):\n%s", response[:500])

        new_norms = parse_vlm_response(
            response, abstract_hyps, self._llm_query_count
        )
        self._grounded_norms = new_norms
        self._grounded_norm_history.append(list(new_norms))
        self._llm_query_count += 1

        if self.on_llm_result is not None:
            self.on_llm_result(new_norms)

        logger.info(
            "LLM query #%d complete: %d grounded norms.",
            self._llm_query_count,
            len(new_norms),
        )
