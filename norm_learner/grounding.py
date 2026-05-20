"""
Grounding layer: bridges the symbolic algorithm and the VLM.

Responsibilities
----------------
1. extract_pre_post      — build a TrajectoryRecord with semantic features.
2. build_abstract_hypotheses — convert the current NormLearnerState into a
                               list of AbstractNormHypothesis objects annotated
                               with trajectory context and human-readable text.
3. build_vlm_prompt      — serialise hypotheses + history into a prompt string.
4. parse_vlm_response    — parse the VLM's JSON reply into GroundedNorm objects.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from .types import (
    AbstractNormHypothesis,
    GroundedNorm,
    Trajectory,
    TrajectoryRecord,
    WorldState,
)

if TYPE_CHECKING:
    from .environment import EnvironmentInterface
    from .symbolic import NormLearnerState

logger = logging.getLogger(__name__)

# Maximum number of hypotheses included in a single VLM prompt to avoid
# overwhelming the model's context window.
_MAX_HYPOTHESES_IN_PROMPT = 8


# ---------------------------------------------------------------------------
# Pre/post condition extraction
# ---------------------------------------------------------------------------

def _diff_world_states(pre: WorldState, post: WorldState) -> WorldState:
    """Return keys whose value differs between *pre* and *post*."""
    delta: WorldState = {}
    for k in set(pre) | set(post):
        v_pre = pre.get(k)
        v_post = post.get(k)
        if v_pre != v_post:
            delta[k] = {"before": v_pre, "after": v_post}
    return delta


def extract_pre_post(
    tau: Trajectory,
    env: "EnvironmentInterface",
) -> TrajectoryRecord:
    """
    Extract a TrajectoryRecord for *tau* using the environment's semantic
    feature extractor.

    The pre-conditions are derived from the first state in *tau*; the
    post-conditions from the last state.  If the environment provides a
    ``successor`` method the true terminal state is used; otherwise the last
    *observed* state in the trajectory is used as a proxy.
    """
    if not tau:
        return TrajectoryRecord(tau, {}, {}, {})

    start_state = tau[0][0]
    last_state, last_action = tau[-1]

    terminal = env.successor(last_state, last_action)
    end_state = terminal if terminal is not None else last_state

    pre = env.extract_semantic_features(start_state)
    post = env.extract_semantic_features(end_state)
    delta = _diff_world_states(pre, post)

    return TrajectoryRecord(
        trajectory=tau,
        pre_conditions=pre,
        post_conditions=post,
        delta=delta,
    )


# ---------------------------------------------------------------------------
# Building abstract hypotheses from the symbolic state
# ---------------------------------------------------------------------------

def _format_pair(sa: tuple, env: "EnvironmentInterface") -> str:
    s, a = sa
    return f"{env.describe_action(a)} from {env.describe_state(s)}"


def build_abstract_hypotheses(
    symbolic_state: "NormLearnerState",
    env: "EnvironmentInterface",
    trajectory_records: list[TrajectoryRecord],
) -> list[AbstractNormHypothesis]:
    """
    Convert the current symbolic state into a list of AbstractNormHypothesis.

    Each hypothesis is annotated with the pre/post conditions of the
    trajectory that generated it (looked up via source_trajectory_id on the
    HypTree) so the VLM has richer context.
    """
    hypotheses: list[AbstractNormHypothesis] = []

    def _record_for(traj_id: int | None) -> TrajectoryRecord:
        if traj_id is not None and traj_id < len(trajectory_records):
            return trajectory_records[traj_id]
        return TrajectoryRecord([], {}, {}, {})

    # --- Hypothesised obligations (H_O) — placed first; strongest signal ----
    if symbolic_state.hyp_obligations:
        n_traj = len(symbolic_state.demonstrations)
        h_o_pairs = sorted(symbolic_state.hyp_obligations, key=str)
        descs = [_format_pair(sa, env) for sa in h_o_pairs]
        joined = "\n  ".join(f"[{d}]" for d in descs)
        hypotheses.append(
            AbstractNormHypothesis(
                norm_type="obligation",
                symbolic_pairs=h_o_pairs,
                trajectory_ids=list(range(n_traj)),
                pre_conditions={},
                post_conditions={},
                state_delta={},
                symbolic_summary=(
                    f"COMMON ACTIONS — present in all {n_traj} demonstrations:\n"
                    f"  {joined}"
                ),
                supporting_count=n_traj,
            )
        )

    # --- Confirmed prohibitions -------------------------------------------
    for sa in sorted(symbolic_state.prohibitions, key=str):
        hypotheses.append(
            AbstractNormHypothesis(
                norm_type="prohibition",
                symbolic_pairs=[sa],
                trajectory_ids=[],
                pre_conditions={},
                post_conditions={},
                state_delta={},
                symbolic_summary=(
                    f"CONFIRMED PROHIBITION: agent never performs "
                    f"[{_format_pair(sa, env)}]"
                ),
                supporting_count=len(symbolic_state.demonstrations),
            )
        )

    # --- Confirmed obligations --------------------------------------------
    for sa in sorted(symbolic_state.obligations, key=str):
        hypotheses.append(
            AbstractNormHypothesis(
                norm_type="obligation",
                symbolic_pairs=[sa],
                trajectory_ids=[],
                pre_conditions={},
                post_conditions={},
                state_delta={},
                symbolic_summary=(
                    f"CONFIRMED OBLIGATION: agent always performs "
                    f"[{_format_pair(sa, env)}]"
                ),
                supporting_count=len(symbolic_state.demonstrations),
            )
        )

    # --- Hypothesised prohibitions (from HypTrees) ------------------------
    # Include ALL active hypothesis trees, not just those refined to singletons.
    # Fresh trees carry the full shortcut as candidate pairs; they are still
    # valuable for the VLM even before further trajectories narrow them down.
    for h in symbolic_state.hyp_prohibitions:
        leaves = h.leaf_nodes()
        if not leaves:
            continue

        # Collect all state-action pairs still in the leaves
        all_leaf_pairs: list = []
        for lf in leaves:
            if lf.sequence:
                all_leaf_pairs.extend(lf.sequence)

        singleton_leaves = [
            lf for lf in leaves
            if lf.sequence is not None and len(lf.sequence) == 1
        ]

        # Classify the hypothesis
        if singleton_leaves and all(
            lf.sequence is not None and len(lf.sequence) == 1 for lf in leaves
        ):
            # All leaves are singletons — near-confirmed
            leaf_pairs = [lf.sequence[0] for lf in singleton_leaves]
            norm_type = (
                "disjunctive_prohibition" if len(leaf_pairs) > 1 else "prohibition"
            )
            if norm_type == "prohibition":
                candidate_desc = f"[{_format_pair(leaf_pairs[0], env)}]"
                prefix = "HYPOTHESISED PROHIBITION (nearly confirmed):"
            else:
                joined = " OR ".join(f"[{_format_pair(sa, env)}]" for sa in leaf_pairs)
                candidate_desc = joined
                prefix = "DISJUNCTIVE PROHIBITION:"
            summary = f"{prefix} one of {candidate_desc} is likely prohibited."
        else:
            # Unrefined: the shortcut area has multiple candidate pairs
            leaf_pairs = all_leaf_pairs[:10]  # cap for readability
            norm_type = "disjunctive_prohibition"
            area_descs = [_format_pair(sa, env) for sa in leaf_pairs[:5]]
            joined = ", ".join(f"[{d}]" for d in area_descs)
            if len(leaf_pairs) > 5:
                joined += f" … (+{len(leaf_pairs)-5} more)"
            summary = (
                f"SHORTCUT AVOIDANCE: agent took a longer path; "
                f"at least one of these pairs may be prohibited: {joined}."
            )

        # Describe the shortcut itself for VLM context
        shortcut_desc = " → ".join(
            _format_pair(sa, env) for sa in h.original_shortcut[:5]
        )
        if len(h.original_shortcut) > 5:
            shortcut_desc += " … (truncated)"
        summary += (
            f"  Shortcut not taken: [{shortcut_desc}]. "
            f"Observed segment: {len(h.observed_segment)} steps vs "
            f"shortcut: {len(h.original_shortcut)} steps."
        )

        record = _record_for(h.source_trajectory_id)
        hypotheses.append(
            AbstractNormHypothesis(
                norm_type=norm_type,
                symbolic_pairs=leaf_pairs,
                trajectory_ids=(
                    [h.source_trajectory_id]
                    if h.source_trajectory_id is not None
                    else []
                ),
                pre_conditions=record.pre_conditions,
                post_conditions=record.post_conditions,
                state_delta=record.delta,
                symbolic_summary=summary,
                supporting_count=1,
            )
        )

    # --- Disjunctive prohibitions (confirmed but ambiguous) ---------------
    for disj in symbolic_state.disjunctive_prohibitions:
        pairs = list(disj)
        descs = [_format_pair(sa, env) for sa in pairs]
        joined = " OR ".join(f"[{d}]" for d in descs)
        hypotheses.append(
            AbstractNormHypothesis(
                norm_type="disjunctive_prohibition",
                symbolic_pairs=pairs,
                trajectory_ids=[],
                pre_conditions={},
                post_conditions={},
                state_delta={},
                symbolic_summary=(
                    f"DISJUNCTIVE PROHIBITION (confirmed): "
                    f"at least one of {joined} is prohibited."
                ),
                supporting_count=len(symbolic_state.demonstrations),
            )
        )

    return hypotheses


# ---------------------------------------------------------------------------
# VLM prompt construction
# ---------------------------------------------------------------------------

def _format_world_state(ws: WorldState) -> str:
    if not ws:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in ws.items())


def build_vlm_prompt(
    abstract_hypotheses: list[AbstractNormHypothesis],
    grounded_norms: list[GroundedNorm],
    n_trajectories: int,
    current_trajectory_steps: list[str] | None = None,
) -> str:
    """
    Build the prompt sent to the VLM.

    Structure
    ---------
    1. Actions present in every observed trajectory (strongest obligation signal).
    2. Current trajectory step sequence.
    3. Previously grounded norms (for iterative refinement).
    4. JSON output instructions.
    """
    lines: list[str] = []

    # --- Common actions (H_O) — the primary signal ------------------------
    h_o_hyp = next(
        (h for h in abstract_hypotheses if h.norm_type == "obligation" and h.supporting_count == n_trajectories),
        None,
    )
    if h_o_hyp and n_trajectories > 0:
        lines.append(
            f"I have observed {n_trajectories} agents completing a task. "
            f"The following actions appear in EVERY trajectory:"
        )
        for sa_desc in h_o_hyp.symbolic_summary.split("\n")[1:]:
            lines.append(sa_desc)
    else:
        lines.append(
            f"I have observed {n_trajectories} agents completing a task. "
            f"Not enough data yet to identify common actions."
        )
    lines.append("")

    # --- Current trajectory -----------------------------------------------
    if current_trajectory_steps:
        lines.append(f"LATEST TRAJECTORY ({len(current_trajectory_steps)} steps):")
        for step in current_trajectory_steps:
            lines.append(f"  {step}")
        lines.append("")

    # --- Previously grounded norms ----------------------------------------
    if grounded_norms:
        lines.append("PREVIOUSLY IDENTIFIED NORMS (refine or drop if contradicted):")
        for gn in grounded_norms:
            lines.append(f"  [{gn.norm_type.upper()}] {gn.description}")
        lines.append("")

    # --- Output instructions ----------------------------------------------
    lines.append(
        "What social norm(s) are the agents following? "
        "Actions present in every trajectory are the strongest evidence for obligations.\n"
        "Output only a JSON array, no prose:\n"
        '[{"type": "obligation"|"prohibition"|"permission", '
        '"description": "...", "reasoning": "..."}]'
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VLM response parsing
# ---------------------------------------------------------------------------

def parse_vlm_response(
    response: str,
    abstract_hypotheses: list[AbstractNormHypothesis],
    iteration: int,
) -> list[GroundedNorm]:
    """
    Parse the VLM's JSON reply into a list of GroundedNorm objects.

    Handles responses where JSON is embedded in surrounding prose (e.g. fenced
    code blocks) and falls back gracefully on parse errors.
    """
    # Try to extract JSON from a fenced code block first
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    json_text = fenced.group(1).strip() if fenced else response.strip()

    # Fallback: find the first '[' … ']' span
    if not json_text.startswith("["):
        start = json_text.find("[")
        end = json_text.rfind("]")
        if start != -1 and end != -1:
            json_text = json_text[start : end + 1]

    try:
        raw_list = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse VLM response as JSON: %s", exc)
        logger.debug("Raw response: %s", response)
        return []

    if not isinstance(raw_list, list):
        logger.warning("VLM response is not a JSON array.")
        return []

    all_pairs: list = []
    for hyp in abstract_hypotheses:
        all_pairs.extend(hyp.symbolic_pairs)

    grounded: list[GroundedNorm] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        description = item.get("description", "").strip()
        if not description:
            continue
        grounded.append(
            GroundedNorm(
                description=description,
                norm_type=item.get("type", "unknown"),
                reasoning=item.get("reasoning", ""),
                source_hypothesis_ids=list(range(len(abstract_hypotheses))),
                iteration=iteration,
                snapshot_pairs=list(all_pairs),
            )
        )

    return grounded
