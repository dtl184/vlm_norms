"""
Grounding layer: bridges the symbolic algorithm and the LLM.

Responsibilities
----------------
1. extract_pre_post           — build a TrajectoryRecord with semantic features.
2. build_abstract_hypotheses  — convert the current NormLearnerState into a
                                list of AbstractNormHypothesis objects annotated
                                with trajectory context and human-readable text.
3. build_naturalization_prompt — Step 1 of two-step LLM grounding: convert a
                                trajectory's pre/post conditions to NL.
4. build_norm_prompt          — Step 2: norm-discovery prompt using NL context.
5. parse_vlm_response         — parse the LLM's JSON reply into GroundedNorm objects.
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

_MOVEMENT_ACTION_NAMES = {"NORTH", "SOUTH", "EAST", "WEST"}


# ---------------------------------------------------------------------------
# Temporal ordering helpers
# ---------------------------------------------------------------------------

def _action_timing(
    demonstrations: list,
    non_movement_pairs: set,
    env: "EnvironmentInterface",
) -> dict:
    """
    For each non-movement action type, compute its mean normalized position
    (0.0 = start of trajectory, 1.0 = end) across all demonstrations.

    Returns a dict mapping action → (mean_position, phase_label).
    """
    action_types = {a for (_, a) in non_movement_pairs
                    if env.describe_action(a) not in _MOVEMENT_ACTION_NAMES}
    if not action_types or not demonstrations:
        return {}

    positions: dict = {a: [] for a in action_types}
    for tau in demonstrations:
        n = max(len(tau) - 1, 1)
        seen: set = set()
        for i, (_, a) in enumerate(tau):
            if a in action_types and a not in seen:
                positions[a].append(i / n)
                seen.add(a)

    result = {}
    for a, pos_list in positions.items():
        if not pos_list:
            continue
        mean = sum(pos_list) / len(pos_list)
        if mean < 0.35:
            phase = "early — prerequisite phase"
        elif mean > 0.70:
            phase = "late — completion phase"
        else:
            phase = "mid-trajectory"
        result[a] = (mean, phase)
    return result


def _temporal_order(
    demonstrations: list,
    non_movement_pairs: set,
    env: "EnvironmentInterface",
) -> list[tuple]:
    """
    Return (a, b) pairs of non-movement action types where action a's first
    occurrence consistently precedes action b's first occurrence across all
    demonstrations that contain both.
    """
    action_types = sorted(
        {a for (_, a) in non_movement_pairs
         if env.describe_action(a) not in _MOVEMENT_ACTION_NAMES}
    )
    if len(action_types) < 2:
        return []

    precedences = []
    for a in action_types:
        for b in action_types:
            if a == b:
                continue
            found_both = False
            consistent = True
            for tau in demonstrations:
                first_a = next((i for i, (_, act) in enumerate(tau) if act == a), None)
                first_b = next((i for i, (_, act) in enumerate(tau) if act == b), None)
                if first_a is not None and first_b is not None:
                    found_both = True
                    if first_a >= first_b:
                        consistent = False
                        break
            if found_both and consistent:
                precedences.append((a, b))
    return precedences


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

    # --- Non-movement actions present in every trajectory (H_O) ---------------
    # Only non-movement actions; movement actions are incidental path artefacts.
    if symbolic_state.hyp_obligations:
        n_traj = len(symbolic_state.demonstrations)
        all_h_o = sorted(symbolic_state.hyp_obligations, key=str)
        key_pairs = [sa for sa in all_h_o
                     if env.describe_action(sa[1]) not in _MOVEMENT_ACTION_NAMES]
        if key_pairs:
            timing = _action_timing(
                symbolic_state.demonstrations, set(key_pairs), env
            )
            order = _temporal_order(
                symbolic_state.demonstrations, set(key_pairs), env
            )

            action_lines = []
            for sa in key_pairs:
                desc = f"  {_format_pair(sa, env)}"
                if sa[1] in timing:
                    mean_pos, phase = timing[sa[1]]
                    desc += f" — occurs at {mean_pos:.0%} through trajectory ({phase})"
                action_lines.append(desc)
            joined = "\n".join(action_lines)

            temporal_note = ""
            if order:
                order_strs = [
                    f"{env.describe_action(a)} before {env.describe_action(b)}"
                    for (a, b) in order
                ]
                temporal_note = (
                    f"\n  Consistent ordering: " + ", ".join(order_strs)
                )

            hypotheses.append(
                AbstractNormHypothesis(
                    norm_type="obligation",
                    symbolic_pairs=key_pairs,
                    trajectory_ids=list(range(n_traj)),
                    pre_conditions={},
                    post_conditions={},
                    state_delta={},
                    symbolic_summary=(
                        f"Non-movement action(s) present in all {n_traj} trajectories:\n"
                        f"{joined}"
                        f"{temporal_note}"
                    ),
                    supporting_count=n_traj,
                )
            )

    # --- Action never observed (confirmed prohibition) ---------------------
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
                    f"Action never observed in any trajectory: "
                    f"{_format_pair(sa, env)}"
                ),
                supporting_count=len(symbolic_state.demonstrations),
            )
        )

    # --- Action always taken (confirmed obligation) -----------------------
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
                    f"Action taken in every trajectory: "
                    f"{_format_pair(sa, env)}"
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
            leaf_pairs = [lf.sequence[0] for lf in singleton_leaves]
            norm_type = (
                "disjunctive_prohibition" if len(leaf_pairs) > 1 else "prohibition"
            )
            if norm_type == "prohibition":
                candidate_desc = _format_pair(leaf_pairs[0], env)
                summary = f"Agent avoided: {candidate_desc}"
            else:
                joined = " or ".join(_format_pair(sa, env) for sa in leaf_pairs)
                candidate_desc = joined
                summary = f"Agent avoided one of: {candidate_desc}"
        else:
            leaf_pairs = all_leaf_pairs[:10]
            norm_type = "disjunctive_prohibition"
            area_descs = [_format_pair(sa, env) for sa in leaf_pairs[:5]]
            joined = ", ".join(area_descs)
            if len(leaf_pairs) > 5:
                joined += f" … (+{len(leaf_pairs)-5} more)"
            summary = f"Agent bypassed a shorter route; avoided area includes: {joined}"

        shortcut_desc = " → ".join(
            _format_pair(sa, env) for sa in h.original_shortcut[:5]
        )
        if len(h.original_shortcut) > 5:
            shortcut_desc += " … (truncated)"
        summary += (
            f". Detour: {len(h.observed_segment)} steps taken vs "
            f"{len(h.original_shortcut)}-step shortcut via [{shortcut_desc}]."
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

    # --- Confirmed ambiguous prohibition (disjunctive) --------------------
    for disj in symbolic_state.disjunctive_prohibitions:
        pairs = list(disj)
        descs = [_format_pair(sa, env) for sa in pairs]
        joined = " or ".join(descs)
        hypotheses.append(
            AbstractNormHypothesis(
                norm_type="disjunctive_prohibition",
                symbolic_pairs=pairs,
                trajectory_ids=[],
                pre_conditions={},
                post_conditions={},
                state_delta={},
                symbolic_summary=(
                    f"At least one of these is never taken "
                    f"(ambiguous — cannot distinguish which): {joined}"
                ),
                supporting_count=len(symbolic_state.demonstrations),
            )
        )

    return hypotheses


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def _format_world_state(ws: WorldState) -> str:
    if not ws:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in ws.items())


def build_naturalization_prompt(
    record: TrajectoryRecord,
    trajectory_steps: list[str],
) -> str:
    """
    Step 1 of two-step LLM grounding.

    Ask the LLM to describe in plain English what happened in this trajectory
    based on its pre/post conditions and step sequence.  The result is then
    passed as ``nl_context`` to ``build_norm_prompt``.
    """
    lines: list[str] = []
    lines.append(
        "An agent completed a task in the Proper Shopper environment. "
        "Describe in 2-3 plain English sentences what the agent did and "
        "what changed in the world. Be concise and factual."
    )
    lines.append("")
    lines.append(f"Pre-conditions : {_format_world_state(record.pre_conditions)}")
    lines.append(f"Post-conditions: {_format_world_state(record.post_conditions)}")
    lines.append(f"State changes  : {_format_world_state(record.delta)}")
    lines.append("")
    lines.append(f"Action sequence ({len(trajectory_steps)} steps):")
    for step in trajectory_steps:
        lines.append(f"  {step}")
    return "\n".join(lines)


def build_norm_prompt(
    abstract_hypotheses: list[AbstractNormHypothesis],
    n_trajectories: int,
    nl_context: str | None = None,
    existing_norms: list[GroundedNorm] | None = None,
) -> str:
    """
    Step 2 of two-step LLM grounding — norm-discovery prompt.

    Structure
    ---------
    1. Natural-language description of the latest trajectory (from Step 1).
    2. Symbolic hypotheses from the norm-discovery algorithm, presented
       neutrally as evidence rather than pre-labelled conclusions.
    3. Existing grounded norms from earlier rounds (for continuity).
    4. Output format instructions.
    """
    lines: list[str] = []

    # --- Natural-language context -----------------------------------------
    if nl_context:
        lines.append(f"LATEST TRAJECTORY ({n_trajectories} observed so far):")
        lines.append(nl_context.strip())
        lines.append("")

    # --- Symbolic evidence, grouped by type -------------------------------
    lines.append(
        "Abstract norm hypotheses as determined from symbolic analysis of trajectories"
    )
    lines.append("")

    # Group by broad category for readability
    # recurring   = [h for h in abstract_hypotheses
    #                if h.norm_type == "obligation" and h.supporting_count == n_trajectories]
    obligations = [h for h in abstract_hypotheses
                   if h.norm_type == "obligation" and h.supporting_count < n_trajectories]
    prohibitions = [h for h in abstract_hypotheses if h.norm_type == "prohibition"]
    disjunctive  = [h for h in abstract_hypotheses
                    if h.norm_type in ("disjunctive_prohibition", "disjunctive_obligation")]

    groups = [
        #("Recurring non-movement actions (appear in every trajectory)", recurring),
        ("Obligations", obligations),
        ("Prohibitions", prohibitions),
        ("Disjunctive Hypotheses", disjunctive),
    ]
    for group_label, group in groups:
        if not group:
            continue
        shown = group[:_MAX_HYPOTHESES_IN_PROMPT]
        lines.append(f"{group_label}:")
        for hyp in shown:
            for ln in hyp.symbolic_summary.strip().split("\n"):
                lines.append(f"  {ln}")
            if hyp.pre_conditions:
                lines.append(f"  Pre:  {_format_world_state(hyp.pre_conditions)}")
            if hyp.post_conditions:
                lines.append(f"  Post: {_format_world_state(hyp.post_conditions)}")
        if len(group) > _MAX_HYPOTHESES_IN_PROMPT:
            lines.append(f"  … and {len(group) - _MAX_HYPOTHESES_IN_PROMPT} more")
        lines.append("")

    # --- Existing grounded norms ------------------------------------------
    if existing_norms:
        lines.append("Previously Identified Norms:")
        for n in existing_norms:
            lines.append(f"  [{n.modality.upper()}] {n.description}")
        lines.append("")

    # --- Task and output format -------------------------------------------
    lines.append(
        "TASK: Given the symbolic patterns and trajectory context above, identify "
        "social norms that best explain why the agents consistently deviate from the "
        "shortest path or include certain actions. A norm should capture the underlying "
        "social rule, not merely describe what was observed.\n"
        "\n"
        "Express each norm as a 4-tuple ⟨Context, Modality, Action, Type⟩:\n"
        "  context: list of contextual conditions as strings\n"
        "  modality: \"obligatory\" | \"forbidden\" | \"permissible\"\n"
        "  action: action schema string, e.g. \"push(agent, box)\"\n"
        "  type: \"safety\" | \"cleanliness\" | \"politeness\" | \"convenience\"\n"
        "\n"
        "Output only a JSON array, no prose before or after:\n"
        '[{"context": ["cond1", ...], "modality": "obligatory|forbidden|permissible", '
        '"action": "...", "norm_type": "safety|cleanliness|politeness|convenience"}]'
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

    _VALID_MODALITIES = {"obligatory", "forbidden", "permissible"}
    _VALID_CATEGORIES = {"safety", "cleanliness", "politeness", "convenience"}

    grounded: list[GroundedNorm] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        description = item.get("description", "").strip()
        if not description:
            continue

        raw_modality = item.get("modality", item.get("type", "")).lower()
        # Map legacy "obligation"/"prohibition" to canonical modality strings
        if raw_modality == "obligation":
            raw_modality = "obligatory"
        elif raw_modality == "prohibition":
            raw_modality = "forbidden"
        elif raw_modality == "permission":
            raw_modality = "permissible"
        modality = raw_modality if raw_modality in _VALID_MODALITIES else "obligatory"

        raw_category = item.get("norm_type", "").lower()
        norm_type = raw_category if raw_category in _VALID_CATEGORIES else "convenience"

        context = item.get("context", [])
        if not isinstance(context, list):
            context = [str(context)] if context else []

        grounded.append(
            GroundedNorm(
                context=context,
                modality=modality,
                action=item.get("action", "").strip(),
                norm_type=norm_type,
                description=description,
                source_hypothesis_ids=list(range(len(abstract_hypotheses))),
                iteration=iteration,
                snapshot_pairs=list(all_pairs),
            )
        )

    return grounded
