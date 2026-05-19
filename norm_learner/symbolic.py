"""
Core symbolic norm-discovery algorithm.

This module is a generalized version of the original algorithm: the only
dependency on a specific environment is through EnvironmentInterface, so it
works for any grid world, social navigation benchmark, or other MDP.

Key data structures
-------------------
HypTree  — a tree of shortcut hypothesis subtrees; each leaf is a sequence of
           state-action pairs that might be prohibited.
NormLearnerState — the full mutable state of the learner: confirmed/hypothesized
           obligations, prohibitions, and permissions accumulated so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import StateActionPair, Trajectory
from .environment import EnvironmentInterface


# ---------------------------------------------------------------------------
# Hypothesis tree
# ---------------------------------------------------------------------------

@dataclass
class HypTreeNode:
    sequence: list[StateActionPair] | None = None
    children: list["HypTreeNode"] = field(default_factory=list)
    parent: Optional["HypTreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaf_nodes(self) -> list["HypTreeNode"]:
        if self.is_leaf:
            return [self]
        leaves: list[HypTreeNode] = []
        for child in self.children:
            leaves.extend(child.leaf_nodes())
        return leaves


@dataclass
class HypTree:
    root: HypTreeNode
    observed_segment: Trajectory = field(default_factory=list)
    original_shortcut: Trajectory = field(default_factory=list)
    source_trajectory_id: int | None = None

    def leaf_nodes(self) -> list[HypTreeNode]:
        return self.root.leaf_nodes()

    def leaf_pairs(self) -> set[StateActionPair]:
        pairs: set[StateActionPair] = set()
        for leaf in self.leaf_nodes():
            if leaf.sequence:
                pairs.update(leaf.sequence)
        return pairs


# ---------------------------------------------------------------------------
# Learner state
# ---------------------------------------------------------------------------

@dataclass
class NormLearnerState:
    obligations: set[StateActionPair] = field(default_factory=set)
    prohibitions: set[StateActionPair] = field(default_factory=set)
    permissions: set[StateActionPair] = field(default_factory=set)

    hyp_obligations: set[StateActionPair] | None = None
    hyp_prohibitions: list[HypTree] = field(default_factory=list)

    disjunctive_prohibitions: list[set[StateActionPair]] = field(default_factory=list)
    disjunctive_obligations: list[set[StateActionPair]] = field(default_factory=list)

    demonstrations: list[Trajectory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tree operations
# ---------------------------------------------------------------------------

def delete_and_prune(tree: HypTree, node: HypTreeNode) -> HypTree | None:
    parent = node.parent
    if parent is not None:
        parent.children = [c for c in parent.children if c is not node]
    else:
        return None

    p = parent
    while p is not None and len(p.children) == 0:
        grandparent = p.parent
        if grandparent is not None:
            grandparent.children = [c for c in grandparent.children if c is not p]
        else:
            return None
        p = grandparent

    leaves = tree.leaf_nodes()
    if len(leaves) == 0:
        return None
    if len(leaves) == 1 and leaves[0] is tree.root and (
        tree.root.sequence is None or len(tree.root.sequence) == 0
    ):
        return None
    return tree


def traverse_and_split(tree: HypTree, x: StateActionPair) -> HypTree | None:
    if tree is None:
        return None

    stack: list[HypTreeNode] = [tree.root]
    while stack:
        node = stack.pop()
        if node.is_leaf and node.sequence is not None:
            if x in node.sequence:
                if len(node.sequence) == 1:
                    tree = delete_and_prune(tree, node)
                    if tree is None:
                        return None
                else:
                    idx = node.sequence.index(x)
                    left_seq = node.sequence[:idx]
                    right_seq = node.sequence[idx + 1:]
                    node.sequence = None
                    node.children = []
                    if left_seq:
                        node.children.append(HypTreeNode(sequence=left_seq, parent=node))
                    if right_seq:
                        node.children.append(HypTreeNode(sequence=right_seq, parent=node))
                    if len(node.children) == 0:
                        tree = delete_and_prune(tree, node)
                        if tree is None:
                            return None
        else:
            for child in list(node.children):
                stack.append(child)
    return tree


def is_subsumed_by(new_tree: HypTree, existing_tree: HypTree) -> bool:
    return new_tree.leaf_pairs().issubset(existing_tree.leaf_pairs())


def make_hypothesis_tree(
    shortcut: Trajectory,
    observed_segment: Trajectory,
    source_trajectory_id: int,
) -> HypTree:
    root = HypTreeNode(sequence=list(shortcut))
    return HypTree(
        root=root,
        observed_segment=list(observed_segment),
        original_shortcut=list(shortcut),
        source_trajectory_id=source_trajectory_id,
    )


# ---------------------------------------------------------------------------
# Core sanitization (no ground-truth dependency)
# ---------------------------------------------------------------------------

def sanitize_norm_state(state: NormLearnerState) -> NormLearnerState:
    """
    Remove logical contradictions and deduplicate disjunctive prohibitions.
    Does NOT filter against any ground-truth oracle — that belongs in
    environment-specific evaluation code.
    """
    state.prohibitions -= state.permissions
    state.prohibitions -= state.obligations

    if state.hyp_obligations is not None:
        # Only remove pairs already confirmed as obligations — H_O tracks
        # *candidates*, and obligations are always also in P, so subtracting P
        # would empty H_O entirely (bug: H_O ⊆ P by construction).
        state.hyp_obligations -= state.obligations

    new_disj: list[set[StateActionPair]] = []
    seen: set[frozenset] = set()
    for disj in state.disjunctive_prohibitions:
        cleaned = {
            sa for sa in disj
            if sa not in state.permissions and sa not in state.obligations
        }
        if len(cleaned) == 0:
            continue
        if len(cleaned) == 1:
            state.prohibitions.update(cleaned)
            continue
        frozen = frozenset(cleaned)
        if frozen not in seen:
            seen.add(frozen)
            new_disj.append(cleaned)
    state.disjunctive_prohibitions = new_disj
    return state


# ---------------------------------------------------------------------------
# Joint confirmation
# ---------------------------------------------------------------------------

def joint_confirm_norms(state: NormLearnerState) -> NormLearnerState:
    """
    Confirm prohibitions and obligations jointly, treating them as competing
    explanations for observed suboptimality (see Algorithm 1 in the paper).
    """
    f_new: set[StateActionPair] = set()
    o_new: set[StateActionPair] = set()
    new_disj_prohibitions: list[set[StateActionPair]] = []

    hyp_o = state.hyp_obligations or set()

    for traj_id, _ in enumerate(state.demonstrations):
        trees_for_tau = [
            h for h in state.hyp_prohibitions
            if h.source_trajectory_id == traj_id
        ]

        for h in trees_for_tau:
            leaves = h.leaf_nodes()
            all_singleton = all(
                leaf.sequence is not None and len(leaf.sequence) == 1
                for leaf in leaves
            )

            if len(leaves) == 1 and all_singleton:
                candidate = leaves[0].sequence[0]
                seg_set = set(h.observed_segment)
                shortcut_set = set(h.original_shortcut)
                o_alt = (seg_set - shortcut_set) & hyp_o
                if len(o_alt) == 0:
                    f_new.add(candidate)

            elif len(leaves) > 1 and all_singleton:
                disj = {
                    leaf.sequence[0]
                    for leaf in leaves
                    if leaf.sequence is not None
                }
                if disj:
                    new_disj_prohibitions.append(disj)

    state.prohibitions.update(f_new)
    state.obligations.update(o_new)
    state.disjunctive_prohibitions.extend(new_disj_prohibitions)
    return state


# ---------------------------------------------------------------------------
# Main incremental norm discovery
# ---------------------------------------------------------------------------

def norm_discovery(
    tau: Trajectory,
    env: EnvironmentInterface,
    state: NormLearnerState,
) -> NormLearnerState:
    """
    Process one new trajectory *tau* and update *state* in-place (returned).

    Steps
    -----
    1. Update hyp_obligations by intersecting with tau.
    2. Add all pairs in tau to permissions; refine existing hyp trees.
    3. For every (i, j) pair, detect shortcuts and create new hyp trees.
    4. Joint-confirm norms; basic sanitization.
    """
    tau_set = set(tau)
    trajectory_id = len(state.demonstrations)

    # Phase 1: obligation hypothesis update
    if state.hyp_obligations is None:
        state.hyp_obligations = set(tau_set)
    else:
        state.hyp_obligations &= tau_set

    state.demonstrations.append(list(tau))

    # Phase 2: process each observed pair
    for i in range(len(tau)):
        x_i = tau[i]
        state.permissions.add(x_i)

        new_hyp_f: list[HypTree] = []
        for h in state.hyp_prohibitions:
            h_prime = traverse_and_split(h, x_i)
            if h_prime is not None:
                new_hyp_f.append(h_prime)
        state.hyp_prohibitions = new_hyp_f

        # Phase 3: shortcut detection
        for j in range(i + 1, len(tau)):
            s_i = tau[i][0]
            s_j = tau[j][0]

            if s_i == s_j:
                continue  # revisited state — trivial zero-cost "shortcut", skip

            shortest_cost = env.shortest_path_cost(s_i, s_j)
            segment_cost = env.trajectory_cost(tau[i:j])
            if shortest_cost >= segment_cost:
                continue  # no shortcut possible

            observed_segment = tau[i:j]
            plans = env.plan(s_i, s_j)

            for tau_short in plans:
                if tau_short and env.trajectory_cost(tau_short) < segment_cost:
                    h_short = make_hypothesis_tree(
                        shortcut=tau_short,
                        observed_segment=observed_segment,
                        source_trajectory_id=trajectory_id,
                    )

                    for x_k in state.permissions:
                        if h_short is None:
                            break
                        if x_k in tau_short:
                            h_short = traverse_and_split(h_short, x_k)

                    if h_short is not None:
                        subsumed = any(
                            is_subsumed_by(h_short, existing)
                            for existing in state.hyp_prohibitions
                        )
                        if not subsumed:
                            state.hyp_prohibitions.append(h_short)

    # Phase 4: confirm + sanitize
    state = joint_confirm_norms(state)
    state = sanitize_norm_state(state)
    return state
