"""
ProperShopper norm-discovery demo.

Loads two recorded trajectories (cheese trip and garlic trip) from
new_trajectories.json, abstracts them into the symbolic state space, and
runs the incremental norm-discovery algorithm.  The target norm is:

    [OBLIGATORY] pay(agent, checkout) — after picking up items, an agent
    is obligated to pay at the checkout before leaving the store.

This norm is detected via hyp_obligations: the INTERACT action at the checkout
zone (with has_basket=True, has_items=True) appears in every trajectory, making
it a candidate obligatory action.  The LLM grounding step then articulates this
as a human-readable norm.

Usage
-----
    cd /home/hrilab/vlm_norms
    python propershopper/run_discovery.py

    # With a real LLM (OpenAI-compatible endpoint):
    python propershopper/run_discovery.py \\
        --llm api \\
        --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
        --api-key $DASHSCOPE_API_KEY \\
        --model qwen-plus

    # With local Qwen weights:
    python propershopper/run_discovery.py \\
        --llm local \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from norm_learner import (
    LocalTransformersLLM,
    MockLLM,
    NormLearner,
    OpenAICompatibleLLM,
)
from norm_learner.environments.propershopper import ProperShopperEnvironment

# ---------------------------------------------------------------------------
# Mock LLM response — what a real LLM would ideally produce
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = json.dumps([
    {
        "context": ["in_supermarket", "has_basket", "has_items_in_basket"],
        "modality": "obligatory",
        "action": "pay(agent, checkout)",
        "norm_type": "convenience",
    }
])

# Actions the algorithm treats as navigation (filtered from obligation display)
_MOVEMENT_ACTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summarise_raw_trajectory(raw_traj: list[dict]) -> None:
    """Print key semantic events from a raw JSON trajectory."""
    actions = [s["action"] for s in raw_traj]
    interact_steps = [i for i, s in enumerate(raw_traj) if s["action"] == "INTERACT"]

    print(f"  Length     : {len(raw_traj)} steps")
    print(f"  Start pos  : ({raw_traj[0]['state']['x']:.2f}, {raw_traj[0]['state']['y']:.2f})")
    print(f"  End pos    : ({raw_traj[-1]['state']['x']:.2f}, {raw_traj[-1]['state']['y']:.2f})")
    print(f"  End money  : {raw_traj[-1]['state']['money']}")

    for i, step in enumerate(raw_traj):
        s = step["state"]
        if i > 0:
            prev = raw_traj[i - 1]["state"]
            if s["has_basket"] != prev["has_basket"]:
                print(f"  Step {i:3d}   : got basket @ ({s['x']:.2f}, {s['y']:.2f})")
            elif s["money"] != prev["money"]:
                print(
                    f"  Step {i:3d}   : PAID {prev['money'] - s['money']:.2f} "
                    f"@ ({s['x']:.2f}, {s['y']:.2f}), items cleared"
                )
            elif set(s["items_in_basket"]) != set(prev["items_in_basket"]):
                added = set(s["items_in_basket"]) - set(prev["items_in_basket"])
                if added:
                    print(f"  Step {i:3d}   : picked up {added} @ ({s['x']:.2f}, {s['y']:.2f})")


def print_hyp_obligations(learner: NormLearner, env: ProperShopperEnvironment) -> None:
    """Print the non-movement pairs in hyp_obligations."""
    sym = learner.get_symbolic_state()
    h_o = sym.hyp_obligations or set()
    non_mv = {sa for sa in h_o if sa[1] not in _MOVEMENT_ACTIONS}

    print(f"\nHypothesised obligations (non-movement, in every trajectory): {len(non_mv)}")
    for sa in sorted(non_mv, key=str):
        print(f"  {env.describe_action(sa[1])} from {env.describe_state(sa[0])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ProperShopper norm-discovery demo")
    parser.add_argument("--llm", choices=["mock", "local", "api"], default="mock",
                        help="LLM backend (default: mock)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model ID for --llm local; model name for --llm api")
    parser.add_argument("--device", default="auto",
                        help="Device for local inference: auto, cuda, cpu")
    parser.add_argument("--dtype", default="auto",
                        help="Torch dtype: auto, bfloat16, float16, float32")
    parser.add_argument("--api-url", default="http://localhost:11434/v1",
                        help="Base URL for --llm api")
    parser.add_argument("--api-key", default="ollama",
                        help="API key for --llm api")
    parser.add_argument(
        "--trajectories",
        default=str(ROOT / "propershopper" / "new_trajectories.json"),
        help="Path to the ProperShopper trajectories JSON file",
    )
    parser.add_argument("--interval", type=int, default=1,
                        help="Query LLM every N trajectories (default: 1)")
    args = parser.parse_args()

    # --- Load raw trajectories ------------------------------------------------
    traj_path = Path(args.trajectories)
    if not traj_path.exists():
        print(f"ERROR: trajectory file not found: {traj_path}", file=sys.stderr)
        sys.exit(1)

    with open(traj_path) as f:
        raw_trajectories = json.load(f)

    print("=" * 64)
    print("ProperShopper Norm Discovery")
    print("=" * 64)
    print(f"\nTarget norm: obligatory to pay at checkout after picking up items\n")
    print(f"Loaded {len(raw_trajectories)} trajectories from {traj_path.name}\n")

    for i, raw_traj in enumerate(raw_trajectories):
        items = set()
        for step in raw_traj:
            items.update(step["state"]["items_in_basket"])
        print(f"Trajectory {i + 1}: {', '.join(sorted(items)) or '(no items)'}")
        summarise_raw_trajectory(raw_traj)
        print()

    # --- Build environment and abstract trajectories --------------------------
    env = ProperShopperEnvironment()
    trajectories = env.load_trajectories(raw_trajectories)

    print(f"Abstract state: (coarse_gx, coarse_gy, has_basket, has_items)")
    print(f"Cell size     : {env.cell_size} sim-units = {env.STEPS_PER_CELL} raw steps\n")

    # --- LLM backend ----------------------------------------------------------
    if args.llm == "mock":
        llm = MockLLM(fixed_response=_MOCK_RESPONSE)
        print("LLM backend: MockLLM (returns stub payment-norm response)")
    elif args.llm == "local":
        llm = LocalTransformersLLM(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
        )
        print(f"LLM backend: local {args.model}  (device={args.device})")
    else:
        llm = OpenAICompatibleLLM(
            base_url=args.api_url, api_key=args.api_key, model=args.model
        )
        print(f"LLM backend: API at {args.api_url}, model={args.model}")
    print()

    # --- Norm learner ---------------------------------------------------------
    learner = NormLearner(env, llm, llm_query_interval=args.interval)

    for idx, tau in enumerate(trajectories):
        print("=" * 64)
        print(f"Processing trajectory {idx + 1} / {len(trajectories)}  "
              f"({len(tau)} abstract steps)")
        print("=" * 64)

        queries_before = learner.llm_query_count
        learner.process_trajectory(tau)

        # Show the symbolic state after this trajectory
        sym = learner.get_symbolic_state()
        h_o = sym.hyp_obligations or set()
        non_mv = {sa for sa in h_o if sa[1] not in _MOVEMENT_ACTIONS}
        print(f"\nAfter trajectory {idx + 1}:")
        print(f"  hyp_obligations (non-movement): {len(non_mv)} pairs")
        print(f"  hyp_prohibitions (trees)      : {len(sym.hyp_prohibitions)}")
        print(f"  confirmed prohibitions         : {len(sym.prohibitions)}")

        if non_mv:
            print(f"\n  Key obligation candidates:")
            for sa in sorted(non_mv, key=str):
                print(f"    INTERACT from {env.describe_state(sa[0])}")

        # Show LLM query results if one was triggered
        if learner.llm_query_count > queries_before:
            print(f"\n--- LLM PROMPT (trajectory {idx + 1}) ---")
            print(learner.last_prompt)
            norms = learner.get_grounded_norms()
            print(f"\n--- LLM OUTPUT ({len(norms)} grounded norm(s)) ---")
            for n in norms:
                print(f"   action    : {n.action}")
                print(f"   context   : {', '.join(n.context)}")
                print(f"   norm_type : {n.norm_type}")

    # --- Final symbolic summary -----------------------------------------------
    sym = learner.get_symbolic_state()
    print("\n" + "=" * 64)
    print("FINAL SYMBOLIC RESULTS")
    print("=" * 64)

    h_o = sym.hyp_obligations or set()
    non_mv_final = {sa for sa in h_o if sa[1] not in _MOVEMENT_ACTIONS}

    print(f"\nHypothesised obligations — non-movement actions in ALL trajectories")
    print(f"({len(non_mv_final)} pairs):")
    if non_mv_final:
        for sa in sorted(non_mv_final, key=str):
            print(f"  INTERACT from {env.describe_state(sa[0])}")
    else:
        print("  (none)")

    print(f"\nConfirmed prohibitions : {len(sym.prohibitions)}")
    print(f"Confirmed obligations  : {len(sym.obligations)}")
    print(f"Disjunctive prohibitions: {len(sym.disjunctive_prohibitions)}")

    # --- Final LLM-grounded norms ---------------------------------------------
    grounded = learner.get_grounded_norms()
    print(f"\n{'=' * 64}")
    print(f"LLM-GROUNDED NORMS ({len(grounded)})")
    print("=" * 64)
    if not grounded:
        print("  (No grounded norms — LLM returned empty or unparseable response)")
        print("  Re-run with --llm api or --llm local for real LLM output.")
    for gn in grounded:
        ctx = ", ".join(gn.context) if gn.context else "(none)"
        print(f"\n  [{gn.modality.upper()}]  {gn.description}")
        print(f"  action     : {gn.action}")
        print(f"  context    : {ctx}")
        print(f"  norm_type  : {gn.norm_type}")
        print(f"  llm round  : {gn.iteration}")


if __name__ == "__main__":
    main()
