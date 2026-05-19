"""
MazeNamo norm-discovery demo.

The environment has ONE social norm baked in:
    "Push the required box (B) before reaching the goal (G)."

Norm-following agents always detour to push the box; the norm-unaware
planner finds the shorter direct path — this discrepancy is what the
norm learner exploits.

Usage
-----
    cd /home/hrilab/vlm_norms
    python mazenamo/run_discovery.py

    # With real Qwen API:
    python mazenamo/run_discovery.py --vlm api \
        --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api-key $DASHSCOPE_API_KEY --model qwen-plus

    # With local Qwen weights (requires downloaded model):
    python mazenamo/run_discovery.py --vlm local \
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from norm_learner import (
    LocalTransformersVLM,
    MockVLM,
    NormLearner,
    OpenAICompatibleVLM,
)

from mazenamo import (
    DEFAULT_LAYOUT,
    TRAINING_STARTS,
    GridConfig,
    MazeNamoEnvironment,
    generate_training_set,
    norm_following_trajectory,
    norm_violating_trajectory,
)
from mazenamo.env import render, SimState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_grid(cfg: GridConfig) -> None:
    sim = SimState(agent=cfg.default_agent_start, box_pos=cfg.required_box_start)
    print(render(cfg, sim))


def print_trajectory_on_grid(cfg: GridConfig, tau, env: MazeNamoEnvironment) -> None:
    """Show the states visited by a trajectory as '*' on the grid."""
    visited = {env.state_to_xy(s) for s, _ in tau}
    sim = SimState(agent=cfg.default_agent_start, box_pos=cfg.required_box_start)
    rows = []
    for y in range(cfg.height):
        row = []
        for x in range(cfg.width):
            if (x, y) in cfg.walls:
                row.append("#")
            elif (x, y) == cfg.goal:
                row.append("G")
            elif (x, y) == cfg.required_box_start:
                row.append("B")
            elif (x, y) in visited:
                row.append("*")
            else:
                row.append(".")
        rows.append("".join(row))
    print("\n".join(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MazeNamo norm-discovery demo")
    parser.add_argument("--vlm", choices=["mock", "local", "api"], default="local")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path for --vlm local; "
                             "model name for --vlm api")
    parser.add_argument("--device", default="auto",
                        help="Device for local inference: auto, cuda, cuda:0, cpu")
    parser.add_argument("--dtype", default="auto",
                        help="Torch dtype: auto, bfloat16, float16, float32")
    parser.add_argument("--api-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--interval", type=int, default=5,
                        help="Query VLM every N trajectories")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Repeat each start position N times")
    args = parser.parse_args()

    # --- Build environment -------------------------------------------------
    cfg = GridConfig.from_layout(DEFAULT_LAYOUT)
    env = MazeNamoEnvironment(cfg)

    print("=" * 60)
    print("MazeNamo Norm Discovery")
    print("=" * 60)
    print(f"\nGrid layout (8×8):")
    print_grid(cfg)
    print(f"\nGoal        : {cfg.goal}")
    print(f"Required box: {cfg.required_box_start}")
    print(f"NORM        : agent must push B before reaching G\n")

    # --- Show example trajectories ----------------------------------------
    print("--- Norm-FOLLOWING trajectory (from default start) ---")
    tau_follow = norm_following_trajectory(env, cfg.default_agent_start)
    if tau_follow:
        print(f"Length: {len(tau_follow)} steps")
        print_trajectory_on_grid(cfg, tau_follow, env)
    print()

    print("--- Norm-VIOLATING trajectory (direct to goal) ---")
    tau_violate = norm_violating_trajectory(env, cfg.default_agent_start)
    if tau_violate:
        print(f"Length: {len(tau_violate)} steps  (shortcut saves "
              f"{len(tau_follow) - len(tau_violate)} steps)")
        print_trajectory_on_grid(cfg, tau_violate, env)
    print()

    # --- VLM backend -------------------------------------------------------
    if args.vlm == "mock":
        import json
        # Realistic mock that demonstrates what a real Qwen response would look like
        vlm = MockVLM(fixed_response=json.dumps([
            {
                "type": "obligation",
                "description": (
                    "The agent must push the required box (B) before reaching "
                    "the goal (G). All observed agents detour to interact with "
                    "the box at (2,4) prior to moving toward the goal."
                ),
                "reasoning": (
                    "Agents consistently take longer paths that pass through "
                    "the box location, even though a shorter direct route to "
                    "the goal exists. The box-push action appears in every "
                    "observed trajectory."
                ),
            },
            {
                "type": "prohibition",
                "description": (
                    "Agents must not move directly north from the start area "
                    "toward the goal without first visiting the box."
                ),
                "reasoning": (
                    "The direct northward path from the lower grid to the goal "
                    "is consistently avoided; instead, agents go west then "
                    "interact with the box before heading to the goal."
                ),
            },
        ]))
        print("VLM backend : MockVLM (realistic stub responses)\n")
    elif args.vlm == "local":
        vlm = LocalTransformersVLM(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
        )
        print(f"VLM backend : local {args.model}  (device={args.device}, dtype={args.dtype})\n")
    else:
        vlm = OpenAICompatibleVLM(
            base_url=args.api_url, api_key="sk-b3d70dac46d54b5b82461eff1bb27256", model=args.model
        )
        print(f"VLM backend : API at {args.api_url}, model={args.model}\n")

    # --- Set up learner ----------------------------------------------------
    def on_vlm(norms):
        print(f"\n  ── VLM grounded {len(norms)} norm(s) ──")
        for n in norms:
            print(f"  [{n.norm_type.upper()}] {n.description}")
            print(f"   reasoning: {n.reasoning}")

    learner = NormLearner(
        env, vlm,
        vlm_query_interval=args.interval,
        on_vlm_result=on_vlm,
    )

    # --- Generate & process trajectories ----------------------------------
    trajectories = generate_training_set(env, TRAINING_STARTS, repeat=args.repeat)
    print(f"Generated {len(trajectories)} norm-following trajectories "
          f"({len(TRAINING_STARTS)} starts × {args.repeat} repeats).\n")

    for idx, tau in enumerate(trajectories):
        learner.process_trajectory(tau)
        sym = learner.get_symbolic_state()
        if (idx + 1) % 5 == 0 or idx == len(trajectories) - 1:
            print(
                f"[{idx+1:3d}] prohibitions={len(sym.prohibitions):2d}  "
                f"obligations={len(sym.obligations):2d}  "
                f"hyp_prohibitions={len(sym.hyp_prohibitions):2d}  "
                f"permissions={len(sym.permissions):3d}  "
                f"vlm_queries={learner.vlm_query_count}"
            )

    # Final VLM query if not already triggered
    if learner.n_trajectories % args.interval != 0:
        print("\nRunning final VLM query…")
        learner.force_vlm_query()

    # --- Print symbolic results -------------------------------------------
    sym = learner.get_symbolic_state()
    print("\n" + "=" * 60)
    print("SYMBOLIC RESULTS")
    print("=" * 60)

    print(f"\nConfirmed prohibitions ({len(sym.prohibitions)}):")
    for sa in sorted(sym.prohibitions, key=str):
        x, y = env.state_to_xy(sa[0])
        print(f"  {env.describe_action(sa[1])} from ({x},{y})  "
              f"[state {sa[0]}]")

    print(f"\nConfirmed obligations ({len(sym.obligations)}):")
    for sa in sorted(sym.obligations, key=str):
        x, y = env.state_to_xy(sa[0])
        print(f"  {env.describe_action(sa[1])} from ({x},{y})  "
              f"[state {sa[0]}]")

    print(f"\nDisjunctive prohibitions ({len(sym.disjunctive_prohibitions)}):")
    for disj in sym.disjunctive_prohibitions:
        parts = []
        for sa in sorted(disj):
            x, y = env.state_to_xy(sa[0])
            parts.append(f"{env.describe_action(sa[1])} from ({x},{y})")
        print(f"  one-of: {' | '.join(parts)}")

    h_o = sym.hyp_obligations or set()
    print(f"\nHypothesised obligations (pairs in EVERY trajectory): {len(h_o)}")
    for sa in sorted(h_o, key=str)[:15]:
        x, y = env.state_to_xy(sa[0])
        print(f"  {env.describe_action(sa[1])} from ({x},{y})  [state {sa[0]}]")
    if len(h_o) > 15:
        print(f"  … and {len(h_o) - 15} more")

    print(f"\nHypothesised prohibitions (remaining trees): "
          f"{len(sym.hyp_prohibitions)}")
    for h in sym.hyp_prohibitions[:10]:
        leaves = h.leaf_nodes()
        leaf_desc = []
        for lf in leaves:
            if lf.sequence:
                for sa in lf.sequence[:3]:
                    x, y = env.state_to_xy(sa[0])
                    leaf_desc.append(f"{env.describe_action(sa[1])}({x},{y})")
        bx, by = cfg.required_box_start
        traj_desc = f"start={env.state_to_xy(sym.demonstrations[h.source_trajectory_id][0][0])}" \
                    if h.source_trajectory_id is not None else ""
        print(f"  tree [{traj_desc}]: leaves={len(leaves)} candidates=[{', '.join(leaf_desc[:5])}]")

    # --- Print VLM-grounded norms -----------------------------------------
    grounded = learner.get_grounded_norms()
    print(f"\n{'=' * 60}")
    print(f"VLM-GROUNDED NORMS ({len(grounded)})")
    print("=" * 60)
    if not grounded:
        print("  (No grounded norms — use --vlm api or --vlm local for real output)")
    for gn in grounded:
        print(f"\n  [{gn.norm_type.upper()}]  {gn.description}")
        print(f"  reasoning  : {gn.reasoning}")
        print(f"  vlm round  : {gn.iteration}")

    # --- Visualise which cells are prohibited / hypothesised  -------------
    print(f"\n{'=' * 60}")
    print("GRID ANNOTATION (P=prohibited, H=hypothesised, G=goal, B=box)")
    print("=" * 60)

    prohibited_cells = {env.state_to_xy(sa[0]) for sa in sym.prohibitions}
    hyp_cells: set[tuple[int, int]] = set()
    for h in sym.hyp_prohibitions:
        for leaf in h.leaf_nodes():
            if leaf.sequence:
                for sa in leaf.sequence:
                    hyp_cells.add(env.state_to_xy(sa[0]))

    for y in range(cfg.height):
        row = []
        for x in range(cfg.width):
            if (x, y) in cfg.walls:
                row.append("#")
            elif (x, y) == cfg.goal:
                row.append("G")
            elif (x, y) == cfg.required_box_start:
                row.append("B")
            elif (x, y) in prohibited_cells:
                row.append("P")
            elif (x, y) in hyp_cells:
                row.append("h")
            else:
                row.append(".")
        print("".join(row))
    print()
    print("P = confirmed prohibited cell  h = hypothesised  . = free")


if __name__ == "__main__":
    main()
