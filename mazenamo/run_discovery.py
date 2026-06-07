"""
MazeNamo norm-discovery demo.

Available scenarios
-------------------
  push_before_goal   — agent must push box B before reaching goal G (temporal obligation)
  avoid_spills       — agent must not step on spill cells (prohibition)
  greet_before_door  — agent must GREET guard before crossing checkpoint (temporal obligation)
  no_restricted_zone — agent must not enter the restricted zone (prohibition)

Usage
-----
    cd /home/hrilab/vlm_norms
    python mazenamo/run_discovery.py --scenario avoid_spills

    # With real Qwen API:
    python mazenamo/run_discovery.py --scenario greet_before_door \\
        --vlm api \\
        --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
        --api-key $DASHSCOPE_API_KEY --model qwen-plus

    # With local Qwen weights:
    python mazenamo/run_discovery.py --scenario push_before_goal \\
        --vlm local --model Qwen/Qwen2.5-7B-Instruct
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

from mazenamo import SCENARIOS, GridConfig, MazeNamoEnvironment
from mazenamo.env import render, SimState


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_grid(cfg: GridConfig) -> None:
    box_pos = cfg.required_box_start if cfg.required_box_start else cfg.goal
    sim = SimState(agent=cfg.default_agent_start, box_pos=box_pos)
    print(render(cfg, sim))


def print_trajectory_on_grid(cfg: GridConfig, tau, env: MazeNamoEnvironment) -> None:
    """Show the states visited by a trajectory as '*' on the grid."""
    visited = {env.state_to_xy(s) for s, _ in tau}
    rows = []
    for y in range(cfg.height):
        row = []
        for x in range(cfg.width):
            if (x, y) in cfg.walls:
                row.append("#")
            elif (x, y) == cfg.goal:
                row.append("G")
            elif cfg.required_box_start and (x, y) == cfg.required_box_start:
                row.append("B")
            elif cfg.guard_pos and (x, y) == cfg.guard_pos:
                row.append("g")
            elif (x, y) in cfg.forbidden_cells:
                lbl = cfg.cell_labels.get((x, y), "?")
                row.append("s" if lbl == "spill" else "R")
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
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="push_before_goal",
        help="Which norm scenario to run (default: push_before_goal)",
    )
    parser.add_argument("--vlm", choices=["mock", "local", "api"], default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path for --vlm local; "
                             "model name for --vlm api")
    parser.add_argument("--device", default="auto",
                        help="Device for local inference: auto, cuda, cuda:0, cpu")
    parser.add_argument("--dtype", default="auto",
                        help="Torch dtype: auto, bfloat16, float16, float32")
    parser.add_argument("--api-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--interval", type=int, default=1,
                        help="Query LLM every N trajectories (default: every trajectory)")
    args = parser.parse_args()

    # --- Load scenario --------------------------------------------------------
    scenario = SCENARIOS[args.scenario]
    cfg = GridConfig.from_layout(scenario.layout)
    env = MazeNamoEnvironment(cfg)

    print("=" * 60)
    print(f"MazeNamo Norm Discovery — scenario: {args.scenario}")
    print("=" * 60)
    print(f"\nGround-truth norm: {scenario.norm_description}\n")
    print(f"Grid layout ({cfg.width}×{cfg.height}):")
    print_grid(cfg)
    print(f"\nGoal: {cfg.goal}")
    if cfg.required_box_start:
        print(f"Required box: {cfg.required_box_start}")
    if cfg.forbidden_cells:
        labels = {v for v in cfg.cell_labels.values() if v != "guard"}
        print(f"Forbidden cells ({', '.join(sorted(labels))}): "
              + ", ".join(str(c) for c in sorted(cfg.forbidden_cells)))
    if cfg.guard_pos:
        print(f"Guard: {cfg.guard_pos}")
    print()

    # --- LLM backend ----------------------------------------------------------
    if args.vlm == "mock":
        llm = MockLLM(fixed_response=scenario.mock_llm_response)
        print("LLM backend : MockLLM (scenario-specific stub)\n")
    elif args.vlm == "local":
        llm = LocalTransformersLLM(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
        )
        print(f"LLM backend : local {args.model}  (device={args.device}, dtype={args.dtype})\n")
    else:
        llm = OpenAICompatibleLLM(
            base_url=args.api_url, api_key=args.api_key, model=args.model
        )
        print(f"LLM backend : API at {args.api_url}, model={args.model}\n")

    # --- Set up learner -------------------------------------------------------
    learner = NormLearner(env, llm, llm_query_interval=args.interval)

    # --- Generate and process trajectories ------------------------------------
    trajectories = scenario.generate(env)
    print(f"Generated {len(trajectories)} norm-following trajectories.\n")

    if not trajectories:
        print("ERROR: no trajectories generated — check scenario layout.")
        sys.exit(1)

    for idx, tau in enumerate(trajectories):
        queries_before = learner.llm_query_count
        learner.process_trajectory(tau)

        start_xy = env.state_to_xy(tau[0][0])
        print(f"\n{'=' * 60}")
        print(f"TRAJECTORY {idx + 1}  start={start_xy}  len={len(tau)}")
        print(f"{'=' * 60}")
        print_trajectory_on_grid(cfg, tau, env)

        if learner.llm_query_count > queries_before and learner.last_prompt is not None:
            print("\n--- LLM PROMPT ---")
            print(learner.last_prompt)

            norms = learner.get_grounded_norms()
            rejected = learner.get_rejected_norms()
            print(f"\n--- LLM OUTPUT  ({len(norms)} active norm(s)) ---")
            for n in norms:
                print(f"  [{n.modality.upper()}] {n.description}")
                print(f"   action: {n.action}  norm_type: {n.norm_type}")
                print(f"   reasoning: {n.reasoning}")
            if rejected:
                print(f"\n--- REJECTED ({len(rejected)} total) ---")
                for n in rejected[-5:]:
                    print(f"  [{n.modality.upper()}] {n.description}")

    # --- Symbolic results -----------------------------------------------------
    sym = learner.get_symbolic_state()
    print("\n" + "=" * 60)
    print("SYMBOLIC RESULTS")
    print("=" * 60)

    print(f"\nConfirmed prohibitions ({len(sym.prohibitions)}):")
    for sa in sorted(sym.prohibitions, key=str):
        x, y = env.state_to_xy(sa[0])
        print(f"  {env.describe_action(sa[1])} from ({x},{y})  [state {sa[0]}]")

    print(f"\nConfirmed obligations ({len(sym.obligations)}):")
    for sa in sorted(sym.obligations, key=str):
        x, y = env.state_to_xy(sa[0])
        print(f"  {env.describe_action(sa[1])} from ({x},{y})  [state {sa[0]}]")

    print(f"\nDisjunctive prohibitions ({len(sym.disjunctive_prohibitions)}):")
    for disj in sym.disjunctive_prohibitions:
        parts = []
        for sa in sorted(disj):
            x, y = env.state_to_xy(sa[0])
            parts.append(f"{env.describe_action(sa[1])} from ({x},{y})")
        print(f"  one-of: {' | '.join(parts)}")

    h_o = sym.hyp_obligations or set()
    non_mv = {sa for sa in h_o if env.describe_action(sa[1]) not in {"NORTH","SOUTH","EAST","WEST"}}
    print(f"\nHypothesised obligations — non-movement (in every trajectory): {len(non_mv)}")
    for sa in sorted(non_mv, key=str):
        x, y = env.state_to_xy(sa[0])
        print(f"  {env.describe_action(sa[1])} from ({x},{y})  [state {sa[0]}]")

    print(f"\nHypothesised prohibitions (remaining trees): {len(sym.hyp_prohibitions)}")
    for h in sym.hyp_prohibitions[:10]:
        leaves = h.leaf_nodes()
        leaf_desc = []
        for lf in leaves:
            if lf.sequence:
                for sa in lf.sequence[:3]:
                    x, y = env.state_to_xy(sa[0])
                    leaf_desc.append(f"{env.describe_action(sa[1])}({x},{y})")
        traj_desc = (
            f"start={env.state_to_xy(sym.demonstrations[h.source_trajectory_id][0][0])}"
            if h.source_trajectory_id is not None else ""
        )
        print(f"  tree [{traj_desc}]: leaves={len(leaves)} candidates=[{', '.join(leaf_desc[:5])}]")

    # --- LLM-grounded norms ---------------------------------------------------
    grounded = learner.get_grounded_norms()
    print(f"\n{'=' * 60}")
    print(f"LLM-GROUNDED NORMS ({len(grounded)})")
    print("=" * 60)
    if not grounded:
        print("  (No grounded norms — use --vlm api or --vlm local for real output)")
    for gn in grounded:
        ctx = ", ".join(gn.context) if gn.context else "(none)"
        print(f"\n  [{gn.modality.upper()}]  {gn.description}")
        print(f"  action     : {gn.action}")
        print(f"  context    : {ctx}")
        print(f"  norm_type  : {gn.norm_type}")
        print(f"  reasoning  : {gn.reasoning}")
        print(f"  llm round  : {gn.iteration}")

    # --- Grid annotation ------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("GRID ANNOTATION  (P=prohibited, h=hypothesised, G=goal, s=spill, R=restricted, g=guard)")
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
            elif cfg.required_box_start and (x, y) == cfg.required_box_start:
                row.append("B")
            elif cfg.guard_pos and (x, y) == cfg.guard_pos:
                row.append("g")
            elif (x, y) in cfg.forbidden_cells:
                lbl = cfg.cell_labels.get((x, y), "?")
                row.append("s" if lbl == "spill" else "R")
            elif (x, y) in prohibited_cells:
                row.append("P")
            elif (x, y) in hyp_cells:
                row.append("h")
            else:
                row.append(".")
        print("".join(row))
    print()
    print("P=confirmed prohibition  h=hypothesised  .=free")


if __name__ == "__main__":
    main()
