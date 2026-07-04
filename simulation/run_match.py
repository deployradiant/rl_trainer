"""CLI for running World Cup match simulations from a saved state snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.match_state import load_match_state, save_match_state
from simulation.monte_carlo import OutcomeSimulationConfig, simulate_outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-file", required=True, help="Path to match_state.json")
    parser.add_argument("--out", required=True, help="Output artifact directory")
    parser.add_argument("--rollouts", type=int, default=10000)
    parser.add_argument("--horizon-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    state = load_match_state(args.snapshot_file)
    config = OutcomeSimulationConfig(
        rollouts=args.rollouts,
        seed=args.seed,
        horizon_minutes=args.horizon_minutes,
    )
    summary, samples = simulate_outcomes(state, config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_match_state(state, out_dir / "match_state.json")
    _write_json(out_dir / "simulation_summary.json", summary)
    _write_json(out_dir / "rollout_samples.json", {"rollouts": samples})
    _write_json(
        out_dir / "search_diagnostics.json",
        {"status": "not_available", "reason": "Level 0 outcome simulation has no MCTS"},
    )
    (out_dir / "prediction_card.md").write_text(_prediction_card(summary))
    print(f"Wrote simulation artifacts to {out_dir}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _prediction_card(summary: dict) -> str:
    metadata = summary["metadata"]
    teams = summary["teams"]
    probs = summary["result_probabilities"]
    return "\n".join(
        [
            "# Match Simulation Prediction",
            "",
            f"- Competition: {metadata['competition']}",
            f"- Match ID: {metadata['match_id']}",
            f"- Source: {metadata['source']}",
            f"- Snapshot epoch UTC: {metadata['snapshot_epoch_utc']}",
            f"- Seed: {summary['config']['seed']}",
            f"- Rollouts: {summary['rollouts']}",
            f"- State: {teams['home']['name']} {teams['home']['goals']} - "
            f"{teams['away']['goals']} {teams['away']['name']}",
            "",
            "## Result Probabilities",
            "",
            f"- Home win: {probs['home_win']:.3f}",
            f"- Draw: {probs['draw']:.3f}",
            f"- Away win: {probs['away_win']:.3f}",
            "",
            "This Level 0 simulator uses score, clock, team strength, and red cards. "
            "It does not yet include event streams, tracking, player movement, or MCTS.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
