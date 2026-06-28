# RL Trainer

This is an experiment repository with the goal of demonstrating end-to-end RL training of an LLM for a specific task.

The task in question is understanding a set of images from a football match and predicting or advising players next moves.

The moves are then overlayed onto the result of the game.

This starts with one player.

See [docs/training_plan.md](docs/training_plan.md) for the current working plan
and minimal initial training goal.

Initial real-data run: deterministic SoccerNet `train` split sequence subsets
with seed `7`, using `10%` of sequences for training and `2%` for held-out eval.
The run writes `sequence_manifest.json`, `training_history.json`,
`trajectory_head.pt`, and `trajectory_preview.png`.

Run the first 10%/2% Modal job with:

```bash
uvx modal run modal_train.py --split train --train-fraction 0.10 --eval-fraction 0.02 --seed 7 --epochs 1
```

Artifacts are copied to the `soccernet-tracking` Modal volume, including
`trajectory_preview.png` for a visual predicted-vs-actual overlay.
