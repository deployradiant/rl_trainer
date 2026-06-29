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
`trajectory_head.pt`, `trajectory_preview.png`, and `trajectory_rollouts.json`.

Run the first 10%/2% Modal job with:

```bash
uvx modal run modal_train.py --split train --train-fraction 0.10 --eval-fraction 0.02 --seed 7 --epochs 1
```

Artifacts are copied to the `soccernet-tracking` Modal volume, including
`trajectory_rollouts.json` for selectable frame-by-frame playback and
`trajectory_preview.png` for a visual predicted-vs-actual overlay.

Serve the visualizer locally through Modal with:

```bash
uvx modal serve modal_visualize.py
```

Deploy the hosted visualizer with:

```bash
uvx modal deploy modal_visualize.py
```

Run the current one-sample learned-value MCTS overfit gate with:

```bash
uvx modal run modal_train.py --split train --train-sequence-count 1 --eval-sequence-count 1 --input-frames 5 --horizon 50 --action-stride 5 --stride 25 --max-frames 250 --overfit-samples 1 --sample-selection max_motion --epochs 200 --batch-size 1 --lr 0.003 --cache-visual-features --feature-cache-batch-size 1 --policy-loss-weight 1.0 --value-loss-weight 1.0 --value-rank-loss-weight 1.0 --value-fde-weight 0.5 --value-target-scale 10.0 --mcts-simulations 192 --mcts-top-k 10 --mcts-c-puct 1.5 --seed 7
```
