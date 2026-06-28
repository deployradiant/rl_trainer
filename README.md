# RL Trainer

This is an experiment repository with the goal of demonstrating end-to-end RL training of an LLM for a specific task.

The task in question is understanding a set of images from a football match and predicting or advising players next moves.

The moves are then overlayed onto the result of the game.

This starts with one player.

See [docs/training_plan.md](docs/training_plan.md) for the current working plan
and minimal initial training goal.

Initial real-data overfit target: SoccerNet Tracking sequence `SNMOT-116`.
Use `modal_train.py --smoke` for the fast synthetic Modal smoke test before
running against the real SoccerNet split.
