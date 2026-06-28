# Training Plan

This project is an experiment in post-training an open-source vision-language
model for football understanding. The long-term demo is a model that sees a few
frames of a match, predicts useful player movement, explains the game state, and
renders the prediction back over the video.

The minimal first goal is narrower:

> Given a short frame window and one selected player, predict that player's next
> one second of pitch-normalized movement.

This gives the project a measurable target before adding full-match reasoning,
multi-player forecasting, MCTS, or RL.

## Working Thesis

The model should be treated as a vision-language-trajectory system, not as a
chat model that emits coordinates as text.

The intended shape is:

```text
match frames + player identity + optional context
        |
Qwen vision-language backbone
        |
football latent state
        |
trajectory head + intent head + value head
```

The language model can produce explanations for the demo, but the core movement
prediction should be structured tensors:

```text
future_xy: [batch, horizon, 2]
intent_logits: [batch, num_intents]
value: [batch, 1]
```

For the first training loop, only `future_xy` is required.

## Model Choice

Use Qwen as the initial open-weight vision-language backbone.

Start with the smallest Qwen VLM variant that makes the training loop easy to
debug. The first implementation should prioritize reliable PyTorch plumbing over
model size.

Expected first setup:

```text
Qwen VLM backbone, frozen
small custom football trajectory head
plain PyTorch training loop
Modal for remote GPU execution
```

The project should avoid high-level training frameworks at first. It is fine to
use existing libraries for model loading, tokenization, image processing, and
weight formats, but the important behavior should remain explicit:

```text
own dataset
own collator
own forward wrapper
own loss
own optimizer and scheduler
own checkpoint format
own evaluation
own RL/MCTS logic later
```

Do not use `Trainer`, `GRPOTrainer`, or similar high-level training orchestration
until there is a strong reason.

## Minimal Initial Training Goal

The first milestone is selected-player trajectory prediction.

The first real-data run should use deterministic SoccerNet `train` split
sequence subsets: `10%` of sequences for training and `2%` for held-out eval,
selected with seed `7`. Persist the exact sequence lists in
`sequence_manifest.json` so future scaling runs compare against a fixed data
slice.

Input:

```text
3-8 consecutive frames from a football match
selected player identity or selected player bounding box
current player position in pitch coordinates if available
```

Output:

```text
next 1 second of selected-player xy positions
```

Coordinate convention:

```text
x, y normalized to the pitch, not the broadcast image
timestamps normalized to epoch timestamps in UTC where source data provides time
```

Primary metrics:

```text
ADE_1s: average displacement error over the future window
FDE_1s: final displacement error at one second
NLL_1s: likelihood of the true future once probabilistic outputs exist
```

The first version can use a deterministic head and Smooth L1 or MSE loss. Once
the deterministic baseline works, replace or extend it with a probabilistic head
because football futures are multi-modal.

## Data Requirements

The first dataset should include tracking or labels that can produce:

```text
frame window
selected player
current pitch coordinates
future pitch coordinates
source
symbol or match identifier
interval or frame rate
fetch/import time
```

Prefer data with existing player tracking over broadcast-only video for the
first milestone. Raw video-only learning should come later, after the trajectory
task is validated.

Potential data sources to evaluate:

```text
Metrica Sports sample data
SoccerNet tracking data
StatsBomb open data / 360 context
other tracking datasets with permissible licensing
```

## Training Stages

### Stage 0: Data And Eval Harness

Build the smallest dataset pipeline that emits:

```text
frames
selected_player_mask or box
future_xy
metadata
```

Create a fixed held-out evaluation split by match, not by random frame, so the
eval set measures generalization rather than near-duplicate memorization.

### Stage 1: Frozen Backbone Supervised Baseline

Freeze Qwen and train only the football trajectory head.

Goal:

```text
prove that the visual backbone plus a small head can learn short-horizon player
movement better than simple baselines
```

Baselines:

```text
constant position
constant velocity
simple Kalman-style extrapolation
```

The learned model must beat these before RL is worth adding.

### Stage 2: Better Trajectory Modeling

Add one or more of:

```text
larger trajectory head
selected visual token pooling
player and ball context
intent classification
probabilistic trajectory output
LoRA or adapter tuning on top Qwen layers
```

This stage should remain supervised.

### Stage 3: Value Model

Train a value head that estimates tactical usefulness:

```text
possession value
shot probability change
turnover risk
pass availability
space creation
```

This creates the reward/value surface required for search.

### Stage 4: MCTS And Offline RL

Add MCTS only after the model has a usable state, dynamics, and value estimate.

MCTS should search over short movement or intent chunks, not free-form text.

Possible search unit:

```text
candidate 1-second movement chunk
candidate intent + movement chunk
```

Possible reward:

```text
trajectory realism
match to actual future movement
increase in possession value
reduced turnover risk
chance creation
```

GRPO-style updates can be implemented directly in PyTorch by sampling multiple
candidate futures, scoring them, normalizing rewards within the group, and
updating candidate log probabilities. Keep a supervised/KL term so the model
does not exploit a weak reward model with unrealistic movement.

## Scaling Evaluation

Scaling laws should be fitted from actual small training runs.

Start with a data scaling law:

```text
metric(N) = E_inf + A * N^-alpha
```

Where:

```text
N = number of training clips
metric = held-out ADE_1s, FDE_1s, or NLL_1s
E_inf = irreducible error floor
```

Initial training sizes:

```text
1k clips
3k clips
10k clips
30k clips
100k clips
```

If data and Modal budget allow:

```text
300k clips
1M clips
```

Use nested, stratified subsets so larger runs contain the smaller runs plus more
data. Stratify by match, team, player role, phase of play, and source where
available.

Run multiple seeds for small sizes:

```text
1k, 3k, 10k: 3 seeds
30k: 2 seeds
100k+: 1-2 seeds
```

Fit the scaling curve on held-out eval metrics, not training loss.

## Demo Path

The first demo should show:

```text
input match clip
selected player highlight
predicted 1-second movement overlay
actual movement overlay
error visualization
```

The initial training loop writes `trajectory_preview.png`, which overlays one
held-out eval sample with the actual future movement in green and the predicted
future movement in red. It also writes `trajectory_rollout.json` so the
visualizer can replay the real SoccerNet frames with actual/predicted traces
over time.

The language explanation can be added after the trajectory model works:

```text
"The model expects the winger to continue outside because the fullback is
holding width and the nearest defender is closing the inside lane."
```

Do not make final-score prediction the first demo. It is too sparse and too
underdetermined from a few frames. A better early target is whether the predicted
movement improves or preserves attacking/defensive value.

## Near-Term Repository Shape

Expected first code layout:

```text
rl_trainer/
  data/
    tracking_dataset.py
  models/
    qwen_football.py
    heads.py
  training/
    train_supervised.py
    eval_trajectory.py
    scaling_fit.py
  modal_train.py
  configs/
    qwen_football_small.yaml
```

Keep the first implementation simple. The project is young, so prefer breaking
changes and direct code over compatibility shims or premature abstraction.
