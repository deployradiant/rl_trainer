# Proposal: ACT-Style Action Chunk Policy

Date: 2026-06-29

## Goal

Replace the next movement-modeling step from "predict one relative heading/speed
action at a time" to "predict a short relative trajectory chunk" while keeping
the system simple, measurable, and compatible with the current Qwen + Modal H100
training loop.

This is the first robotics-derived implementation step from
[robotics_vla_research.md](robotics_vla_research.md). It should happen before
diffusion/flow policies, motion-token vocabularies, MCTS over learned chunks, or
GRPO.

## Why This Step

The current model proves the plumbing works:

- Qwen2-VL full visual-stack finetuning runs on Modal H100.
- The one-sample real SoccerNet overfit gate passes.
- The current relative action codebook can be used by MCTS.

But the current action representation is too hand-authored:

```text
action = heading_delta_bucket + speed_multiplier_bucket
```

That is useful for debugging, but it hides curvature, acceleration, hesitation,
and multi-frame intent. Robotics systems such as ACT-style policies usually
predict action chunks because chunks carry temporal structure directly.

For football, the equivalent chunk is:

```text
relative future movement over the next 1 second
```

## Proposed Model Shape

Keep Qwen as the visual backbone and add a dedicated chunk action expert:

```text
frames + selected-player history
        |
Qwen visual encoder
        |
pooled visual features + player state
        |
chunk transformer / MLP action expert
        |
relative trajectory chunk: [batch, horizon, 2]
```

For the first implementation, do not add a complicated decoder. A compact
transformer or MLP head is enough:

```text
visual projection
selected-player bbox history
step embeddings for future timesteps
        |
chunk head
        |
relative dx/dy for every future frame
```

The model predicts a full chunk in one forward pass:

```text
pred_relative_xy[t] = future_xy[t] - current_xy
pred_future_xy[t] = current_xy + pred_relative_xy[t]
```

## Scope

### In Scope

- Add constant-position and constant-velocity baselines to evaluation.
- Add an ACT-style chunk head that predicts the full future relative trajectory.
- Keep Qwen visual-stack training as the default path.
- Keep the current codebook/value/MCTS path available only long enough to compare
  it against the chunk head during this proposal.
- Add metrics that make the comparison obvious in `training_history.json`,
  `sequence_manifest.json`, and visualizer rollouts.
- Run one-sample overfit, 5-sequence slice, and then 10%/2% split if the smaller
  gates pass.

### Out Of Scope

- Diffusion or flow matching.
- Learned FAST-style action tokenization.
- GRPO or other RL updates.
- Multi-player state tokens.
- Language-conditioned policy prompts.
- Tactical value labels beyond the existing learned action-value head.

Those are good next steps, but only after the chunk policy beats the current
relative-codebook path and simple baselines.

## Baselines

Every run should report these in addition to learned-policy metrics:

### Constant Position

Predict the selected player's last observed center for every future frame.

```text
pred[t] = current_xy
```

### Constant Velocity

Estimate recent velocity from the input history and extrapolate.

```text
velocity = (xy[-1] - xy[0]) / (input_frames - 1)
pred[t] = current_xy + velocity * (t + 1)
```

### Current Relative Codebook

The current model path:

```text
Qwen visual features -> policy/value heads -> relative heading/speed actions
```

This is the baseline the chunk policy must beat before we retire the codebook.

## Metrics

Primary metrics:

- `ADE`: average displacement error over the future window.
- `FDE`: final displacement error.
- `baseline_constant_ADE/FDE`.
- `baseline_velocity_ADE/FDE`.
- `codebook_ADE/FDE` while the old path exists.
- `chunk_ADE/FDE`.

Secondary metrics:

- Endpoint-weighted Smooth L1 loss.
- Curvature or acceleration error.
- Off-frame / impossible movement rate.
- Runtime per epoch.
- Visual quality in `trajectory_rollouts.json`.

Acceptance should be based on held-out eval clips, not only the one-sample
overfit gate.

## Loss

Start with direct supervised losses:

```text
chunk_loss = SmoothL1(pred_future_xy, target_future_xy)
endpoint_loss = SmoothL1(pred_future_xy[:, -1], target_future_xy[:, -1])
loss = chunk_loss + endpoint_weight * endpoint_loss
```

Recommended initial weights:

```text
endpoint_weight = 1.0
```

Keep the value/action losses out of the first chunk-head version unless needed
for MCTS compatibility. The point of this experiment is to isolate action
representation quality.

## Implementation Plan

### Step 1: Baseline Metrics

Add `training/baselines.py`:

```text
constant_position(player_bbox, horizon)
constant_velocity(player_bbox, horizon)
evaluate_baselines(loader)
```

Add baseline metrics to:

- training logs
- `training_history.json`
- final result dict
- `trajectory_rollouts.json`
- visualizer metric cards

Done when every Modal run shows learned-vs-baseline ADE/FDE.

### Step 2: Chunk Head

Add `models/action_chunk.py` with a small `ActionChunkHead`.

Recommended first shape:

```text
input:
  visual_proj: [B, D]
  player_bbox: [B, input_frames, 4]
output:
  relative_xy: [B, horizon, 2]
```

Implementation sketch:

```text
state = concat(visual_proj, flattened bbox history)
future_step_embedding[t]
head(state, step_embedding[t]) -> dxdy[t]
```

This can be an MLP over each future step before adding a transformer. If the MLP
beats the codebook, great. If it underfits direction changes, upgrade to a
small transformer decoder over future step tokens.

### Step 3: Training Switch

During the experiment, add a single explicit model target:

```text
--policy-kind chunk
```

The default for the proposal branch should be `chunk`. Do not preserve a
compatibility maze. Keep the old codebook path only as a comparison until we
decide to delete it.

### Step 4: Artifacts

Update `save_prediction_artifacts` so rollouts include:

```text
pred_xy
baseline_constant_xy
baseline_velocity_xy
metrics for all three
```

The web player should be able to show/hide:

```text
actual
chunk prediction
constant velocity
constant position
MCTS/codebook if present
```

### Step 5: Validation Runs

Run gates in this order:

1. Local static checks.
2. Modal one-sample overfit on the existing real sample:

```text
SNMOT-107, track 9, start frame 176
```

3. Modal 5-sequence train/eval slice with fixed seed `7`.
4. Modal 10% train / 2% eval split with fixed seed `7`.

Only run the larger split if the 5-sequence slice beats constant velocity.

## Acceptance Criteria

### One-Sample Gate

The chunk head must overfit the same fixed sample:

```text
ADE <= current codebook overfit ADE, or visibly equivalent
FDE <= current codebook overfit FDE, or visibly equivalent
finite losses
nonzero Qwen gradients
```

Current reference:

```text
ADE: 0.012159
FDE: 0.021636
```

### 5-Sequence Gate

The chunk head must beat constant velocity on held-out eval:

```text
chunk_ADE < velocity_ADE
chunk_FDE < velocity_FDE
```

It should also beat or match the current relative-codebook policy:

```text
chunk_ADE <= codebook_ADE
chunk_FDE <= codebook_FDE
```

### 10%/2% Gate

The chunk head should preserve or improve eval ADE/FDE and produce cleaner
visual rollouts in the web player.

If it overfits training but loses to constant velocity on eval, stop and improve
state/context before adding RL.

## Expected Outcomes

Best case:

- Chunk policy beats the codebook and constant velocity.
- Rollouts are smoother and easier to inspect.
- We can delete the hand-built heading/speed codebook later.
- MCTS can move from codebook actions to chunk candidates.

Acceptable case:

- Chunk policy matches the codebook but simplifies the path toward diffusion,
  motion tokenization, and MCTS over chunks.

Bad case:

- Chunk policy fails to beat constant velocity.
- This means the model needs better state inputs before a fancier action head:
  ball, other players, pitch coordinates, or stronger tracking features.

## Risks

1. SoccerNet bbox/image centers may encode camera motion more than pitch motion.
2. Single-player history may not contain enough context for sharp turns.
3. Full-Qwen visual training can memorize small slices without generalizing.
4. A chunk head may look smoother but be less tactically plausible.
5. Adding a `policy-kind` branch can get messy if we keep both paths too long.

## Decision After This Proposal

If the proposal is accepted, the next coding task should be:

```text
Implement baseline metrics and an ACT-style chunk policy head.
```

The first PR should not include diffusion, GRPO, learned action tokens, or
multi-player context. Keep the knife sharp: prove that chunks beat points.
