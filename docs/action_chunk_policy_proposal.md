# Proposal: ACT-Style Action Chunk Policy

Date: 2026-06-29

## BLUF

The next implementation should replace the hand-built heading/speed codebook
with an ACT-style chunk head that predicts the selected player's full next
1-second relative trajectory. Add simple baselines first, then compare the chunk
head against constant position, constant velocity, and the current codebook.

Do this before diffusion, motion-token vocabularies, chunk-level MCTS, or GRPO.

## Why

The current Qwen + H100 path proves the plumbing works, but the action space is
too hand-authored:

```text
action = heading_delta_bucket + speed_multiplier_bucket
```

Robotics VLA work points toward structured action chunks. For football, the
chunk should be:

```text
future_relative_xy: [batch, horizon, 2]
```

where:

```text
pred_future_xy[t] = current_xy + pred_relative_xy[t]
```

This preserves acceleration, curvature, and hesitation better than one
heading/speed decision at a time.

## Model Shape

Keep Qwen as the visual backbone. Add a small dedicated action head:

```text
frames + selected-player history
        |
Qwen visual encoder
        |
visual projection + bbox history + future step embeddings
        |
chunk head
        |
relative trajectory chunk: [B, horizon, 2]
```

Start with an MLP-per-future-step head. Only upgrade to a small transformer
decoder if the MLP fails on direction changes.

## Scope

In:

- Constant-position and constant-velocity baselines.
- ACT-style chunk head.
- Qwen visual-stack training as the default.
- Metrics in logs, history JSON, final result dict, rollout JSON, and web UI.
- One-sample, 5-sequence, and 10%/2% validation gates.

Out:

- Diffusion or flow matching.
- Learned motion token vocabularies.
- GRPO or other RL updates.
- Multi-player context.
- Language-conditioned policy prompts.

## Baselines

Every run should report:

```text
constant_position: pred[t] = current_xy
constant_velocity: pred[t] = current_xy + recent_velocity * (t + 1)
current_codebook: Qwen -> heading/speed policy/value heads
chunk_policy: Qwen -> relative trajectory chunk head
```

Primary metrics:

```text
ADE
FDE
baseline_constant_ADE/FDE
baseline_velocity_ADE/FDE
codebook_ADE/FDE while comparison path exists
chunk_ADE/FDE
```

## Loss

Start simple:

```text
chunk_loss = SmoothL1(pred_future_xy, target_future_xy)
endpoint_loss = SmoothL1(pred_future_xy[:, -1], target_future_xy[:, -1])
loss = chunk_loss + endpoint_loss
```

Keep value/action losses out of the first chunk-head version unless needed for a
clean comparison. The first question is whether chunk supervision beats the
current action representation.

## Implementation Plan

1. Add `training/baselines.py`

```text
constant_position(player_bbox, horizon)
constant_velocity(player_bbox, horizon)
```

2. Add `models/action_chunk.py`

```text
ActionChunkHead(visual_proj, player_bbox) -> relative_xy [B, horizon, 2]
```

3. Add a temporary policy selector

```text
--policy-kind chunk
```

Default the proposal branch to `chunk`. Keep the codebook path only for
comparison until we decide to delete it.

4. Update artifacts

`trajectory_rollouts.json` should include:

```text
actual_xy
chunk_pred_xy
baseline_constant_xy
baseline_velocity_xy
codebook_pred_xy if present
metrics for each path
```

5. Update the web player

Allow toggling:

```text
actual
chunk prediction
constant velocity
constant position
codebook/MCTS if present
```

## Validation Gates

Run in order:

1. Static checks.
2. One-sample Modal overfit on:

```text
SNMOT-107, track 9, start frame 176
```

3. Fixed-seed 5-sequence train/eval slice.
4. Fixed-seed 10% train / 2% eval split.

Only run the 10%/2% split if the 5-sequence slice beats constant velocity.

## Acceptance Criteria

One-sample gate:

```text
finite loss
nonzero Qwen gradients
ADE/FDE roughly match or beat current overfit reference
```

Current reference:

```text
ADE: 0.012159
FDE: 0.021636
```

5-sequence gate:

```text
chunk_ADE < velocity_ADE
chunk_FDE < velocity_FDE
chunk_ADE <= codebook_ADE, ideally
chunk_FDE <= codebook_FDE, ideally
```

10%/2% gate:

```text
chunk policy improves or preserves eval ADE/FDE
visual rollouts look at least as clean as the codebook path
```

If the chunk head loses to constant velocity on held-out clips, stop. The next
bottleneck is state/context, not RL.

## Decision

If accepted, the next coding task is:

```text
Implement baseline metrics and an ACT-style chunk policy head.
```

Keep the first implementation narrow. No diffusion, GRPO, learned motion tokens,
or multi-player context until chunks beat the current point/action baseline.
