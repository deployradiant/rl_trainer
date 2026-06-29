# Proposal: ACT-Style Action Chunk Policy

Date: 2026-06-29

## BLUF

Replace the hand-built heading/speed codebook with an ACT-style chunk head that
predicts the selected player's next 1-second trajectory residual:

```text
future_residual_xy: [batch, horizon, 2]
base_future_xy[t] = current_xy + recent_velocity * (t + 1)
pred_future_xy[t] = base_future_xy[t] + future_residual_xy[t]
```

This should be the next step before diffusion, learned motion tokens, chunk-level
MCTS, or GRPO. First prove that direct chunk prediction can overfit and beat
simple motion baselines.

## Why

The current Qwen + H100 path proves the training plumbing works, but the action
space is too hand-authored. Robotics VLA work points to a cleaner split:

```text
Qwen visual backbone -> compact action expert -> structured motion chunk
```

For football, chunks preserve acceleration, curvature, and hesitation better
than a single heading/speed action.

## Model

Keep Qwen visual-stack training as the default. Add a small per-step MLP head:

```text
frames + selected-player history
  -> Qwen visual encoder
  -> visual projection + bbox history + future step embeddings
  -> chunk head
  -> residual trajectory [B, horizon, 2]
```

Only consider a transformer, diffusion, or tokenized motion head after this
simple bounded head fails or plateaus. Initialize the final projection to zero
so the first prediction is constant velocity, then learn bounded residual xy
residuals from there.

## Scope

Implement now:

- Constant-position and constant-velocity baselines.
- ACT-style chunk head.
- Direct PyTorch training loss for full future xy.
- Metrics in logs, history JSON, rollout JSON, final result dict, and web UI.
- Validation on one-sample overfit, fixed 5-sequence slice, and 10%/2% split.

Leave out for now:

- Diffusion or flow matching.
- Learned motion-token vocabularies.
- GRPO or other RL updates.
- Multi-player context.
- Language-conditioned policy prompts.

## Loss And Metrics

Use the simplest supervised objective:

```text
chunk_loss = SmoothL1(pred_future_xy, target_future_xy)
endpoint_loss = SmoothL1(pred_future_xy[:, -1], target_future_xy[:, -1])
loss = chunk_loss + endpoint_loss
```

Report:

```text
ADE/FDE
constant_position_ADE/FDE
constant_velocity_ADE/FDE
codebook_ADE/FDE while the comparison path exists
chunk_ADE/FDE
```

## Validation Gates

Run in order:

1. Static checks.
2. One-sample Modal overfit on `SNMOT-107`, track `9`, start frame `176`.
3. Fixed-seed 5-sequence train/eval slice.
4. Fixed-seed 10% train / 2% eval split.

Only run the 10%/2% split if the 5-sequence run beats constant velocity.

## Acceptance Criteria

One-sample gate:

- Finite loss.
- Nonzero Qwen gradients.
- ADE/FDE roughly match or beat the current reference: ADE `0.012159`, FDE
  `0.021636`.

5-sequence and 10%/2% gates:

- Chunk ADE/FDE beat constant velocity.
- Chunk ADE/FDE match or beat the codebook path while it exists.
- Visual rollouts look plausible in the web player.

If the chunk head loses to constant velocity on held-out clips, stop and improve
state/context before spending H100 time on RL.

## Next Task

Implement baseline metrics and the ACT-style chunk policy head, default training
to the chunk policy, and verify it with the gates above.
