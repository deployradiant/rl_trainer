# Football Player Tracking Training Plan

The first objective is player tracking and short-horizon player movement
prediction on SoccerNet Tracking. Match state, team state, scoreline simulation,
and World Cup outcome modeling are later layers. They should not be in the
current build until the tracking loop is reliable and inspectable.

## Current Scope

Train and debug the existing Qwen football trajectory stack:

```text
SoccerNet frames + selected player box/history
        -> frozen Qwen2-VL visual features
        -> trainable football heads
        -> action logits, value estimates, future xy rollout
        -> visualizer + debug trace
```

Primary task:

> Given a short frame window and one selected player, predict the next 1-2
> seconds of that player's movement, compare against simple motion baselines,
> and make failures visible.

Non-goals for this phase:

- No match-state schema.
- No team-state schema.
- No scoreline or outcome simulator.
- No event model.
- No tactical value model beyond the existing imitation/search-guidance head.
- No World Cup prediction workflow.

## Execution Model

Use Modal by default.

- Training and debug runs are Modal GPU jobs following the `modal_train.py`
  pattern.
- SoccerNet downloads live on the `soccernet-tracking` Modal volume.
- Qwen/Hugging Face weights live on the `hf-cache` Modal volume.
- Qwen visual-feature caches should live on the Modal volume and be keyed by:
  `(source, split, sequence, frame_range, image_size, model_name, processor_version)`.
- Run artifacts should be keyed by `(git_sha, command, seed, schema_version)`.
- Local execution is for ruff, small plumbing tests, and reading artifacts.

The point is not to optimize around limited local compute. The point is to make
small real SoccerNet Modal runs fast through caching and deterministic enough to
use as the main development loop.

## Where The Repo Is Today

Already present:

- `data/tracking_dataset.py`: SoccerNet Tracking loader for frame windows,
  selected-player boxes, and future xy targets.
- `models/qwen_football.py`: frozen Qwen2-VL feature extraction, relative-motion
  action codebook, policy head, and learned per-action value head.
- `training/train_supervised.py`: supervised loop, feature caching option,
  teacher-forced action labels, MCTS option, and artifact writing.
- `training/mcts.py`: PUCT search over the discrete relative-motion codebook.
- `training/artifacts.py`: rollout JSON and preview image generation.
- `modal_train.py`: Modal training entrypoint with SoccerNet and HF cache volumes.
- `modal_visualize.py` and `web/index.html`: frame playback and actual/pred/MCTS
  trajectory overlays.

Missing and load-bearing:

- No scripted Modal debug gate.
- No `debug_trace.json`.
- No activation/logit/value/gradient diagnostics.
- No constant-position or constant-velocity baseline in the first loop.
- No schema validation for artifacts.
- No CI.
- No tracking data validation report.
- Current coordinates are image-normalized; camera motion is still confounded
  with player motion.

## Step 1: Modal One-Player Debug Loop

This is the first real implementation target.

Scope:

- **Fast Modal debug gate.** Script a tiny SoccerNet run into a single entrypoint
  (`scripts/debug_one_sample.sh` or a make target): Modal full-GPU job, real
  SoccerNet tracking data, one deterministic high-motion sample, seed 7, one
  forward pass, one backward pass, one short training loop, one rollout, and one
  MCTS pass. It should use the Modal SoccerNet volume, Hugging Face cache, and
  Qwen feature cache; first run may populate caches, later runs should reuse
  them. This is the default gate for every training change.
- **Debug trace artifact.** The fast gate writes `debug_trace.json` with enough
  model internals to explain failures:
  - batch metadata: sequence, split, track id, frame range, horizon, coordinate
    space, seed, checkpoint ids, cache keys;
  - baseline paths: constant position and constant velocity;
  - rollout paths: target, learned, greedy, sampled if available, MCTS if enabled;
  - policy details: action labels, action logits, top-k probabilities, entropy,
    teacher-forced vs rollout action choices;
  - value details: imitation-value outputs, selected-action values, loss terms;
  - activation stats: mean/std/min/max and saturation counts for the pooled
    Qwen visual features (the head input — catches stale/corrupt feature cache
    entries and fp16 saturation) and for each trainable head;
  - gradient stats: per-head grad norms, zero-gradient warnings, NaN/Inf checks,
    exploding-gradient warnings.
- **Quality Modal e2e gate.** Script the existing overfit-style command into a
  second entrypoint (`scripts/e2e_train.sh` or a make target): a Modal full-GPU
  job on the same cached SoccerNet volume with a tiny deterministic
  train/eval-sequence slice (`--train-sequence-count 1 --eval-sequence-count 1`,
  few epochs, seed 7) that runs the whole pipeline — download/cache, dataset,
  frozen Qwen features, training, eval, MCTS, artifact writing — and then
  validates every emitted artifact (`sequence_manifest.json`,
  `training_history.json`, `trajectory_head.pt`, `trajectory_preview.png`,
  `trajectory_rollouts.json`, `debug_trace.json`) against schemas. The e2e run
  emits the same `debug_trace.json` for its first eval sample so the trace
  schema is exercised by both gates. Run this before merging substantive
  training changes; the fast debug gate runs more often.
- **Motion baselines in the first loop.** Add constant-position and
  constant-velocity baselines immediately, written into `training_history.json`,
  `trajectory_rollouts.json`, and `debug_trace.json`. The visualizer shows these
  next to target, learned, and MCTS paths from the first debugging pass.
- **Modal cache contract.** Large data and derived artifacts are Modal-first:
  `soccernet-tracking` for SoccerNet downloads, `hf-cache` for Qwen weights, a
  feature-cache path keyed by `(source, split, sequence, frame_range, image_size,
  model_name, processor_version)`, and run artifact directories keyed by
  `(git_sha, command, seed, schema_version)`.
- **Plumbing tests only where synthetic inputs are the right tool:**
  codebook labeling round-trip (`action_labels` of a codebook-generated future
  recovers the actions), deterministic sample selection, baseline generation,
  artifact schema validation, and dataset windowing. No mocked training loops.
- GitHub Actions running ruff + the plumbing tests on push. The e2e training
  gate runs on Modal — manually at first, wired into CI via a Modal token if
  that proves worth it.
- `docs/artifacts.md` (or JSON Schema files) versioning every artifact,
  including the report artifacts introduced by later steps. Every artifact
  gains a `schema_version`; the e2e gate is what enforces the schemas.

Required artifacts:

- `sequence_manifest.json`
- `training_history.json`
- `trajectory_head.pt`
- `trajectory_preview.png`
- `trajectory_rollouts.json`
- `debug_trace.json`

`debug_trace.json` must include:

- sample metadata: split, sequence, track id, frame range, horizon, seed;
- cache metadata: model name, processor version, feature-cache key;
- selected-player input boxes and target xy;
- baseline xy paths: constant position and constant velocity;
- learned xy paths: teacher-forced, greedy rollout, MCTS if enabled;
- action labels, action logits, top-k probabilities, entropy;
- imitation/search value outputs and selected-action values;
- loss terms by head;
- activation mean/std/min/max for pooled Qwen visual features and trainable heads;
- per-head gradient norms;
- zero-gradient warnings;
- NaN/Inf checks;
- speed/acceleration sanity checks in current coordinate space.

Gate:

- Modal job completes on real SoccerNet data.
- Artifacts exist and validate against schemas.
- Loss is finite.
- One short overfit loop reduces supervised loss on the selected sample.
- Trainable heads have nonzero finite gradients.
- Constant-position and constant-velocity baselines are present in both JSON and
  visual output.
- The visualizer can show the debug sample without manual artifact editing.

## Step 2: Artifact Schemas And Visual Debug Panels

Scope:

- Add artifact schemas or `docs/artifacts.md`.
- Add schema version to all training artifacts.
- Add visualizer panels for:
  - actual vs baseline vs learned vs MCTS trajectories;
  - action logits and target action bucket;
  - value outputs;
  - loss terms;
  - activation histograms or summary stats;
  - gradient norms and warnings.
- Add plumbing tests for:
  - action-codebook label round trip;
  - artifact schema validation;
  - deterministic sample selection;
  - baseline path generation.

Gate:

- `uvx pre-commit run --all-files` passes.
- Debug artifacts from Step 1 pass schema validation.
- Visualizer displays the one-sample debug trace.

## Step 3: Tracking Baselines And Data Validation

Scope:

- Extend the Step 1 baselines to eval runs:
  - constant position;
  - constant velocity;
  - capped acceleration;
  - Kalman-style extrapolation if it remains simple.
- Add `python -m data.validate_tracking`.
- Validation report includes:
  - train/eval sequence overlap;
  - future-frame leakage checks;
  - track continuity through input and target windows;
  - bbox bounds and size sanity;
  - motion buckets: stationary, walking, sprinting, turning;
  - sample counts by sequence and track id.
- Write baseline metrics into `training_history.json` and
  `trajectory_rollouts.json`.

Metrics:

- ADE/FDE at 0.4s, 1.0s, and 2.0s where horizon supports it.
- Aggregate ADE/FDE.
- Moving-player ADE/FDE.
- Motion-bucket ADE/FDE.
- Action accuracy and top-k action accuracy.
- Speed/acceleration violation rate in image-normalized space.

Gate:

- The deterministic 10%/2% SoccerNet split passes validation.
- Learned model beats constant velocity on held-out moving-player samples before
  any backbone tuning.
- Failure cases are visible in the visualizer.

## Step 4: Modal Feature Cache And Throughput

Scope:

- Batch visual feature extraction without the per-image PIL round-trip,
  populating the persistent feature cache from the Step 1 Modal cache contract
  (same key, no second cache format) at full-split scale.
- Add cache hit/miss counts and feature extraction throughput to run artifacts.
- Scale training 10% -> 25% -> 100% of train sequences on a full GPU, holding
  the seeded eval split fixed. Begin the data scaling curve from
  docs/training_plan.md (nested subsets, multiple seeds at small sizes).
- Then K-sample futures (mixture head or sampled decoding over the codebook)
  with minADE@K/minFDE@K and NLL/CRPS.

Artifacts:

- `feature_cache_manifest.json`
- `scaling_report.json`
- throughput fields in `training_history.json`

Gate:

- Feature cache is reused across Modal runs.
- Eval ADE/FDE improves or fails explainably as data scales.
- Throughput numbers are recorded for every scaling run.

## Step 5: Better Trajectory Modeling

Scope:

- Keep Qwen frozen until Step 3 and Step 4 gates pass.
- Add one improvement at a time:
  - continuous residual on top of the discrete action codebook;
  - mixture head for multi-modal futures;
  - sampled codebook decoding;
  - selected-player crop or mask;
  - nearby player context.

Losses:

- Smooth L1 or MSE for xy.
- Cross entropy for action labels.
- NLL/CRPS once probabilistic outputs exist.
- Behavior-cloning anchor for any post-training changes.

Metrics:

- ADE/FDE.
- minADE@K and minFDE@K.
- NLL/CRPS for probabilistic futures.
- Action accuracy and top-k action accuracy.
- Violation rate.

Gate:

- Each modeling change beats the previous checkpoint on held-out moving-player
  samples or is reverted.
- Multi-sample futures improve minADE@K without producing physically implausible
  averages or samples.

## Step 6: MCTS Realism And Post-Training

Scope:

- Keep MCTS focused on selected-player movement.
- Add realism penalties:
  - low policy likelihood;
  - impossible speed;
  - impossible acceleration;
  - large teleport steps;
  - off-image/off-pitch once pitch coordinates exist.
- Promote `search_diagnostics.json` from stub-like summary to a first-class
  artifact:
  - root priors;
  - visits;
  - values;
  - selected path;
  - realism penalties;
  - policy/value disagreement.
- Add post-training only after the supervised tracking loop and baselines are
  stable:
  - sample K futures;
  - score with imitation value and realism penalties;
  - normalize rewards within the group;
  - update with AWR/DPO/GRPO-style objective;
  - keep behavior-cloning and KL anchors.

Gate:

- MCTS improves imitation/search value without increasing physical violations.
- Post-training improves selected metrics without degrading ADE/FDE beyond a
  stated tolerance.
- Manual review finds no obvious reward exploitation.

## Step 7: Pitch Coordinates And Multi-Player Tracking

Do this after the selected-player loop is stable.

Scope:

- Add pitch calibration confidence where available.
- Project player footpoints or center-bottom box points to pitch coordinates.
- Add nearby teammate/opponent context.
- Add ball context if available.
- Move from selected-player prediction toward multi-player state rollout.

Gate:

- Pitch-space metrics exist.
- Image-space and pitch-space artifacts are both inspectable.
- Player movement remains better than motion baselines.

## Visualizer Requirements

The visualizer is a debugging tool, not only a demo.

It should show:

- input frames;
- selected-player box;
- actual future path;
- constant-position path;
- constant-velocity path;
- learned greedy path;
- MCTS path;
- action logits and target action bucket;
- value outputs;
- loss terms;
- activation summary stats;
- gradient norms and warnings;
- worst/failure examples by FDE and motion bucket.

## Do Not Do Yet

- Do not build match-state or team-state abstractions.
- Do not build scoreline or World Cup outcome simulation.
- Do not start event modeling.
- Do not tune the Qwen backbone before the frozen-feature tracking head beats
  motion baselines.
- Do not add post-training until the supervised tracking loop is measurable,
  visual, and stable.
- Do not build broad orchestration frameworks; artifact contracts and Modal
  scripts are enough for now.
