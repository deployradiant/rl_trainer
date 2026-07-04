# Football Training And Simulation Plan

The goal is to simulate a current World Cup match from a saved match state, run
many futures, and show where the game is likely to go. SoccerNet is the training
ground for player movement. World Cup match snapshots are the inference target.

The work stays centered on what we train:

1. A score/outcome simulator (M0).
2. A football event model (M1).
3. A player trajectory/world model (M2).
4. A tactical value model (M3).
5. A post-trained policy searched with MCTS (M4).

The plan is a single ordered build. Every step ends in a versioned artifact
that can be tested, inspected, and shown in the visualizer or a report — no
waterfall, no grand integration step at the end. Each step makes the demo
strictly better than the one before it.

Execution model: all training and debug runs are Modal GPU jobs following the
`modal_train.py` pattern, on a full GPU. There is no separate CPU training
track and no effort spent optimizing around compute constraints. Local
execution is for tests and the Level 0 simulator only. Large downloads and
derived features live on Modal volumes: SoccerNet data, Hugging Face weights,
Qwen visual-feature caches, run artifacts, and debug traces should be reused
across runs instead of redownloaded or recomputed.

## Where The Repo Is Today

Done and verified:

- **Level 0 outcome simulator (M0 v0, heuristic).** `data/match_state.py`
  defines the snapshot schema; `simulation/monte_carlo.py` runs a seeded
  Poisson-style scoreline Monte Carlo; `simulation/run_match.py` writes the
  full artifact set (`match_state.json`, `simulation_summary.json`,
  `rollout_samples.json`, `search_diagnostics.json` stub, `prediction_card.md`).
- **M2 v0 trajectory stack.** Frozen Qwen2-VL-2B visual features + a
  relative-motion action codebook (9 headings x 5 speed multipliers = 45
  actions), a step-aware rollout policy head, and teacher-forced training in
  `models/qwen_football.py` / `training/train_supervised.py`. Deterministic
  10%/2% SoccerNet sequence splits with seed 7.
- **Learned-value MCTS v0.** `training/mcts.py` runs PUCT over the codebook
  using policy logits as priors and a learned per-action value head. The
  one-sample learned-value MCTS overfit gate is green (see README command).
- **Visualizer.** Modal-served web app (`modal_visualize.py`, `web/index.html`)
  with sequence selection, frame-by-frame playback, and actual/predicted/MCTS
  trace overlays driven by `trajectory_rollouts.json`.

Not built yet, and load-bearing for what follows:

- **No tests, no CI, and no repeatable end-to-end training gate.** The README
  overfit command is the closest thing to a pipeline check, but it is not
  scripted, not schema-validated, and not run as a matter of course. The
  verification strategy is real runs first: a scripted end-to-end training job
  on real SoccerNet data is the primary gate; synthetic fixtures are for small
  plumbing checks only.
- **No motion baselines or model x-ray traces.** The M2 gate "beats constant
  velocity" cannot be evaluated today because constant position/velocity/Kalman
  baselines do not exist anywhere in the repo. The current artifacts also do
  not show activations, logits, value components, loss terms, or gradient norms,
  so failures are harder to localize than they should be.
- **No event data ingestion and no M1.** M0 hazard rates are hard-coded
  heuristics with no backtest against historical matches.
- **No tactical value.** The existing value head learns *imitation advantage*
  (how much worse each codebook action is than the best action, measured
  against the ground-truth future). That is a search-guidance signal, not a
  tactical judgment. M3 is entirely open.
- **Integration gaps.** `run_match` only knows M0; `search_diagnostics.json`
  is a stub; no next-goal/next-shot outputs; no snapshot capture or post-match
  audit tooling; trajectory coordinates are image-normalized (camera motion is
  confounded with player motion) with no pitch calibration.

Known small fixes folded into the steps below rather than fixed ad hoc: M0
ignores `stoppage_minute`; horizon-truncated runs report "state at horizon" as
if it were the final result; per-sample PIL round-trips make feature
extraction slow at scale.

## Calendar Constraint

The 2026 World Cup final is July 19, 2026. Live, pre-registered predictions —
snapshot saved before the simulated window, audited after — are only possible
until then. That is why the M0 fit and snapshot tooling come immediately after
the foundation step. After July 19 the World Cup surface becomes replay-based
backtesting, which is still fully useful for calibration claims.

## End State

Input:

- Current match snapshot: score, clock, teams, lineups, cards, substitutions,
  possession/event state, and optional tracking/video-derived player positions.
- Model checkpoints for outcome, event, trajectory, value, and policy heads.
- Rollout count, horizon, and seed.

Command shape:

```bash
python3 -m simulation.run_match \
  --snapshot-file snapshots/MATCH_ID/SNAPSHOT_EPOCH.json \
  --models m0,m1 \
  --rollouts 10000 \
  --horizon-minutes 30 \
  --seed 7 \
  --out artifacts/world_cup/MATCH_ID/SNAPSHOT_EPOCH
```

Output:

- `match_state.json`: exactly what was known at simulation time.
- `simulation_summary.json`: scoreline and win/draw/loss or advance
  probabilities always; next goal, next shot, and xG when M1 is enabled.
- `rollout_samples.json`: representative simulated futures.
- `search_diagnostics.json`: MCTS branch priors, visits, values, and selected
  path when search is enabled; explicit `not_available` otherwise.
- `prediction_card.md`: timestamped prediction plus limitations.
- Post-match audit comparing predicted distributions to the observed future.

## Training Stack

| Layer | What We Train | Main Data | Output | Status |
| --- | --- | --- | --- | --- |
| M0 | Score/outcome model | SoccerNet goal/card labels + results | Future scoreline probabilities | v0 heuristic shipped; unfitted, unbacktested |
| M1 | Event transition model | SoccerNet action-spotting labels | Next event, time, shot, goal | not started |
| M2 | Trajectory/world model | SoccerNet tracking frames and boxes | Player movement chunks | v0 shipped; overfit gate green; no baselines/debug trace yet |
| M3 | Tactical value model | SoccerNet events + trajectories | Chance/possession value | not started (imitation-advantage head exists, is not this) |
| M4 | Post-trained policy | Candidate rollouts scored by M3/MCTS | Better action priors | MCTS v0 shipped; post-training loop not started |

## Build Order

Eight steps, in order. Each lists scope, artifacts, and its gate. The
visualizer grows with every step — each step ships its visual surface together
with the model or tool, not later. The Step 1 Modal debug gate is rerun for
every later training change, and the quality e2e gate grows with each step's
new artifacts: a training change is not done until the real SoccerNet run
completes and its artifacts validate.

### Step 1: Foundation — Modal debug loop, baselines, artifact contracts

The primary check is a real training run on real data, not mocks or synthetic
fixtures. Synthetic inputs are allowed only for small plumbing checks where a
real run would prove nothing extra.

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
- **Level 0 simulator smoke.** A second scripted invocation covers the Level 0
  simulator artifact set (CPU, runs anywhere) so match-state JSON and Monte
  Carlo determinism stay covered.
- **Plumbing tests only where synthetic inputs are the right tool:**
  match-state validation, Monte Carlo determinism for seed 7, codebook
  labeling round-trip (`action_labels` of a codebook-generated future
  recovers the actions), dataset windowing. No mocked training loops.
- GitHub Actions running ruff + the plumbing tests on push. The e2e training
  gate runs on Modal — manually at first, wired into CI via a Modal token if
  that proves worth it.
- `docs/artifacts.md` (or JSON Schema files) versioning every artifact,
  including the report artifacts introduced by later steps. Every artifact
  gains a `schema_version`; the e2e gate is what enforces the schemas.

Artifacts: `scripts/debug_one_sample.sh`, `scripts/e2e_train.sh`, `debug_trace.json`,
baseline fields in `training_history.json` and `trajectory_rollouts.json`,
visualizer debug panels, `tests/` (plumbing only), green CI run, schema doc,
one archived debug artifact set and one archived e2e artifact set on the Modal
volume as reference runs.

Gate: the fast Modal debug job completes on real SoccerNet data; loss is finite;
one short training loop reduces the supervised loss on the selected sample;
trainable heads have nonzero finite gradients; constant-position and
constant-velocity baselines are present; all artifacts validate against schema.
The quality Modal e2e job completes before merging substantive training changes.
Plumbing tests run in seconds on CPU.

### Step 2: Live World Cup path — M0 fit, snapshot capture, audit

The calendar-critical step. Small, and everything after it benefits from the
prediction/audit loop existing.

Scope:

- Extract goal times, cards, and final scores from the SoccerNet
  action-spotting labels (same downloader and Modal volume as the tracking
  data). Fit scoring hazards conditioned on score state, time, and red cards.
  Team strength stays a snapshot *input* (a small manual ratings table for
  World Cup sides; `TeamState.strength` already exists), not a fitted
  quantity.
- Backtest from minute 0, halftime, and minute 75 snapshots of held-out
  SoccerNet games.
- Make hazards time-varying inside the rollout so the simulator emits
  next-goal time distributions; add fan-chart data to
  `simulation_summary.json`.
- Fix the known M0 issues here: honor `stoppage_minute`; label
  horizon-truncated output as `state_at_horizon` rather than a final result.
- Snapshot capture CLI (manual entry first) writing a valid `MatchState` with
  `snapshot_epoch_utc`, plus an append-only `predictions_ledger.jsonl`
  recording (snapshot path, artifact dir, git SHA, seed) at prediction time.
- Post-match audit CLI: predicted probability of what happened, Brier
  contribution, rank of observed scoreline, appended to a cumulative
  `calibration_ledger.json`.

Artifacts: `training/fit_outcome.py`, fitted parameters in a checked-in
config, `backtest_report.json`, `simulation/capture_snapshot.py`,
`simulation/audit.py`, ledger files, at least one real World Cup snapshot
committed under `snapshots/`. Visual: match outcome panel (scoreline
histogram, win/draw/loss bar) and next-goal fan chart.

Gate: fitted M0 beats the current hard-coded heuristic and a static pre-match
team-strength baseline on NLL and Brier across all three snapshot times;
deterministic for seed 7; a real knockout-match snapshot captured, simulated,
and audited before/after the simulated window.

### Step 3: Make M2 falsifiable at eval scale — validation, stronger baselines, naming

Scope:

- Extend the Step 1 baselines across the full deterministic eval pass and add
  capped acceleration plus Kalman-style extrapolation. Written into
  `training_history.json` and `trajectory_rollouts.json`; ADE/FDE at
  0.4s/1.0s/2.0s reported by motion bucket
  (stationary/walking/sprinting/turning), not just aggregate.
- Tracking data validation CLI (`python -m data.validate_tracking`):
  train/eval sequence overlap check, future-frame leakage check, track
  continuity through input+target windows, bbox bounds/size sanity,
  motion-bucket histogram. Violations listed, not silently dropped.
- Rename the existing value head to `imitation_value` in code and artifacts so
  it can never be confused with the tactical value model (M3). Zero behavior
  change.

Artifacts: `training/baselines.py`, extended eval report,
`data_validation_report.json`, rename PR + schema bump. Visual: baseline
trace overlaid next to actual/pred/MCTS in the existing player view, reusing
the Step 1 debug panels at eval scale.

Gate: held-out moving-player samples must beat constant velocity before any
backbone tuning. The 10%/2% split passes validation, and the debug trace still
shows finite activations and gradients for the Step 1 reference sample.

### Step 4: Scale M2 on Modal — throughput, full split, probabilistic futures

Scope:

- Batch visual feature extraction without the per-image PIL round-trip,
  populating the persistent feature cache from the Step 1 Modal cache contract
  (same key, no second cache format) at full-split scale.
- Scale training 10% -> 25% -> 100% of train sequences on a full GPU, holding
  the seeded eval split fixed. Begin the data scaling curve from
  docs/training_plan.md (nested subsets, multiple seeds at small sizes).
- Then K-sample futures (mixture head or sampled decoding over the codebook)
  with minADE@K/minFDE@K and NLL/CRPS.

Artifacts: cache module, `scaling_report.json`, eval-metric-vs-data curve,
extended rollout schema. Visual: sample fan rendering in the player view.

Gate: eval ADE/FDE improves with data and beats the Step 3 baselines at 100%
of the current split; minADE@K beats the deterministic head; samples respect
speed caps; throughput numbers recorded.

### Step 5: M1 event model, wired into the simulator

Scope:

- Train the event transition model (spec below) on SoccerNet action-spotting
  event sequences as a Modal GPU job. Small MLP/sequence model over recent
  events; separate from Qwen so it stays debuggable.
- Wire into rollouts as event-chain simulation between goals.
- Give `run_match` capability flags: `--models m0[,m1,...]` with graceful
  degradation; each layer adds fields to `simulation_summary.json`; absent
  layers produce explicit `not_available` blocks.

Artifacts: `training/train_events.py`, `event_model_report.json` with
calibration by game state and time bucket, updated CLI + schema bump. Visual:
event rollout tree panel.

Gate: beats team-rate and time-bucket baselines for next event type, next
shot, and next goal; the same `run_match` command runs with any subset of
available checkpoints.

### Step 6: M3 tactical value

Scope: per the M3 spec below, structured-input model (no VLM features at
first): P(shot within T) and P(goal within T) straight from the SoccerNet
event labels; possession-value and turnover targets activate once events are
joined with the tracking clips.

Artifacts: `training/train_value.py`, `value_model_report.json` with
calibration by score state/zone/time. Visual: pitch value heatmap panel.

Gate: beats location-only and team-rate heuristics; does not collapse to "run
toward goal"; calibrated enough that search cannot exploit obvious bugs.

### Step 7: Search realism, then post-training

Scope, in two halves that ship separately:

- Realism penalties in trajectory MCTS (off-pitch, speed cap,
  teleport/acceleration violations, low policy likelihood); a real
  `search_diagnostics.json` at the trajectory level (root branch priors,
  visits, values, selected path — the data already exists in
  `training/mcts.py`); violation-rate metrics in eval.
- M4 post-training: sample K candidate futures from M2, score with imitation
  value + tactical value + realism penalties, normalize rewards within the
  group, update with advantage-weighted regression or a GRPO-style objective,
  keep a behavior-cloning + KL anchor. Implemented directly in PyTorch per
  docs/training_plan.md (no Trainer frameworks).

Artifacts: penalized search + diagnostics artifact, `training/post_train.py`,
before/after eval report. Visual: MCTS root branch table; side-by-side
policy-vs-post-trained rollout comparison.

Gate (search): MCTS matches or beats the raw policy on imitation value without
increasing physical-violation rate. Gate (post-training): value metrics
improve, ADE/FDE degradation stays within a stated tolerance, KL from the
supervised policy bounded, violation rate flat, manual review finds no reward
exploits.

### Step 8: Full integration and the failure gallery

Scope: `run_match` runs M0+M1 always, attaches M2/M3/M4 player-level output
for snapshots where tracking or video-derived state exists; post-match audits
accumulate in the calibration ledger; the visualizer gains a failure gallery
(worst held-out samples by FDE, identity switches, calibration misses, reward
exploits) that is curated continuously from Step 3 onward.

Artifacts: integrated CLI, growing calibration ledger, failure gallery.

Gate: one command produces the full End State artifact set from a real
snapshot, and every claim in the prediction card traces to a backtest or
audit artifact.

## Model Specs

The per-model specs below are the reference for the steps above.

### M0: Score And Outcome Model

- Fit scoring hazards from historical matches; condition on score state, time,
  red cards, team strength, knockout context.
- Loss/metrics: NLL for final scoreline/outcome; Brier and calibration for
  win/draw/loss and next-goal; backtests from minute 0, halftime, minute 75.
- Gate: beats static pre-match team-strength baseline; deterministic for
  seed 7.

### M1: Event Transition Model

- Data: SoccerNet action-spotting event sequences (event type + timestamp per
  event, with team side and score derivable from the label stream). Location,
  possession, and xG fields are extensions that require joining events with
  tracking or adding a richer event corpus later. Current World Cup feeds are
  inference snapshots only, never training data for pre-registered
  predictions.
- Inputs: clock, score, strengths, cards, subs, previous N events; possession
  team, field zone, player role, and pressure proxy once available.
- Targets: next event type, time-to-event, shot and goal probability;
  location/zone and possession continuation once available.
- Model: small tabular/MLP or sequence transformer; separate from Qwen.
- Loss/metrics: CE for type; NLL/pinball for time; BCE/Brier for shot/goal;
  calibration by game state and time bucket.
- Gate: beats team-rate and time-bucket baselines.

### M2: Trajectory And World Model

- Data: SoccerNet tracking frame windows and MOT boxes. Image-normalized
  coordinates for now — note that camera pan is confounded with player motion
  in this space, which is exactly why the constant-velocity baseline (Step 1),
  eval-scale baseline reporting (Step 3), and motion-bucket reporting are
  mandatory. Move to pitch coordinates (with calibration confidence) before
  making any tactical claims.
- Inputs: last 3-8 frames, selected player box, recent xy history; later ball
  state, nearby players, team side, event context, pitch calibration.
- Targets: next 1-2s of selected-player xy; codebook action labels; later
  multi-player and K-sample futures.
- Model: frozen Qwen2-VL features + trainable heads (current baseline). Keep
  the VLM frozen until the Step 3 gates pass. Add mixture/residual outputs
  after the discrete baseline.
- Loss/metrics: Smooth L1/MSE for xy; CE for actions; NLL/CRPS once
  probabilistic; ADE/FDE at 0.4/1.0/2.0s; minADE@K; speed/acceleration/
  off-pitch/teleport violation rates.
- Gates: overfit gate (done); held-out moving samples beat constant velocity;
  results reported by motion bucket.

### M3: Tactical Value Model

- Data: event sequences labeled by future shot/goal/possession outcomes;
  trajectory rollouts aligned to event outcomes where available;
  counterfactual negatives from perturbed but physically plausible
  trajectories.
- Targets: P(shot within T), P(goal within T), possession value / xT delta,
  turnover risk, optional pairwise preference between candidate actions.
- Model: MLP/transformer over structured state first; VLM features only after
  the structured model is calibrated.
- Loss/metrics: BCE/Brier; pairwise ranking loss; calibration curves; manual
  review of high-value recommendations.
- Gate: beats location-only and team-rate heuristics; no "run toward goal"
  collapse; robust enough that MCTS cannot exploit it.

### M4: MCTS And Post-Training

- MCTS state: recent positions, velocity, event context, score, clock, value
  state. Actions: short movement chunks (intent + chunk later). Priors: M2
  policy. Value: tactical value + imitation value + event continuation.
  Penalties: low likelihood, off-pitch, impossible speed, collision, spacing.
- Post-training: group-normalized candidate scoring with AWR/DPO/GRPO-style
  updates plus BC + KL anchor; re-evaluate against held-out SoccerNet and
  historical event snapshots.
- Metrics: value improvement over policy rollout, ADE/FDE degradation
  tolerance, KL from supervised policy, violation rate, search latency,
  reward-exploit rate from manual review.
- Gate: search and post-training improve value without degrading realism.

## Data Sources

All training runs on SoccerNet, which is already wired into this repository:
the `soccernet-tracking` Modal volume, downloaded via `SoccerNetDownloader` in
`modal_train.py`. Steps that need data SoccerNet's tracking task does not
carry pull it from other SoccerNet tasks through the same downloader onto the
same volume — no new external corpus is a prerequisite for any step.

- **M2:** SoccerNet tracking-2023 (already on the volume; the current training
  loop uses it).
- **M1/M3:** SoccerNet action-spotting labels — goals, shots, cards,
  substitutions, and the other labeled event classes with timestamps across
  hundreds of broadcast games. Default corpus for the event and value models.
- **M0:** goal timestamps, card events, and final scores derived from the same
  action-spotting labels give scoring hazards and outcome labels directly.
- **World Cup 2026 snapshots:** inference inputs and audit ground truth only,
  never training data before the audit closes. The one thing SoccerNet cannot
  provide is strength priors for international sides — keep that as a small
  manual ratings table inside the snapshot (`TeamState.strength` already
  exists), not a training dependency.

External corpora (e.g. StatsBomb Open Data with full event locations) are an
optional later extension if the SoccerNet event labels prove too coarse — not
a prerequisite.

## Data Validation That Matters

SoccerNet tracking (Step 3):

- No train/eval sequence overlap; no future frames or boxes in input.
- Track continuity through input and target windows.
- Bounding boxes in bounds with sane sizes.
- Motion buckets: stationary, walking, sprinting, turning, occluded.
- Pitch calibration confidence before pitch-space tactical claims.

Event data (Steps 2 and 5):

- Chronological order and clock consistency; score/card state matches event
  history; stable identifiers; one coordinate convention; licensing permits
  training and derived artifacts.

World Cup snapshots (Step 2):

- Snapshot saved before the simulated future; source and fetch time stored as
  epoch UTC; manual entries marked manual; missing fields explicit unknowns.

## What We Show Visually

Each item lands with its step, not at the end:

- One-sample debug view with real SoccerNet frames, selected player, target path,
  learned path, constant-position baseline, constant-velocity baseline, MCTS
  path, action logits, action labels, value outputs, loss terms, activation
  stats, and gradient norms (Step 1).
- Scoreline histogram, win/draw/loss panel, and next-goal fan chart (Step 2).
- Eval-scale baseline vs. learned vs. MCTS trajectory overlay on real frames
  (Step 3; the actual/pred/MCTS overlay already exists).
- Probabilistic future fan (Step 4).
- Possession/event rollout tree (Step 5).
- Pitch value heatmap (Step 6).
- MCTS root branch table with priors, visits, values, selected action;
  policy-vs-post-trained side-by-side (Step 7).
- Failure gallery: worst held-out samples, identity switches, calibration
  misses, reward exploits (grows continuously from Step 3).

## Decision Points

Revisit these explicitly, not silently:

- After Step 4: is the frozen backbone + head still beating baselines at full
  split scale? If not, consider LoRA on top Qwen layers — but only then.
- After Step 5: are event rollouts good enough to replace the Poisson goal
  layer inside `run_match`, or do they stay a diagnostic?
- After July 19: shift the World Cup surface from live prediction to replay
  backtesting and calibration reporting.

## Do Not Do Yet

- Do not tune the Qwen backbone before the frozen-feature baseline beats
  motion baselines (Step 3) at scale (Step 4).
- Do not claim optimal strategy from imitation alone.
- Do not run RL post-training without a calibrated value model (Step 6) and
  realism penalties (Step 7).
- Do not use current World Cup results or events after the snapshot time in a
  pre-registered prediction.
- Do not block the simulator on live tracking access; event/outcome layers
  first, player-level tracking attached where available.
- Do not build config/orchestration frameworks; the artifact contracts and CI
  from Step 1 are the coordination mechanism.
