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

This plan is organized so the pieces can be built in parallel. Every work
package ends in a versioned artifact that can be tested, inspected, and shown in
the visualizer or a report. Nothing waits on a grand integration step; the
integration surface (`simulation.run_match` + artifact schemas) is itself an
early work package.

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

- **No tests and no CI.** Parallel work needs contract tests on the artifact
  schemas and fast CPU-only unit tests on the codebook, dataset windowing,
  match-state validation, and MCTS mechanics.
- **No motion baselines.** The M2 gate "beats constant velocity" cannot be
  evaluated today because constant position/velocity/Kalman baselines do not
  exist anywhere in the repo.
- **No event data and no M1.** M0 hazard rates are hard-coded heuristics with
  no backtest against historical matches.
- **No tactical value.** The existing value head learns *imitation advantage*
  (how much worse each codebook action is than the best action, measured
  against the ground-truth future). That is a search-guidance signal, not a
  tactical judgment. M3 is entirely open.
- **Integration gaps.** `run_match` only knows M0; `search_diagnostics.json`
  is a stub; no next-goal/next-shot outputs; no snapshot capture or post-match
  audit tooling; trajectory coordinates are image-normalized (camera motion is
  confounded with player motion) with no pitch calibration.

Known small fixes to fold into the relevant work packages rather than fix ad
hoc: M0 ignores `stoppage_minute`; horizon-truncated runs report "state at
horizon" as if it were the final result; per-sample PIL round-trips make
feature extraction slow at scale.

## Calendar Constraint

The 2026 World Cup final is July 19, 2026. Live, pre-registered predictions —
snapshot saved before the simulated window, audited after — are only possible
until then. That makes Workstream A (outcome/event simulation) plus snapshot
capture tooling the time-critical path. The SoccerNet trajectory track has no
calendar deadline; after July 19 the World Cup surface becomes replay-based
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
| M0 | Score/outcome model | Historical match results/snapshots | Future scoreline probabilities | v0 heuristic shipped; unfitted, unbacktested |
| M1 | Event transition model | Event logs and match state snapshots | Next event, time, zone, shot, goal | not started |
| M2 | Trajectory/world model | SoccerNet tracking frames and boxes | Player movement chunks | v0 shipped; overfit gate green; no baselines yet |
| M3 | Tactical value model | Events, trajectories, future outcomes | Chance/possession value | not started (imitation-advantage head exists, is not this) |
| M4 | Post-trained policy | Candidate rollouts scored by M3/MCTS | Better action priors | MCTS v0 shipped; post-training loop not started |

## Workstreams

Five workstreams. A, B, and C(event half) touch disjoint data and code and can
run fully in parallel once E and the schema half of D land. Every work package
(WP) lists its artifact, its gate, and its dependencies.

```text
E1 tests+CI ─┐
D1 schemas ──┼──> A1 ──> A2 ──> A3 ─────────┐
             ├──> B1 ──> B3 ──> B4          ├──> C3 ──> C4 ──> D3 (full integration)
             ├──> B2                        │
             ├──> C2 (event value, with A3) ┘
             └──> D2 snapshot capture ──> D4 audit          (live WC path: A1+A2+D2+D4)
```

### Workstream E: Engineering Foundation

**WP-E1 — Test suite and CI.**
Scope: pytest with tiny synthetic fixtures (no SoccerNet download, no GPU):
match-state validation, Monte Carlo determinism for seed 7, dataset windowing
against a fabricated MOT sequence, codebook labeling round-trip
(`action_labels` of a codebook-generated future recovers the actions), MCTS
runs end-to-end on a random-weight model. GitHub Actions running ruff +
pytest on push.
Artifacts: `tests/`, CI badge, green run.
Gate: suite runs under ~2 minutes on CPU.
Depends on: nothing. **Start immediately; unblocks safe parallel work.**

### Workstream D: Integration, Artifacts, Product Surface

**WP-D1 — Artifact schemas as contracts.**
Scope: write `docs/artifacts.md` (or JSON Schema files) versioning every
artifact: `match_state.json`, `simulation_summary.json`,
`rollout_samples.json`, `search_diagnostics.json`, `trajectory_rollouts.json`,
plus the new report artifacts below. Add contract tests (E1) that validate
emitted artifacts against the schemas. Add a `schema_version` field to each.
Artifacts: schema doc + contract tests.
Gate: existing Level 0 run and existing trajectory run both validate.
Depends on: nothing. **Start immediately; this is what makes parallel work
non-colliding.**

**WP-D2 — Live snapshot capture and pre-registration ledger.**
Scope: a small CLI to enter/save a live match snapshot (manual entry first,
feed adapter optional later) writing a valid `MatchState` with
`snapshot_epoch_utc`, plus an append-only `predictions_ledger.jsonl` recording
(snapshot path, artifact dir, git SHA, seed) at prediction time.
Artifacts: `simulation/capture_snapshot.py`, ledger file, at least one real
World Cup snapshot committed under `snapshots/`.
Gate: a real knockout-match snapshot captured and simulated before kickoff of
the simulated window.
Depends on: D1. **Time-critical: do during the tournament.**

**WP-D3 — `run_match` capability flags.**
Scope: `--models m0[,m1,m2,m3,m4]` with graceful degradation; each layer adds
fields to `simulation_summary.json`; absent layers produce explicit
`not_available` blocks (as `search_diagnostics.json` already does). Fix the
known M0 CLI issues here: honor `stoppage_minute`, and label horizon-truncated
output as `state_at_horizon` rather than a final result.
Artifacts: updated CLI + schema bump + contract tests.
Gate: same command runs with any subset of available checkpoints.
Depends on: D1; grows as A/B/C layers land.

**WP-D4 — Post-match audit.**
Scope: CLI that takes a ledger entry plus the observed final/horizon state and
writes `audit_report.json` (predicted probability of what happened, Brier
contribution, rank of observed scoreline) and appends to a running calibration
ledger across matches.
Artifacts: `simulation/audit.py`, `audit_report.json`, cumulative
`calibration_ledger.json`.
Gate: audits run for every ledger entry from D2.
Depends on: D2.

**WP-D5 — Visualizer growth.**
Scope: extend the existing web app incrementally as artifacts appear: match
outcome panel (scoreline histogram, win/draw/loss bar) reading
`simulation_summary.json`; next-goal fan chart (A2); baseline-vs-learned
trajectory overlay (B1); event rollout tree (A3); value heatmap (C2); failure
gallery (curated worst-N eval samples by FDE with playback links).
Artifacts: visualizer panels, each landing with its upstream WP.
Gate: every shipped model layer has at least one visual surface.
Depends on: the corresponding upstream WPs; each panel is independent.

### Workstream A: Outcome And Event Simulation (CPU, SoccerNet event labels)

**WP-A1 — M0 backtest and hazard fitting.**
Scope: extract goal times, cards, and final scores from the SoccerNet
action-spotting labels (same downloader and Modal volume as the tracking
data); fit scoring hazards conditioned on score state, time, and red cards;
backtest from minute 0, halftime, and minute 75 snapshots of held-out
SoccerNet games. Team strength stays a snapshot *input* (a small manual
ratings table for World Cup sides), not a quantity fitted here.
Artifacts: `training/fit_outcome.py`, fitted parameters in a checked-in config,
`backtest_report.json`, calibration plots surfaced in the visualizer or a
static report page.
Gate: beats the current hard-coded heuristic and a static pre-match
team-strength baseline on NLL and Brier across all three snapshot times;
deterministic artifacts for seed 7.
Depends on: E1, D1.

**WP-A2 — Time-resolved outcomes: next goal, next shot.**
Scope: make hazards time-varying inside the rollout so the simulator emits
next-goal time distributions and (once M1 exists, shot proxies before then)
next-shot estimates; add the fan-chart data to `simulation_summary.json`.
Artifacts: extended summary schema + fan chart panel (D5).
Gate: next-goal calibration checked in the A1 backtest harness.
Depends on: A1.

**WP-A3 — M1 event transition model v0.**
Scope: per the M1 spec below, trained on SoccerNet action-spotting event
sequences (goals, shots, cards, substitutions, and the other labeled classes
with timestamps). Start with a small MLP/sequence model over recent events;
keep it separate from Qwen so it stays debuggable. Wire into rollouts as
event-chain simulation between goals.
Artifacts: `training/train_events.py`, `event_model_report.json` with
calibration by game state and time bucket, event rollout tree JSON + viz panel.
Gate: beats team-rate and time-bucket baselines for next event type, next
shot, and next goal. Zone and possession targets activate only once event
labels are joined with tracking data — they are not in the action-spotting
labels.
Depends on: A1 (shares the backtest harness); parallel to all of B.

### Workstream B: Trajectory And World Model (GPU/Modal, SoccerNet)

**WP-B1 — Motion baselines in the eval loop.**
Scope: constant position, constant velocity, capped acceleration, Kalman-style
extrapolation, computed in the same eval pass as the learned model and written
into `training_history.json` and `trajectory_rollouts.json`; report ADE/FDE at
0.4s/1.0s/2.0s by motion bucket (stationary/walking/sprinting/turning), not
just aggregate. Visualizer overlays the baseline trace next to
actual/pred/MCTS.
Artifacts: `training/baselines.py`, extended eval report, baseline overlay
panel (D5).
Gate: this makes the standing M2 gate *evaluable*: held-out moving-player
samples must beat constant velocity before any backbone tuning.
Depends on: E1, D1. **Highest-priority trajectory work; without it no M2
claim is falsifiable.**

**WP-B2 — Tracking data validation CLI.**
Scope: `python -m data.validate_tracking` over a split: train/eval sequence
overlap check, future-frame leakage check, track continuity through
input+target windows, bbox bounds/size sanity, motion-bucket histogram.
Artifacts: `data_validation_report.json` per split.
Gate: 10%/2% split passes; violations listed, not silently dropped.
Depends on: E1. Parallel to B1.

**WP-B3 — Feature pipeline performance and scale run.**
Scope: batch visual feature extraction without the per-image PIL round-trip;
persistent feature cache on the Modal volume keyed by (sequence, frame,
image_size, model name); then scale training 10% -> 25% -> 100% of train
sequences, holding the seeded eval split fixed. Begin the data scaling curve
from docs/training_plan.md (nested subsets, multiple seeds at small sizes).
Artifacts: cache module, `scaling_report.json`, eval-metric-vs-data curve in
the report page.
Gate: eval ADE/FDE improves monotonically-ish with data, and beats baselines
(B1) at 100% of the current split; throughput numbers recorded.
Depends on: B1 (metrics), B2 (data trust).

**WP-B4 — Probabilistic futures.**
Scope: K-sample futures (mixture head or sampled decoding over the codebook)
with minADE@K/minFDE@K and NLL/CRPS; visualizer draws the sample fan.
Artifacts: extended rollout schema + fan rendering.
Gate: minADE@K beats the deterministic head; samples respect speed caps.
Depends on: B3.

### Workstream C: Value And Search (M3/M4)

**WP-C1 — Name the two values.**
Scope: small refactor + docs: the existing head is `imitation_value`
(advantage vs. ground-truth future, used by MCTS today); `tactical_value`
(M3) is a separate model. Rename fields in code and artifacts so no one
conflates them.
Artifacts: rename PR + schema bump.
Gate: gate is the naming itself; zero behavior change, tests stay green.
Depends on: E1, D1. Small; do early.

**WP-C2 — Tactical value v0 from event data.**
Scope: per the M3 spec below, structured-input model (no VLM features) trained
on event sequences: P(shot within T), P(goal within T), xT-style possession
value delta, turnover risk. P(shot) and P(goal) targets come straight from the
SoccerNet event labels; possession-value and turnover targets activate once
events are joined with the tracking clips. Parallel to all of B.
Artifacts: `training/train_value.py`, `value_model_report.json` with
calibration by score state/zone/time, pitch value heatmap panel (D5).
Gate: beats location-only and team-rate heuristics; does not collapse to "run
toward goal"; calibrated enough that search cannot exploit obvious bugs.
Depends on: A3's data pipeline (can share ingestion; model itself parallel).

**WP-C3 — Realism penalties and real search diagnostics.**
Scope: add penalty terms to trajectory MCTS (off-pitch, speed cap,
teleport/acceleration violations, low policy likelihood); emit a real
`search_diagnostics.json` at the trajectory level (root branch priors, visits,
values, selected path — the data already exists in `training/mcts.py`);
violation-rate metrics in eval.
Artifacts: penalized search + diagnostics artifact + root-branch table in the
visualizer.
Gate: MCTS matches or beats the raw policy on imitation value without
increasing physical-violation rate.
Depends on: B1 (violation metrics defined against baselines), C1.

**WP-C4 — M4 post-training loop.**
Scope: per the M4 spec below: sample K candidate futures from M2, score with
imitation value + tactical value (C2) + realism penalties, normalize rewards
within the group, update with advantage-weighted regression or a GRPO-style
objective, keep a behavior-cloning + KL anchor. Implemented directly in
PyTorch per docs/training_plan.md (no Trainer frameworks).
Artifacts: `training/post_train.py`, before/after eval report, side-by-side
rollout comparison in the visualizer.
Gate: value metrics improve, ADE/FDE degradation stays within a stated
tolerance, KL from the supervised policy bounded, violation rate flat, manual
review finds no reward exploits.
Depends on: B3/B4, C2, C3. This is deliberately last; everything else stays
useful even if this step is hard.

## Model Specs

The per-model specs below are the reference for the WPs above.

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
  in this space, which is exactly why the constant-velocity baseline (B1) and
  motion-bucket reporting are mandatory. Move to pitch coordinates (with
  calibration confidence) before making any tactical claims.
- Inputs: last 3-8 frames, selected player box, recent xy history; later ball
  state, nearby players, team side, event context, pitch calibration.
- Targets: next 1-2s of selected-player xy; codebook action labels; later
  multi-player and K-sample futures.
- Model: frozen Qwen2-VL features + trainable heads (current baseline). Keep
  the VLM frozen until B1/B2 gates pass. Add mixture/residual outputs after
  the discrete baseline.
- Loss/metrics: Smooth L1/MSE for xy; CE for actions; NLL/CRPS once
  probabilistic; ADE/FDE at 0.4/1.0/2.0s; minADE@K; speed/acceleration/
  off-pitch/teleport violation rates.
- Gates: overfit gate (done); held-out moving samples beat constant velocity;
  results reported by motion bucket.

### M3: Tactical Value Model

- Data: event sequences labeled by future shot/xG/goal/possession outcomes;
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
`modal_train.py`. Tracks that need data SoccerNet's tracking task does not
carry pull it from other SoccerNet tasks through the same downloader onto the
same volume — no new external corpus is a prerequisite for any WP.

- **M2:** SoccerNet tracking-2023 (already on the volume; the current training
  loop uses it).
- **M1/M3:** SoccerNet action-spotting labels — goals, shots, cards,
  substitutions, and the other labeled event classes with timestamps across
  hundreds of broadcast games. Default corpus for the event and value models.
- **M0:** goal timestamps, card events, and final scores derived from the same
  action-spotting labels give scoring hazards and outcome labels directly.
- **World Cup 2026 snapshots:** manual entry via WP-D2 first; automated feed
  adapters only if time permits. Snapshots are inference inputs and audit
  ground truth, never training data before the audit closes. The one thing
  SoccerNet cannot provide is strength priors for international sides — keep
  that as a small manual ratings table inside the snapshot (`TeamState.
  strength` already exists), not a training dependency.

External corpora (e.g. StatsBomb Open Data with full event locations) are an
optional later extension if the SoccerNet event labels prove too coarse — not
a prerequisite.

## Data Validation That Matters

SoccerNet tracking (WP-B2):

- No train/eval sequence overlap; no future frames or boxes in input.
- Track continuity through input and target windows.
- Bounding boxes in bounds with sane sizes.
- Motion buckets: stationary, walking, sprinting, turning, occluded.
- Pitch calibration confidence before pitch-space tactical claims.

Event data (WP-A3):

- Chronological order and clock consistency; score/card state matches event
  history; stable identifiers; one coordinate convention; licensing permits
  training and derived artifacts.

World Cup snapshots (WP-D2/D4):

- Snapshot saved before the simulated future; source and fetch time stored as
  epoch UTC; manual entries marked manual; missing fields explicit unknowns.

## What We Show Visually

Each item lands with its work package, not at the end:

- Scoreline histogram and win/draw/loss panel (A1/D5).
- Next-goal and next-shot fan chart (A2).
- Possession/event rollout tree (A3).
- Baseline vs. learned vs. MCTS trajectory overlay on real frames (B1; the
  actual/pred/MCTS overlay already exists).
- Probabilistic future fan (B4).
- Pitch value heatmap (C2).
- MCTS root branch table with priors, visits, values, selected action (C3).
- Failure gallery: worst held-out samples, identity switches, calibration
  misses, reward exploits (D5, grows continuously).

## Sequencing

Start immediately, in parallel (no interdependencies):

1. **WP-E1** tests + CI.
2. **WP-D1** artifact schemas + contract tests.
3. **WP-B1** motion baselines (the single most important trajectory-track WP).
4. **WP-A1** M0 backtest + hazard fitting.
5. **WP-D2** snapshot capture (time-boxed; tournament ends July 19).

Then, still largely parallel: A2, A3, B2, B3, C1, C2, D4, D5 panels.

Convergence points: C3 needs B1+C1; C4 needs B3/B4+C2+C3; D3 grows as each
model layer lands and is the final integration surface.

Decision points to revisit explicitly (not silently):

- After B3: is the frozen backbone + head still beating baselines at full
  split scale? If not, consider LoRA on top Qwen layers — but only then.
- After A3: are event rollouts good enough to replace the Poisson goal layer
  inside `run_match`, or do they stay a diagnostic?
- After July 19: shift Workstream A from live prediction to replay backtesting
  and calibration reporting.

## Do Not Do Yet

- Do not tune the Qwen backbone before the frozen-feature baseline beats
  motion baselines (B1) at scale (B3).
- Do not claim optimal strategy from imitation alone.
- Do not run RL post-training without a calibrated value model (C2) and
  realism penalties (C3).
- Do not use current World Cup results or events after the snapshot time in a
  pre-registered prediction.
- Do not block the simulator on live tracking access; event/outcome layers
  first, player-level tracking attached where available.
- Do not build a config/orchestration framework for the parallel WPs; the
  artifact contracts (D1) and CI (E1) are the coordination mechanism.
