# Football RL Implementation Plan

This document turns the project thesis into an execution plan for engineers:
post-train a small open vision-language model on football tracking trajectories,
then use value learning, search, and RL-style post-training to simulate live match
states and recommend plausible player movement.

The end goal is concrete:

> Pick a current World Cup match state, run many simulations forward from that
> state, and show where the game is likely to go: next possessions, player
> movement, scoring chances, scoreline distribution, and tactical alternatives.

The near-term target is not "solve football strategy." The first defensible claim is:

> A small VLM plus structured trajectory/action/value heads can learn short-horizon
> player motion from SoccerNet tracking data, beat simple motion baselines, and
> produce visually inspectable candidate plays that can be searched with MCTS.

The long-term demo is a tactical match simulator: given the latest match clock,
score, lineups, event state, and any available frames/tracking, generate thousands
of rollouts and explain which simulated futures are most likely.

As of July 4, 2026, the FIFA World Cup is active and official FIFA pages list live
fixtures/results. Do not hardcode fixtures or teams into the model. The runtime
system should load the target match from an official or configured feed, persist
the exact snapshot timestamp, and produce pre-registered simulation artifacts.

## Current Starting Point

The repo already has enough scaffolding to make progress:

- `data/tracking_dataset.py`: SoccerNet Tracking v3 loader for MOT-style player
  boxes and frame windows.
- `models/qwen_football.py`: frozen Qwen2-VL feature extractor, relative-motion
  action codebook, policy head, and action-value head.
- `training/train_supervised.py`: deterministic train/eval sequence subsets,
  one-sample overfit mode, feature caching, supervised losses, and artifact export.
- `training/mcts.py`: PUCT search over discrete relative motion actions.
- `training/artifacts.py` and `web/index.html`: preview image and interactive
  rollout visualizer with actual, predicted, and MCTS paths.
- `data/match_state.py`: canonical JSON match-state schema for live/historical
  simulation snapshots.
- `simulation/run_match.py` and `simulation/monte_carlo.py`: first seeded Level 0
  scoreline/outcome simulator from a saved snapshot.
- `examples/world_cup_snapshot.example.json`: runnable example input for the
  simulator scaffold.
- `docs/training_plan.md`: minimal training goal and current Modal commands.

This plan assumes we keep that spine and add validation, stronger labels, pitch
geometry, better trajectory distributions, and more honest evaluation gates.

## Target Product: World Cup Match Simulator

The final product is a command and dashboard, not just a training notebook.

Example command shape:

```bash
python3 -m simulation.run_match \
  --competition fifa-world-cup-2026 \
  --match-id MATCH_ID_FROM_FEED \
  --snapshot live \
  --rollouts 10000 \
  --horizon-minutes 30 \
  --seed 7 \
  --out artifacts/world_cup/MATCH_ID/SNAPSHOT_EPOCH
```

Required outputs:

- `match_state.json`: exact input state, source feed, snapshot epoch timestamp,
  score, clock, lineups, cards, substitutions, possession/event state, and any
  detected player/ball positions.
- `simulation_summary.json`: win/draw/loss or advance probabilities, scoreline
  distribution, next-goal probability, next-shot probability, xG distribution,
  possession/territory projections, and top tactical branches.
- `rollout_samples.json`: representative simulated futures with event timelines,
  trajectory chunks, rewards, values, and terminal outcomes.
- `search_diagnostics.json`: MCTS root actions, priors, visits, values, selected
  branches, and realism penalties for player-level recommendations.
- `prediction_card.md`: human-readable timestamped prediction with caveats.
- `web` artifact payloads: frame overlay, pitch minimap, simulation fan chart,
  scoreline histogram, and branching possession tree.

The dashboard should answer:

- If the match is at minute X with score Y, what happens in 10,000 rollouts?
- Which team is most likely to score next?
- Which player movement or possession branch changes the outcome distribution?
- How sensitive are results to lineups, red cards, fatigue, and uncertainty?
- What did the simulator get wrong after the real match finishes?

## Simulation Fidelity Ladder

Build the simulator in levels so the project can show useful output before full
player-level simulation is solved.

### Level 0: Match Outcome Monte Carlo

Input:

- Teams, score, clock, venue, knockout/group context, lineups if available.
- Pre-match or in-match team strength priors.
- Current cards/substitutions/injuries when available.

Output:

- Scoreline distribution.
- Win/draw/loss or advance probabilities.
- Next-goal probability.

Model:

- Poisson or hazard-rate model with clock, score state, team strength, and cards.
- This is the first deployable World Cup simulation baseline.

Acceptance gate:

- Can backtest on historical matches from minute 0, halftime, and minute 75.
- Produces calibrated probability intervals, not only point predictions.

### Level 1: Event-Level Possession Simulator

Input:

- Level 0 state plus possession, field zone, recent event history, and team style.

Output:

- Sequences of passes, carries, turnovers, fouls, shots, and goals.
- xG/xT trajectory over simulated possessions.

Model:

- Event transition model trained on open event data where licensing allows.
- State includes score, time, zone, pressure proxy, team, and player role.

Acceptance gate:

- Backtests next event type, next shot, and next goal windows better than simple
  team-rate baselines.

### Level 2: Player-Trajectory Rollout

Input:

- Recent broadcast frames or extracted tracking state.
- Pitch geometry, selected players, ball position, and event context.

Output:

- Player movement chunks for the next 1-10 seconds.
- Multi-agent pitch-state rollouts.

Model:

- Current Qwen trajectory policy/value stack, upgraded to pitch coordinates and
  multi-agent context.

Acceptance gate:

- Beats motion baselines on held-out tracking sequences.
- Can initialize from an arbitrary current-match snapshot if tracking extraction
  is available.

### Level 3: Tactical Search And Strategy Branches

Input:

- Level 2 state plus value model and candidate actions.

Output:

- MCTS branches for a player or possession.
- Recommended movement/actions with expected value deltas.

Model:

- MCTS over action chunks using policy priors, value estimates, realism penalties,
  and event-level continuation values.

Acceptance gate:

- Search produces plausible alternatives and improves the value proxy without
  increasing physical-rule violations.

The World Cup demo can launch with Level 0 + Level 1, then progressively unlock
Level 2 and Level 3 for clips where video/tracking is available.

## System Architecture

The implementation should split into explicit packages:

```text
data/
  tracking_dataset.py          # existing SoccerNet Tracking loader
  validate_soccernet.py        # dataset integrity and geometry checks
  match_feeds.py               # fixture, lineup, score, event snapshot ingestion
  match_state.py               # canonical live/historical match-state schema
  calibration.py               # pitch homography and coordinate transforms

models/
  qwen_football.py             # existing trajectory/action/value model
  event_model.py               # event transition and scoring hazard models
  value_model.py               # possession value and chance creation

simulation/
  match_simulator.py           # game-clock/event/player rollout engine
  monte_carlo.py               # seeded repeated rollout orchestration
  run_match.py                 # CLI for current World Cup match simulations
  backtest.py                  # historical replay evaluation

training/
  train_supervised.py          # existing trajectory training
  train_event_model.py         # event-level training
  train_value_model.py         # tactical value training
  post_train_policy.py         # advantage/preference/RL-style updates

web/
  index.html                   # existing visualizer, extended for match sims
```

Canonical state objects should be plain JSON-serializable dataclasses. Every
artifact should include `source`, `fetch_time_epoch_utc`, `snapshot_epoch_utc`,
`competition`, `match_id`, `seed`, and model/checkpoint identifiers.

## World Cup Data Reality

SoccerNet trains and validates the movement model. It is not a live World Cup
feed. To simulate a current World Cup game, add a separate ingestion path.

Required live or near-live inputs:

- Fixture/match id and kickoff metadata from official FIFA pages or a configured
  provider.
- Score, match clock, half/period, stoppage time, and knockout state.
- Lineups, formations, cards, substitutions, and injuries when available.
- Event stream if licensed or manually entered: passes, shots, fouls, cards,
  possession, location, xG.
- Broadcast frames or legally available clips if player-level rollout is enabled.

Fallback modes:

- **Manual state entry**: engineer enters score, clock, lineups, and possession.
- **Event-only mode**: run Level 0/1 without tracking.
- **Clip mode**: run Level 2/3 on legally available short clips after extracting
  tracking from video.

Do not block the whole project on live tracking access. Start with a robust state
schema and event/outcome simulator, then attach player tracking where available.

## Research Translation From Robotics

Recent robotics VLA work is useful because football movement can be framed as
"vision-language-action" without robot hardware:

- Observations: broadcast frames, player boxes, field geometry, ball state, match
  context, and optional text prompts.
- Actions: short movement chunks or intent plus movement chunks.
- Demonstrations: real tracked player trajectories.
- Corrections: analyst or coach-preferred alternatives, if we later collect them.
- Rewards: realism, chance creation, possession value, shot probability change,
  and rule/constraint satisfaction.

The most useful robotics ideas to import:

- **Action chunking**: predict 0.4-1.0 second chunks instead of frame-by-frame
  jitter. The existing action stride is already a first version of this.
- **Frozen foundation model plus small action expert**: start with frozen VLM
  features and train small football heads before tuning the backbone.
- **Trajectory-level post-training**: optimize complete rollouts, not isolated
  coordinate tokens.
- **Value-conditioned behavior**: train a value model and condition or rank action
  candidates by advantage, while keeping a supervised behavior prior.
- **Search over structured actions**: use MCTS over legal movement chunks, with
  value and realism terms, not over text.

References that motivated this mapping:

- SoccerNet Tracking task: https://www.soccer-net.org/tasks/tracking
- SoccerNet data access: https://www.soccer-net.org/data
- Qwen2.5-VL release notes: https://qwenlm.github.io/blog/qwen2.5-vl/
- pi0 VLA paper: https://www.physicalintelligence.company/download/pi0.pdf
- RECAP / pi*0.6 post-training: https://www.pi.website/blog/pistar06
- Open X-Embodiment / RT-X: https://robotics-transformer-x.github.io/

## Success Criteria

### Demo Success

The World Cup simulator demo should show:

- Target match, source feed, snapshot timestamp, score, clock, lineups, cards,
  substitutions, and possession/event state.
- Number of Monte Carlo rollouts and seed.
- Scoreline distribution.
- Win/draw/loss or advance probabilities.
- Next-goal, next-shot, and xG distributions.
- Representative rollout timelines.
- Sensitivity to red cards, substitutions, and uncertain possession state.
- Post-match audit once the real future is known.

The player-level trajectory demo should show a specific player in a real
SoccerNet or current-match clip with:

- Input frame history and selected player identity.
- Actual future trajectory.
- Constant-position and constant-velocity baselines.
- Learned predicted trajectory distribution.
- MCTS alternative trajectory.
- A pitch-view minimap showing the same paths in field coordinates.
- A value overlay that makes the MCTS choice interpretable.

The viewer should be able to answer:

- What match state was simulated, and when was the snapshot captured?
- What did 10,000 rollouts predict about the next phase and final outcome?
- Which assumptions or uncertain inputs move the distribution the most?
- Did the model track the right player?
- Did the learned forecast beat simple extrapolation?
- Did the predicted movement look physically plausible?
- Did MCTS choose a different path for a tactical reason, not a visual artifact?
- Where does the model fail?

### Quantitative Success

Minimum publishable internal gates:

- One-sample overfit reaches near-zero train ADE/FDE and high action accuracy.
- Deterministic held-out train split is fixed by manifest and seed.
- Learned supervised model beats constant-position and constant-velocity baselines
  on held-out sequences by a meaningful margin.
- Pitch-normalized evaluation is available, not only image-normalized pixels.
- Probabilistic model reports calibrated NLL or CRPS, plus minADE@K.
- MCTS improves a value proxy without increasing realism violations or search
  latency beyond the interactive demo budget.
- Failure gallery is included in the blog and paper, not hidden.

Do not claim optimal football strategy until the reward model is independently
validated. "Better than observed future" is not automatically "better football";
the actual future is one sample from a partial-observation game, and real players
may be constrained by information not visible in the broadcast frame.

## Data Plan

### Primary Dataset

Use SoccerNet Tracking as the first real data source because it contains complete
single-camera soccer games with annotated player tracklets and frames. The current
loader handles the MOT-style frame, track id, and bounding-box format.

Required raw inputs:

- Sequence id and split.
- Frame images.
- `seqinfo.ini` for frame rate, image size, and frame count.
- `gameinfo.ini` for tracklet role and jersey metadata.
- `gt/gt.txt` for MOT boxes.
- Import timestamp as epoch seconds in UTC.

Required derived sample fields:

- `source`: `soccernet-tracking-2023`.
- `split`, `sequence`, `match_id` when available.
- `track_id`, role, team side, jersey when available.
- Input frame numbers and future frame numbers.
- Input timestamp and future timestamps, normalized to epoch seconds in UTC when
  source timing supports it.
- `fps`, `interval_seconds`, `input_frames`, `horizon`, `action_stride`.
- Selected player boxes in image coordinates.
- Selected player centers in image-normalized coordinates.
- Selected player centers in pitch coordinates once calibration is available.
- Ball and neighbor-player context once supported.
- Data import timestamp and code version.

### Secondary Sources

Evaluate these only after the SoccerNet baseline is reproducible:

- SoccerNet camera calibration for homography and pitch-coordinate validation.
- SoccerNet action spotting for event-aligned tactical labels.
- StatsBomb open data or 360 freeze frames for event/value-model pretraining.
- Metrica sample data for alternate tracking format sanity checks.

Secondary sources are useful for value modeling, but mixing them too early risks
licensing, schema, and split-comparison confusion.

## Data Validation

Add a deterministic validation command before scaling training:

```bash
python3 -m data.validate_soccernet --data-root /data/tracking-2023 --split train --seed 7
```

The command should write `data_validation_report.json` and fail on hard blockers.

### Dataset Integrity Checks

- Split exists as either extracted directories or split zip.
- Sequence list is stable and sorted before seeded sampling.
- Train/eval sequences do not overlap.
- Frame files exist for every sampled window.
- `seqLength`, image dimensions, and frame file counts agree within expected
  SoccerNet conventions.
- Track ids referenced in `gt.txt` exist in `gameinfo.ini`.
- Selected tracklets are players, not ball or referees.
- Bounding boxes have positive width/height and are mostly inside image bounds.
- Track continuity covers every input and target frame in a sample.
- Missing-frame and missing-box rates are reported per sequence and per track.
- Zip-backed and extracted-backed loaders return identical metadata for the same
  sample.

### Target And Leakage Checks

- The model never receives future frames or future boxes as input.
- Splits are by sequence or match, not by random frame.
- Near-duplicate windows are not split across train/eval.
- Horizon is reported in frames and seconds.
- Input frame ranges and target frame ranges are serialized in artifacts.
- All randomness uses explicit seeds.

### Geometry Checks

The current dataset uses image-normalized coordinates. That is useful for plumbing
but not enough for tactical claims. Add pitch geometry before value learning:

- Estimate or load homography per frame or segment.
- Reproject player footpoints or center-bottom box points onto the pitch.
- Normalize attacking direction so left-to-right and right-to-left sequences are
  comparable.
- Report reprojection error on pitch landmarks when labels exist.
- Reject or mark frames where calibration is uncertain.
- Validate speed and acceleration in meters/second and meters/second^2.
- Mark off-pitch projections and improbable teleports.

### Movement Plausibility Checks

For every sampled player trajectory:

- Current speed.
- Future displacement.
- Maximum step speed.
- Maximum acceleration.
- Track gap count.
- Direction change.
- Distance to nearest sideline in pitch coordinates.

Use these checks to find label errors and to stratify evaluation into stationary,
walking, sprinting, turning, and occluded cases.

## Baseline Ladder

Every learned model must be compared against simple baselines:

1. Constant position.
2. Constant velocity from the input window.
3. Constant acceleration, capped by plausible player acceleration.
4. Kalman-style smoother/extrapolator.
5. Nearest-neighbor trajectory retrieval by recent velocity and field location.
6. Existing learned head.
7. Probabilistic learned head.
8. Search-enhanced policy.

This ladder prevents spending time on RL before the model beats cheap physics and
retrieval.

## Metrics

### Forecasting Metrics

Report at 0.4s, 1.0s, and 2.0s when data supports those horizons:

- ADE: average displacement error.
- FDE: final displacement error.
- minADE@K and minFDE@K for multi-sample predictions.
- NLL or CRPS for probabilistic predictions.
- Action accuracy for discrete movement chunks.
- Top-k action accuracy.
- Calibration curves for predicted trajectory likelihood.
- Off-pitch rate.
- Teleport rate.
- Speed and acceleration violation rates.

Report metrics in both image-normalized space and pitch meters. Paper/blog claims
should prefer pitch-space metrics once calibration exists.

### Tactical Metrics

Add these after event/value labels are available:

- xT or possession-value delta.
- Shot probability delta within the next N seconds.
- Goal probability or scoring-sequence classifier AUC.
- Brier score and calibration for value predictions.
- Pass-lane or space-creation proxy accuracy when labels exist.
- Human preference win rate for recommended alternatives on curated clips.

### Search And RL Metrics

- MCTS search latency per sample.
- Number of simulations and expanded nodes.
- Root entropy and selected-action prior.
- Policy/value disagreement.
- Value improvement over policy rollout.
- Realism penalty of searched rollout.
- KL divergence from the supervised policy.
- Reward-model exploit rate from manual review.

### Data Quality Metrics

- Clips per sequence and per player.
- Track coverage and missing-box rate.
- Stationary vs moving sample balance.
- Camera calibration coverage.
- Outlier speed/acceleration rate.
- Train/eval distribution drift by sequence, team side, camera view, and motion.

## Implementation Phases

Two workstreams should run in parallel:

- **Simulator workstream**: build the canonical match-state schema, Level 0
  outcome Monte Carlo, Level 1 event simulator, backtests, and World Cup CLI.
- **Trajectory workstream**: improve SoccerNet tracking validation, pitch-space
  movement prediction, value learning, MCTS, and RL-style post-training.

This avoids the trap of waiting for full player tracking before we can simulate
an actual World Cup game.

### First 10 Engineering PRs

These are intentionally scoped so skilled engineers can pick them up without
waiting for research breakthroughs.

The current PR starts the first executable slice by adding the match-state schema,
manual snapshot input, and seeded Level 0 outcome simulator. The remaining items
turn that scaffold into a backtested, feed-driven, event-aware system.

1. **Canonical match state**
   - Add `data/match_state.py` dataclasses for team, player, lineup, score, clock,
     possession, event history, tracking snapshot, and feed metadata.
   - Add JSON round-trip tests.
   - Acceptance: a hand-authored World Cup snapshot serializes and loads without
     loss.

2. **Manual World Cup snapshot CLI**
   - Add `simulation/run_match.py --snapshot-file path.json --rollouts N`.
   - Start with a no-op simulator that validates state and writes artifact shells.
   - Acceptance: command writes the required output file names with metadata.

3. **Level 0 outcome simulator**
   - Add seeded Poisson/hazard-rate score simulation conditioned on score, clock,
     team priors, red cards, and home/neutral venue.
   - Add `simulation/monte_carlo.py`.
   - Acceptance: 10,000 rollouts complete locally in seconds and are deterministic
     for seed `7`.

4. **Historical backtest harness**
   - Add `simulation/backtest.py` for minute 0, halftime, and late-game snapshots
     from a small fixture/result fixture file.
   - Metrics: log loss, Brier score, calibration bins, scoreline error.
   - Acceptance: baseline backtest report is written as JSON.

5. **World Cup feed adapter interface**
   - Add `data/match_feeds.py` with an abstract feed client and a manual/static
     implementation. Keep provider credentials out of the repo.
   - Acceptance: `run_match.py --competition fifa-world-cup-2026 --match-id ...`
     can load from a local cached feed snapshot.

6. **Event-level simulator V0**
   - Add event states: possession, zone, next event, shot, turnover, goal.
   - Train or configure simple transition rates from available open event data or
     curated priors.
   - Acceptance: rollouts include event timelines and next-shot/next-goal metrics.

7. **SoccerNet validation CLI**
   - Add `data/validate_soccernet.py` with split, track, bbox, continuity, and
     metadata checks.
   - Acceptance: writes `data_validation_report.json` and deterministic sample
     manifest.

8. **Trajectory baseline table**
   - Add constant-position, constant-velocity, capped-acceleration, and retrieval
     baselines to the supervised training/eval artifacts.
   - Acceptance: every training run writes learned-vs-baseline metrics.

9. **Pitch minimap artifact**
   - Add pitch-coordinate schema and visualizer payload support. Use calibration
     when available and clearly mark image-space fallback.
   - Acceptance: dashboard can show a match-state pitch view and a trajectory
     rollout pitch view.

10. **MCTS strategy branch artifact**
    - Extend `training/mcts.py` artifact output with root branch diagnostics and
      make `web/index.html` display selected branches.
    - Acceptance: for one deterministic clip, policy and MCTS branches are visible
      and reproducible.

After these PRs, the project can already run a current World Cup match simulation
at Level 0/1 and show player-level trajectory demos on SoccerNet/available clips.
The later PRs connect those systems into a richer simulator.

### Working World Cup Demo Milestones

Milestone A: **Outcome simulator**

- Input: score, clock, lineups, team priors.
- Output: 10,000 scoreline rollouts and win/advance probabilities.
- Demo: dashboard scoreline histogram and probability timeline.
- Target: within 1 week of focused engineering.

Milestone B: **Event simulator**

- Input: outcome state plus possession, field zone, event history.
- Output: next-shot, next-goal, possession-chain, and xG rollout distributions.
- Demo: branching possession tree and next-10-minute event fan chart.
- Target: 1-2 additional weeks after Milestone A.

Milestone C: **Player movement overlay**

- Input: SoccerNet or current-match clip with extracted tracking.
- Output: likely movement paths for selected players plus baselines.
- Demo: broadcast overlay and pitch minimap.
- Target: after SoccerNet validation and pitch-coordinate work pass.

Milestone D: **MCTS tactical alternatives**

- Input: selected player/possession state and learned value model.
- Output: searched alternatives with value deltas and realism penalties.
- Demo: MCTS tree and recommended branch.
- Target: after the value model is calibrated.

Milestone E: **Pre-registered World Cup prediction**

- Input: current World Cup match snapshot saved before the future outcome.
- Output: timestamped prediction card and artifacts.
- Demo: post-match audit comparing predicted distributions to what happened.
- Target: once Level 0/1 is backtested and reproducible.

### Phase 0: Make Experiments Reproducible

Goal: every run can be reproduced and compared.

Engineering tasks:

- Add a validation CLI that writes `data_validation_report.json`.
- Extend `sequence_manifest.json` with code version, import timestamp, dataset
  root type, frame rate, coordinate space, and command-line arguments.
- Save baseline metrics into `training_history.json`, not only learned metrics.
- Add small deterministic tests for sequence subset selection, MOT parsing, and
  metric helpers.
- Keep the one-sample Modal overfit gate in README as the fastest sanity check.

Acceptance gate:

- A fresh run with seed `7` produces the same sequence manifest and the same first
  N sample metadata records.
- The validation command catches at least one synthetic bad bbox, missing track,
  or overlapping train/eval fixture.

### Phase 1: Strong Supervised Baseline

Goal: beat simple trajectory baselines on held-out SoccerNet sequences.

Engineering tasks:

- Add constant-position, constant-velocity, and capped-acceleration baselines.
- Train the current Qwen2-VL frozen-feature model on the deterministic 10%/2%
  sequence split.
- Cache visual features for repeatability and faster training loops.
- Report ADE/FDE by horizon bucket and motion bucket.
- Add a failure-gallery artifact with the worst held-out rollouts.
- Compare current action-codebook rollout to a direct continuous xy head.

Acceptance gate:

- One-sample overfit passes.
- Learned model beats constant velocity on moving samples, not only stationary
  averages.
- Worst-case visualizations are understandable enough to debug identity switches,
  bad boxes, or bad motion priors.

### Phase 2: Pitch Coordinates And Multi-Agent Context

Goal: make the task football-aware instead of image-plane-aware.

Engineering tasks:

- Add pitch-coordinate projection and calibration confidence.
- Use player footpoint or center-bottom box point rather than center of box when
  projecting to pitch coordinates.
- Normalize direction of play where metadata or event context allows it.
- Include nearby teammate/opponent/ball context as structured tensors.
- Add selected-player crops or masks to reduce ambiguity in crowded frames.
- Add team-side embeddings and track-role embeddings.

Acceptance gate:

- Pitch-space ADE/FDE is reported.
- The visualizer can switch between broadcast frame overlay and pitch minimap.
- Speed/acceleration sanity checks are measured in physical units.

### Phase 3: Probabilistic Trajectory And Action Chunk Model

Goal: represent multiple plausible futures.

Engineering tasks:

- Replace single deterministic xy output with one of:
  - Mixture density head over future xy chunks.
  - Discrete action chunks plus residual continuous offsets.
  - Lightweight diffusion or flow-matching head for 1-second chunks.
- Train with supervised behavior cloning plus likelihood loss.
- Sample K futures at eval time.
- Add minADE@K, NLL/CRPS, and calibration plots.
- Keep the existing relative-action codebook as a baseline, not a permanent
  assumption.

Acceptance gate:

- Multi-sample predictions improve minADE@K without destroying NLL calibration.
- Sampled futures look diverse but physically plausible.
- The model no longer averages two plausible runs into an implausible middle path.

### Phase 4: Value Model

Goal: score movement by tactical usefulness, not only imitation accuracy.

Value labels can start as weak proxies:

- Move toward open space.
- Maintain possession-likely spacing.
- Reduce pressure from nearby opponents.
- Increase xT or possession value when event data is aligned.
- Increase probability of future shot/chance in the next N seconds.

Engineering tasks:

- Define a value target schema separate from trajectory targets.
- Train a value head on trajectory rollouts and event-derived outcomes.
- Calibrate value predictions with reliability curves and Brier score.
- Add counterfactual negatives by perturbing plausible trajectories.
- Keep reward terms visible in artifacts so search decisions are inspectable.

Acceptance gate:

- Value predictions are better than location-only heuristics on held-out clips.
- High-value recommendations are not simply "run toward goal" in every context.
- Manual review of top/bottom value rollouts finds fewer obvious reward hacks
  over time.

### Phase 5: MCTS Planner

Goal: search over plausible short-horizon movement chunks.

MCTS state:

- Recent selected-player position and velocity.
- Frozen visual features or cached latent state.
- Nearby player and ball context.
- Remaining horizon.

MCTS actions:

- Discrete action chunk from the policy prior.
- Optional residual movement sampled from the probabilistic head.
- Optional intent such as hold, press, overlap, drop, receive, or attack space.

MCTS reward/value:

- Learned tactical value.
- Realism prior from supervised likelihood.
- Speed, acceleration, and off-pitch penalties.
- Collision and spacing penalties.
- Optional rule constraints once available.

Engineering tasks:

- Extend `training/mcts.py` to return full search diagnostics.
- Compare policy rollout vs MCTS rollout on the same fixed clips.
- Add search-time constraints before increasing simulations.
- Add latency measurements and cap demo-time search.
- Build a pitch-view tree overlay for root children and selected path.

Acceptance gate:

- MCTS changes actions only when value gain exceeds a configured margin.
- MCTS does not increase realism violations relative to policy rollouts.
- Search diagnostics make the chosen path explainable.

### Phase 6: RL-Style Post-Training

Goal: improve the policy using rollout-level rewards while staying close to the
demonstration distribution.

Start offline. Do not start with online RL, because the environment is a logged
football dataset with partial observability.

Training recipe:

1. Generate K candidate trajectories per state from the supervised policy.
2. Score each candidate with value, realism, constraints, and event proxies.
3. Normalize rewards within the candidate group.
4. Update candidate log probabilities with an advantage-weighted objective.
5. Add supervised behavior cloning loss on the observed trajectory.
6. Add KL or likelihood regularization against the supervised policy.
7. Evaluate on held-out sequences and curated visual cases.

This is close in spirit to robotics post-training with demonstrations, experience,
corrections, and value conditioning, but adapted to logged sports trajectories.

Possible objectives:

- Advantage-weighted regression.
- Direct preference optimization over candidate trajectory pairs.
- GRPO-style group-relative policy updates.
- Value-conditioned imitation, where the policy observes a desired advantage bin.

Acceptance gate:

- RL/post-training improves tactical value metrics without worse ADE/FDE beyond a
  defined tolerance.
- Reward-model exploit cases are tracked and reduced.
- The final policy remains physically plausible and close enough to real football
  trajectories for the demo to be credible.

### Phase 7: Visual Demo

Goal: make success and failure obvious without reading logs.

Demo views:

- Broadcast overlay: actual, learned, baseline, and MCTS paths.
- Pitch minimap: normalized field coordinates with nearby players and ball.
- Distribution view: sampled futures as translucent paths or density contours.
- MCTS tree view: root actions, priors, visits, values, and selected branch.
- Value surface: colored field heatmap for candidate player locations.
- Case selector: best, median, worst, high-motion, occluded, and manually curated.
- Timeline: frame scrubber with synchronized metrics.

Artifacts:

- `trajectory_rollouts.json`: all paths, metrics, and diagnostics.
- `trajectory_preview.png`: single static preview for README/blog.
- `training_history.json`: epoch metrics and baseline metrics.
- `data_validation_report.json`: data quality and split integrity.
- `failure_gallery.json`: worst clips and failure tags.

Acceptance gate:

- The demo can be opened by a reviewer and understood without extra context.
- At least one figure-quality clip shows supervised prediction improvement.
- At least one figure-quality clip shows MCTS choosing a plausible alternative.
- At least one failure case is included and explained.

### Phase 8: Paper And Blog

Paper claim ladder:

1. SoccerNet trajectory forecasting setup and validation.
2. Frozen small VLM plus football heads beats simple motion baselines.
3. Action-chunk modeling improves plausibility and multimodal forecasts.
4. A value head plus MCTS can generate interpretable alternative movement.
5. RL-style post-training improves value-conditioned rollouts under constraints.

Recommended paper figures:

- Dataset/task diagram.
- Model architecture: frames + context -> VLM -> latent -> policy/value heads.
- Baseline table: constant position, velocity, Kalman, retrieval, learned, MCTS.
- Scaling curve by number of clips or sequences.
- Pitch-space qualitative examples.
- MCTS tree/value visualization.
- Failure taxonomy.

Recommended blog structure:

- The motivation: treating football tracking like embodied trajectory learning.
- The first working baseline.
- What robotics post-training contributed.
- Visual demo with interactive examples.
- What worked, what failed, and what remains open.

### Phase 9: Current World Cup Simulation Runbook

Goal: run the simulator on an actual current World Cup match snapshot.

Runbook:

1. Select a match from the configured fixture feed.
2. Capture a snapshot before or during the match.
3. Persist the raw feed response or manual-entry source separately from derived
   artifacts.
4. Convert into canonical `match_state.json`.
5. Run Level 0/1 simulations with `--rollouts 10000 --seed 7`.
6. If legal frames/clips are available, extract tracking and run Level 2/3 on
   selected possessions.
7. Publish `prediction_card.md` with the exact snapshot timestamp and caveats.
8. After the match or future interval finishes, run the audit command and record
   calibration/error metrics.

Example audit command:

```bash
python3 -m simulation.backtest \
  --prediction artifacts/world_cup/MATCH_ID/SNAPSHOT_EPOCH/simulation_summary.json \
  --actual artifacts/world_cup/MATCH_ID/final_state.json
```

Acceptance gate:

- The simulation can be reproduced from saved inputs.
- The prediction was generated before the evaluated future window.
- The audit reports both what was right and what was wrong.

## Pitfalls

- **Image coordinates are not football coordinates.** Camera pans and zooms can
  make image-normalized motion look predictive while hiding poor pitch motion.
- **Track ids can switch or disappear.** Bad labels can look like model failures.
- **Random frame splits leak.** Split by sequence or match.
- **Stationary players dominate averages.** Always report moving-player buckets.
- **Actual future is not optimal play.** It is a demonstration, not a guaranteed
  tactical optimum.
- **Value labels can be weak.** A model can exploit a flawed reward by running
  toward goal, drifting off pitch, or ignoring teammates.
- **MCTS can amplify model errors.** Search over a weak learned dynamics/value
  model may produce confident nonsense.
- **Partial observability matters.** Broadcast frames may miss off-screen players,
  pressure, pass targets, and tactical instructions.
- **Backbone tuning too early will hide bugs.** Freeze features until data,
  geometry, and baselines are trustworthy.
- **Licensing and redistribution matter.** Keep dataset access separate from
  committed code and never commit credentials or downloaded data.

## Suggested PR Sequence

1. Canonical match-state schema and JSON tests.
2. Manual snapshot `run_match.py` CLI and artifact shell outputs.
3. Level 0 outcome Monte Carlo simulator.
4. Historical backtest harness and calibration metrics.
5. World Cup feed adapter interface with cached/static implementation.
6. Event-level possession simulator V0.
7. SoccerNet data validation CLI and tests.
8. Baseline metrics and training-history schema update.
9. Failure-gallery artifact generation.
10. Pitch-coordinate projection and minimap artifact schema.
11. Multi-agent context tensors.
12. Probabilistic trajectory head.
13. Value target schema and value-model training.
14. MCTS diagnostics and constraints.
15. RL-style post-training objective.
16. Current World Cup prediction runbook and dashboard polish.
17. Paper/blog figures and post-match audit report.

Each PR should include one runnable command, one artifact example, and one
acceptance gate. Avoid broad refactors while the experiment is still proving its
measurement stack.
