# Football Training And Simulation Plan

The goal is to simulate a current World Cup match from a saved match state, run
many futures, and show where the game is likely to go. SoccerNet is the training
ground for player movement. World Cup match snapshots are the inference target.

The work should stay centered on what we train:

1. A score/outcome simulator.
2. A football event model.
3. A player trajectory/world model.
4. A tactical value model.
5. A post-trained policy searched with MCTS.

The first PR already starts the executable path with `data/match_state.py`,
`simulation/monte_carlo.py`, `simulation/run_match.py`, and an example snapshot.
That Level 0 simulator is intentionally simple; it exists to establish the
artifact contract and Monte Carlo loop before the learned models arrive.

## End State

Input:

- Current match snapshot: score, clock, teams, lineups, cards, substitutions,
  possession/event state, and optional tracking/video-derived player positions.
- Model checkpoints for outcome, event, trajectory, value, and policy heads.
- Rollout count, horizon, and seed.

Command shape:

```bash
python3 -m simulation.run_match \
  --competition fifa-world-cup-2026 \
  --match-id MATCH_ID \
  --snapshot live \
  --rollouts 10000 \
  --horizon-minutes 30 \
  --seed 7 \
  --out artifacts/world_cup/MATCH_ID/SNAPSHOT_EPOCH
```

Output:

- `match_state.json`: exactly what was known at simulation time.
- `simulation_summary.json`: scoreline, win/draw/loss or advance probabilities,
  next goal, next shot, xG, and top branches.
- `rollout_samples.json`: representative simulated futures.
- `search_diagnostics.json`: MCTS branch priors, visits, values, and selected path.
- `prediction_card.md`: timestamped prediction plus limitations.
- Post-match audit comparing predicted distributions to the observed future.

## Training Stack

| Layer | What We Train | Main Data | Output |
| --- | --- | --- | --- |
| M0 | Score/outcome model | Historical match states/results | Future scoreline probabilities |
| M1 | Event transition model | Event logs and match state snapshots | Next event, time, zone, shot, goal |
| M2 | Trajectory/world model | SoccerNet tracking frames and boxes | Player movement chunks |
| M3 | Tactical value model | Events, trajectories, future outcomes | Chance/possession value |
| M4 | Post-trained policy | Candidate rollouts scored by M3/MCTS | Better action priors |

Train these in order. MCTS is useful only after M2 and M3 are credible.

## M0: Score And Outcome Model

Purpose:

- Give the project a runnable current-match simulator immediately.
- Provide the outcome backbone for full-match rollouts.

Current state:

- `simulation/monte_carlo.py` runs a seeded Poisson-style scoreline simulation
  from score, clock, team strength, and red cards.

What to train next:

- Fit scoring hazard rates from historical matches and event snapshots.
- Condition hazards on score state, time, red cards, team strength, and knockout
  context.

Loss and metrics:

- Negative log likelihood for final scoreline/outcome.
- Brier score and calibration for win/draw/loss and next-goal probabilities.
- Backtest from minute 0, halftime, and minute 75.

Gate:

- Beats a static pre-match team-strength baseline on historical snapshots.
- Produces deterministic artifacts for seed `7`.

## M1: Event Transition Model

Purpose:

- Simulate possessions, shots, turnovers, and goals instead of jumping straight
  from current score to final score.

Training data:

- Open or licensed event data with time, team, player, event type, location,
  possession, shot/xG, cards, substitutions, and score.
- Current World Cup feeds can be used for inference snapshots, but training should
  start on historical events.

Input:

- Match clock, score, team strengths, cards, substitutions.
- Possession team, field zone, previous N events, player role, and pressure proxy
  when available.

Targets:

- Next event type.
- Time to next event.
- Event location/zone.
- Shot probability and goal probability.
- Possession continuation or turnover.

Model:

- Start with a small tabular/MLP or sequence transformer over recent events.
- Keep it separate from Qwen at first; event modeling should be debuggable.

Loss and metrics:

- Cross entropy for event type.
- NLL or pinball loss for time-to-event.
- BCE/Brier for shot and goal.
- Calibration by game state and time bucket.

Gate:

- Beats simple team-rate and zone-rate baselines for next shot, next goal, and
  possession continuation.

## M2: Trajectory And World Model

Purpose:

- Learn how players move from frames and tracking context.
- Provide short-horizon dynamics for MCTS and player-level simulation.

Training data:

- SoccerNet Tracking frame windows and MOT-style player boxes.
- Start in image-normalized coordinates because the repo already supports it.
- Move to pitch coordinates before making tactical claims.

Input:

- Last 3-8 frames.
- Selected player box or identity.
- Recent player xy history.
- Later: ball state, nearby teammates/opponents, team side, event context, pitch
  calibration.

Targets:

- Next 1-2 seconds of selected-player xy movement.
- Action-chunk labels from the relative motion codebook.
- Later: multi-player movement and probabilistic K-sample futures.

Model:

- Current baseline: frozen Qwen2-VL features plus trainable football heads.
- Keep the VLM frozen until data validation and baselines are trustworthy.
- Add continuous residuals or mixture outputs after the discrete action baseline.

Loss and metrics:

- Smooth L1/MSE for xy.
- Cross entropy for action chunk labels.
- NLL/CRPS once probabilistic outputs exist.
- ADE/FDE at 0.4s, 1.0s, and 2.0s.
- minADE@K/minFDE@K for sampled futures.
- Speed, acceleration, off-pitch, and teleport violation rates.

Baselines:

- Constant position.
- Constant velocity.
- Capped acceleration.
- Kalman-style extrapolation.
- Nearest-neighbor trajectory retrieval.

Gate:

- One-sample overfit reaches near-zero train ADE/FDE.
- Held-out moving-player samples beat constant velocity.
- Results are reported by motion bucket, not only aggregate average.

## M3: Tactical Value Model

Purpose:

- Score candidate movement by tactical usefulness, not just imitation.
- Give MCTS and post-training a reward/value surface.

Training data:

- Event sequences labeled by future shot, xG, goal, possession continuation, or
  possession value.
- Trajectory rollouts aligned to event outcomes where available.
- Counterfactual negatives from perturbed but physically plausible trajectories.

Input:

- Match state, possession, field zone, player/ball positions, candidate action or
  trajectory chunk, and recent event context.

Targets:

- Probability of shot in next T seconds.
- Probability of goal in next T seconds.
- Possession value or xT delta.
- Turnover risk.
- Optional pairwise preference between two candidate actions.

Model:

- Start with an MLP/transformer over structured state.
- Add VLM features only after the structured value model is calibrated.

Loss and metrics:

- BCE/Brier for shot/goal/value buckets.
- Pairwise ranking loss for candidate comparisons.
- Calibration curves by score state, zone, and time bucket.
- Manual review of high-value recommendations.

Gate:

- Beats location-only and team-rate heuristics.
- Does not collapse to "run toward goal" for every state.
- Calibrated enough that MCTS does not exploit obvious reward bugs.

## M4: MCTS And RL-Style Post-Training

Purpose:

- Search plausible futures from the current state.
- Improve the policy using rollout-level rewards while staying near real football
  behavior.

MCTS:

- State: recent positions, velocity, event context, score, clock, and value state.
- Actions: short movement chunks or intent plus movement chunk.
- Prior: M2 policy.
- Value: M3 tactical value plus M1 event continuation.
- Penalties: low likelihood, off-pitch, impossible speed, collision, bad spacing.

Post-training loop:

1. Sample K candidate futures from M2.
2. Score candidates with M3, M1 continuation value, and realism penalties.
3. Normalize rewards within the candidate group.
4. Update action log probabilities with advantage-weighted regression, DPO, or a
   GRPO-style objective.
5. Keep a supervised behavior-cloning loss and KL/likelihood regularizer.
6. Re-evaluate against held-out SoccerNet trajectories and historical event
   snapshots.

Metrics:

- Value improvement over policy rollout.
- ADE/FDE degradation tolerance.
- KL from supervised policy.
- Realism violation rate.
- Search latency.
- Reward exploit rate from manual review.

Gate:

- MCTS improves value without increasing physical violations.
- Post-training improves value metrics while preserving credible movement.

## How A World Cup Simulation Runs

1. Capture or manually enter a current World Cup match snapshot.
2. Save the source, match id, and `snapshot_epoch_utc`.
3. If only score/clock/lineups are available, run M0.
4. If event state is available, run M0 + M1.
5. If legal frames or extracted tracking are available, initialize M2/M3/M4.
6. Run 10,000 seeded rollouts.
7. Save summary, representative futures, and prediction card.
8. After the future window or match finishes, run the audit and record calibration.

This means the project can show a real current-match simulator before full
player-level tracking is solved. Player-level recommendations become available
only for snapshots where tracking or video-derived state exists.

## Data Validation That Matters

SoccerNet tracking:

- No train/eval sequence overlap.
- No future frames or future boxes in input.
- Track continuity through input and target windows.
- Bounding boxes in bounds with sane size.
- Motion buckets: stationary, walking, sprinting, turning, occluded.
- Pitch calibration confidence before pitch-space tactical claims.

Event data:

- Chronological order and clock consistency.
- Score and card state match event history.
- Team/player identifiers are stable.
- Event locations use one coordinate convention.
- Licensing permits training and derived artifacts.

World Cup snapshots:

- Snapshot is saved before the simulated future.
- Source and fetch time are stored as epoch UTC.
- Manual entries are marked as manual.
- Missing lineups, cards, possession, or tracking are explicit unknowns.

## What We Show Visually

- Scoreline histogram and win/draw/loss probability timeline.
- Next-goal and next-shot fan chart.
- Possession/event rollout tree.
- Broadcast overlay: actual, baseline, learned, and MCTS trajectory.
- Pitch minimap for player movement and value.
- MCTS root branch table with priors, visits, values, and selected action.
- Failure gallery: identity switches, bad calibration, reward exploits, and
  unrealistic movement.

## Next PRs

1. Backtest M0 on historical snapshots and fit hazard parameters.
2. Add SoccerNet validation CLI and baseline trajectory metrics.
3. Add constant velocity/capped acceleration/retrieval baselines to training
   artifacts.
4. Train M1 event transition model with calibration reports.
5. Train M3 value model from event outcomes and trajectory rollouts.
6. Extend MCTS artifacts with branch diagnostics and realism penalties.
7. Add M4 post-training objective with supervised/KL regularization.
8. Wire M0/M1/M2/M3/M4 into `simulation.run_match` behind capability flags.

## Do Not Do Yet

- Do not tune the Qwen backbone before the frozen-feature baseline beats motion
  baselines.
- Do not claim optimal strategy from imitation alone.
- Do not run RL without a calibrated value model and realism penalties.
- Do not use current World Cup results or events after the snapshot time in a
  pre-registered prediction.
- Do not block the simulator on live tracking access; start event/outcome first,
  then attach player-level tracking where available.
