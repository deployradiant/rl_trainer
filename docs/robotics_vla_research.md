# Robotics VLA Research Run For Football Policy Learning

Date: 2026-06-29

## Research Contract

Goal: identify recent robotics and vision-language-action lessons that should
shape this repo's Qwen football policy roadmap.

Baseline: current branch `codex/qwen-train-mode` trains Qwen2-VL's full visual
stack plus football policy/value heads on Modal H100. The latest overfit gate
used real SoccerNet sample `SNMOT-107`, track `9`, start frame `176`, with
`665,271,296` Qwen trainable visual parameters. The 200-epoch H100 overfit run
reached ADE `0.012159`, FDE `0.021636`, action accuracy `1.0`, value-action
accuracy `1.0`, and MCTS action accuracy `1.0`.

Metric for follow-up experiments: held-out ADE/FDE, action accuracy, MCTS
rollout ADE/FDE, value-action accuracy, wall time, and visual rollout quality
in the web player.

Scope: literature synthesis and architecture recommendations only. No new model
experiments were run in this note.

Stop rule: do not add RL complexity until a supervised action-chunk policy beats
the current relative-action baseline on fixed SoccerNet train/eval slices.

## High-Level Takeaway

The robotics lesson is not "make the LLM emit coordinates." The lesson is to
use the VLM as a perception, grounding, and language-conditioning backbone, then
attach an action expert that produces structured motion chunks. Modern robotics
systems increasingly separate:

```text
vision-language state understanding
        +
structured robot/player state tokens
        +
action expert over short future chunks
        +
value / critic / search layer
```

That maps cleanly to football:

```text
match frames + selected player + track history + task text
        |
Qwen visual/language tokens
        |
football state tokens
        |
action-chunk policy + value head + optional MCTS
```

The current repo already has the right early shape: Qwen visual features,
relative action labels, a value head, and MCTS over discrete movement chunks.
The next jump should be better action sequencing, not text generation.

## Source Summaries

| Source | Core Idea | Useful Lesson For This Repo |
| --- | --- | --- |
| [Physical Intelligence pi0](https://www.pi.website/blog/pi0), [paper](https://arxiv.org/abs/2410.24164) | A vision-language-action foundation model that uses Internet-scale VLM pretraining plus a robot action expert trained with flow matching. | Keep Qwen for visual-language grounding, but make the football motion policy a dedicated action expert. Flow/diffusion-style continuous action chunks are a natural successor to the current discrete codebook. |
| [Physical Intelligence pi0.5](https://www.pi.website/blog/pi05), [paper](https://arxiv.org/abs/2504.16054) | Co-trains heterogeneous robot data, vision-language data, web data, and action data to improve open-world robustness. | Our football model should co-train on tracking, phase labels, tactical captions, detection/visibility, and action chunks instead of only future xy. |
| [FAST action tokenization](https://www.pi.website/research/fast), [paper](https://arxiv.org/abs/2501.09747) | Compresses continuous robot action chunks into efficient tokens using frequency-domain structure and tokenization. | Replace or augment the hand-built heading/speed codebook with learned football motion tokens over 1-2 second chunks. A DCT or vector-quantized tokenizer over trajectories is a good near-term experiment. |
| [RT-1](https://arxiv.org/abs/2212.06817) | A transformer policy trained on large real-robot data with language instructions and action outputs. | Scale and data diversity matter. We need fixed sequence slices, role/phase stratification, and many clips before judging architecture. |
| [RT-2](https://arxiv.org/abs/2307.15818) | Turns robot actions into tokens so a VLM can transfer web knowledge into control. | Action tokens can work, but they should be compact motion tokens. Do not ask Qwen to free-form text coordinates. |
| [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) | Cross-robot datasets and models improve generalization when data is standardized. | Standardize football examples around `frames`, `player tokens`, `ball tokens`, `phase labels`, `future action chunk`, and metadata. |
| [OpenVLA](https://arxiv.org/abs/2406.09246) and [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | Open VLA models; OFT improves fine-tuning with continuous actions, action chunking, and efficient inference. | Our path should support continuous action chunks and direct PyTorch fine-tuning before trying full autoregressive motion tokens. |
| [Octo](https://arxiv.org/abs/2405.12213) | An open generalist robot policy with flexible observations and action heads. | Keep model inputs modular: image frames, track state, selected-player identity, ball state, and optional language should all be first-class tokens. |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Uses denoising diffusion for visuomotor action sequences and handles multi-modal behavior. | Football futures are multi-modal. A diffusion or flow action head is more appropriate than one deterministic xy path once the baseline works. |
| [Action Chunking Transformer / ALOHA](https://arxiv.org/abs/2304.13705) | Predicts chunks of future actions and executes them with temporal smoothing. | The next supervised policy should predict a chunk of relative future movement, not one point at a time. Receding-horizon rollout and temporal ensembling fit our visualizer well. |
| [DROID](https://arxiv.org/abs/2403.12945) | Large in-the-wild robot manipulation dataset with real diversity. | Data collection/curation is part of the model. Track metadata, camera quality, occlusions, player roles, and game phase so scaling runs are interpretable. |
| [RDT-1B](https://arxiv.org/abs/2410.07864) | A large robotics diffusion transformer for generalist control. | A transformer/diffusion action expert conditioned on Qwen and player state is a plausible medium-term architecture. |
| [NVIDIA GR00T N1](https://arxiv.org/abs/2503.14734) | Uses a dual-system design: a VLM for reasoning/grounding and a diffusion action model for control. | This is almost exactly the split we want: Qwen as System 2 state interpreter; action expert as System 1 movement generator. |
| [GR-2](https://arxiv.org/abs/2410.06158) and [UniPi](https://arxiv.org/abs/2302.00111) | Use video/world-model style prediction as part of policy learning. | Longer term, predict future player/ball state or visual rollouts as auxiliary supervision, not just selected-player xy. |
| [VideoMimic](https://arxiv.org/abs/2505.03729) | Extracts robot behavior from videos by recovering motion/action structure. | For broadcast football, tracking and pseudo-action extraction are the key bridge from video to policy. |
| [SmolVLA](https://arxiv.org/abs/2506.01844) | Compact VLA policy work focused on efficient open-source training/inference. | Keep the football action expert small and explicit. We do not need the whole Qwen language tower to generate every action. |

## What To Apply Here

### 1. Model Action Chunks, Not Points

The current codebook predicts relative heading and speed every `action_stride`
frames. That is already closer to robotics action chunks than raw xy
regression, but it is still coarse and hand-authored.

Next policy target:

```text
action_chunk = relative trajectory over 1-2 seconds
             = [dx_t, dy_t, dvx_t, dvy_t] or compressed chunk token
```

Good coordinate frames:

- Player-centric: relative to the selected player's current position and recent
  velocity.
- Team/pitch-centric: optionally include attacking direction and pitch zones.
- Camera/image-centric: useful only as an auxiliary target unless homography is
  available.

Why this matters: action chunks encode acceleration, curvature, hesitation, and
change of intent. A single heading/speed label hides those dynamics.

### 2. Add A Dedicated Action Expert

Do not make Qwen directly responsible for every movement coordinate. Use Qwen to
encode frames and language context, then feed its tokens plus structured player
state into an action expert.

Near-term action expert options:

1. ACT-style chunk transformer: predicts `K` future relative positions or
   velocities in one shot.
2. FAST-style motion tokenizer: compresses future chunks into discrete tokens,
   then trains a classifier over chunk tokens.
3. Diffusion/flow action head: samples multi-modal continuous future chunks
   conditioned on Qwen/state tokens.

Recommended order:

```text
ACT chunk head -> learned motion tokenizer -> diffusion/flow action head
```

This keeps the next step close to our current PyTorch code while opening the
door to modern VLA-style policies.

### 3. Treat Tracking As Proprioception

Robots have proprioception. Football players have track state.

For each visible entity, create state tokens:

```text
selected flag
team id / role if available
x, y
vx, vy
ax, ay
bbox cx, cy, w, h
visibility / confidence
distance to ball
distance to nearest teammate/opponent
```

The current selected-player bbox history is a good seed, but the model will not
understand football tactics until it sees other players and ball context.

### 4. Use Language As Conditioning And Explanation, Not The Motion Surface

Robotics VLA papers use language to specify goals and transfer semantic
knowledge. For us, language should condition the policy:

```text
"Predict the selected player's next second of movement."
"The selected player is attacking left to right."
"The player is a winger in possession."
"Forecast a realistic movement, not the optimal tactical move."
```

The model can later produce explanations:

```text
"The winger continues outside because the defender is closing the inside lane."
```

But the policy target should remain structured tensors or compact motion tokens.

### 5. Co-Train Semantic Football Tasks

Physical Intelligence's pi0.5 lesson is that heterogeneous data can improve
robustness. For football, add auxiliary labels even if they are weak:

```text
phase: possession / transition / set piece / defensive shape
role: goalkeeper / defender / midfielder / forward / unknown
intent: hold width / press / track runner / receive / carry / pass lane
visibility: occluded / visible / partial
motion type: stationary / accelerate / decelerate / curve / cut / sprint
```

These do not need to be perfect initially. They can be heuristics from tracking:
speed thresholds, ball proximity, local density, and movement curvature. The
language side of Qwen becomes useful when these labels become natural-language
state descriptions.

### 6. Make MCTS Search Chunks

Current MCTS searches over discrete heading/speed actions. That is good for a
first proof. The robotics-informed version should search over chunk candidates:

```text
root state = Qwen/frame state + tracking state
candidate action = 1s or 2s motion chunk
transition = update selected-player state, optionally update local context
value = realism + tactical value + consistency constraints
```

Candidate generators:

- Policy top-k motion tokens.
- Diffusion/flow samples.
- Retrieval from similar historical chunks.
- Simple baselines like constant velocity for calibration.

Value terms:

- Imitation likelihood or action log-probability.
- Trajectory realism.
- Off-pitch / collision / impossible acceleration penalties.
- Tactical proxy: space gained, pass availability, pressure reduction, shot or
  possession value if labels exist.

### 7. Use RL Only After Better Supervised Policy And Value

A GRPO-style update can fit this later:

```text
sample N candidate chunks or rollouts
score each candidate with value + realism + tracking reward
normalize rewards within the group
update log-probs for higher-relative-reward candidates
keep supervised and KL terms
```

But the immediate bottleneck is not RL. It is action representation and state
conditioning. RL on a weak action representation will reward artifacts.

## Recommended Architecture

```text
frames
  -> Qwen visual tokens

task text + optional football context text
  -> Qwen language tokens

tracking state
  -> player/ball state tokens

Qwen tokens + state tokens
  -> cross-attending football state encoder
  -> action expert
  -> value head
  -> MCTS / GRPO later
```

Minimal near-term version:

```text
Qwen visual tokens + selected-player track history
  -> chunk transformer head
  -> K-step relative trajectory chunk
  -> supervised loss + action classification loss + value loss
```

Better next version:

```text
Qwen visual tokens + all-player/ball tokens + task text
  -> motion-token policy or diffusion action head
  -> sample M candidate chunks
  -> MCTS/value rerank
```

## Concrete Next Experiments

| Priority | Experiment | Change | Success Signal |
| --- | --- | --- | --- |
| P0 | ACT-style chunk head | Predict the full 1s relative future chunk directly, while keeping current Qwen visual training. | Beats current relative-codebook ADE/FDE on fixed 5-sequence and 10%/2% slices. |
| P0 | Baseline comparison | Add constant-position and constant-velocity metrics to every run. | Learned policy beats both on held-out clips. |
| P1 | Motion chunk tokenizer | Build DCT or k-means/VQ tokenizer over SoccerNet future trajectory chunks. | Compact token vocab reconstructs chunks with low ADE/FDE; token labels are less noisy than heading/speed labels. |
| P1 | Multi-player state tokens | Add all visible player tracks and optional ball state as structured tokens. | Better held-out ADE/FDE, especially for direction changes and congested phases. |
| P1 | Temporal ensembling | Smooth overlapping chunk predictions in the visualizer and metrics. | Reduces jitter without hurting FDE. |
| P2 | Diffusion/flow action head | Sample multiple continuous future chunks conditioned on Qwen/state. | Better NLL/diversity and better best-of-K ADE than deterministic head. |
| P2 | Language conditioning | Add fixed task prompts and weak phase/role labels. | Improves performance on role/phase-stratified eval or explanation quality without hurting ADE. |
| P2 | Chunk-level MCTS | Search over learned chunk candidates instead of hand-coded heading/speed. | MCTS improves or matches greedy policy without unrealistic movement. |
| P3 | GRPO-style post-training | Sample groups of candidate futures, score with value/realism/tactical reward, update normalized advantages. | Improves value-weighted metrics while preserving imitation realism. |

## Implementation Notes For This Repo

Short-term code changes should stay simple:

- Keep direct PyTorch loops.
- Keep Modal H100 for full-Qwen runs, but use smaller fixed slices for research
  iteration.
- Keep `trajectory_rollouts.json` as the visual truth artifact.
- Add baselines before adding RL.
- Treat the current one-sample overfit as a contract test, not evidence of
  generalization.

Suggested next files:

```text
models/action_chunk.py
training/baselines.py
training/motion_tokenizer.py
training/eval_chunks.py
docs/research_log.md
```

Do not preserve backwards compatibility for the current hand-built codebook if a
chunk policy clearly wins. The project is still young enough to replace it.

## Biggest Risks

1. Broadcast coordinates are not pitch coordinates. If the target is image-space
   bbox centers, the model can learn camera behavior instead of football
   behavior. Homography or tracking-space coordinates matter.
2. Single-player prediction without ball/opponent context will plateau quickly.
3. Full-Qwen visual fine-tuning can overfit a sample, but data scale and
   baselines decide whether it generalizes.
4. Text generation is tempting for demos but should not become the control
   surface.
5. RL can exploit weak reward models. Keep imitation, KL, and physics/football
   constraints in the loop.

## Recommended Next Action

Implement the ACT-style chunk head and baseline metrics before another RL
iteration. It is the cleanest robotics-derived step:

See [action_chunk_policy_proposal.md](action_chunk_policy_proposal.md) for the
implementation proposal.

```text
input: last N frames + selected-player history
target: next 1s relative trajectory chunk
loss: SmoothL1 chunk loss + optional endpoint loss + value/action losses
eval: ADE/FDE vs current codebook, constant position, constant velocity
```

If that beats the codebook on the fixed 5-sequence slice, the next research run
should compare:

```text
hand codebook vs ACT chunk head vs motion-token classifier
```

Only after that comparison should we spend H100 time on diffusion/flow or GRPO.
