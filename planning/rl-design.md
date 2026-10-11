# Training agents on Kirby diagrams: design notes

*2026-10-11. `src/knotml/rl/`, configured by `configs/ppo.yaml`; run with
`scripts/train_agent.py` and `scripts/evaluate_agent.py`. Needs the `rl`
extra (PyTorch, PyYAML).*

## Problem

- **State:** a `Handlebody` (4-manifold calculus) or `FramedLink` (3-manifold).
- **Goal:** the empty diagram (`KirbyEnv` terminates there).
- **Starts:** `ScrambleStart` builds each start from the empty diagram by random construction moves, so
  every start is solvable and the scramble length is a difficulty knob.
- **Curriculum:** begins at 1–3 construction moves. It raises the maximum by one once 80% of the last 200
  episodes started at the current level succeed. The minimum stays, so short problems keep appearing.

## Components and how to swap them

| Module | Role | Extension point |
|---|---|---|
| `rl/config.py` | YAML → dataclasses; `--set a.b=v` overrides; unknown keys are errors | add fields to a section |
| `rl/envs.py` | `KirbyEnv` + `CandidateActions` + `ScrambleControl`, sync/async vector envs with same-step autoreset | another wrapper / action parameterisation |
| `rl/policies.py` | `Policy` base (masked categorical, `act`/`evaluate`/`value`) and `GraphPolicy` (`gnn`) | `@register_policy("name")` |
| `rl/algorithms/` | `Algorithm` base (`iteration()` → metrics) and `PPO` (`ppo`) | `@register_algorithm("name")` |
| `rl/rollout.py` | `Collector` (steps envs, tracks episodes, bootstraps truncations), `RolloutBuffer` + GAE | reuse for any on-policy method |
| `rl/curriculum.py` | scramble-length curriculum | |
| `envs/rewards.py` | `zero`, `sparse`, `potential`, `my_reward`, `example`, `module:function` | the `reward` section; edit `my_reward` in `envs/my_rewards.py` |
| `rl/trainer.py` | loop, CSV/TensorBoard logs, evaluation, checkpoints, resume | |
| `rl/evaluate.py` | success rate on fresh scrambles; recorded episodes as step pages | |
| `rl/solve.py` | a trained agent on a given diagram: greedy and sampled attempts, step limit, cycle stop | `scripts/solve_link.py` |

**Adding an algorithm.**

1. Subclass `Algorithm`.
2. Declare its hyperparameters as a nested `Hyperparameters` dataclass.
3. Implement `iteration(global_step) -> {"env_steps": n, ...}`; add `state_dict`/`load_state_dict` if it
   keeps optimizer state.
4. Decorate it with `@register_algorithm("name")` and import its module in `rl/algorithms/__init__.py`.

The trainer handles everything else: environments, curriculum, logging, evaluation and checkpoints.

- **On-policy methods** (A2C, PPG) reuse `Collector.collect` and `RolloutBuffer`.
- **Off-policy or search-based methods** step `collector.envs` themselves. `KirbyEnv.restore(state, steps)`
  lets tree search (MCTS) branch from a saved state.

## Actions: score the legal moves

The raw action space is `MultiDiscrete([kinds, anchors, anchors, 3])`.

- **Raw factorisation:** almost every combination is illegal, so a factorised policy would waste nearly
  all its probability mass.
- **Candidate scoring instead:** `CandidateActions` lists the legal moves, and the policy gives each one a
  logit. Padding is masked, so the agent never takes an illegal action.
- **Over `max_actions`:** all simplifying moves (removals, R3, cancellations) are kept and the rest are
  subsampled.
- **Typical counts:** about 60–300 legal moves for 4-manifold diagrams of 5–15 crossings.

## Network (`gnn`)

**Nodes.**
- **Ports** (`4 · max_crossings`): slot embedding plus crossing sign.
- **Component slots** (`max_components`): handle type, clipped framing embedding, signed-log framing,
  crossingless flag.

**One layer** (`layers` times, residual + LayerNorm).
- **Each port aggregates:**
  - itself;
  - the port across its edge (`nbr`);
  - the next and previous ports around its crossing, and the opposite port;
  - the mean over its face (`side_face`);
  - its component.
- **Each component aggregates:**
  - itself;
  - the mean of its ports;
  - the other components weighted by signed-log linking number, and by linked or not;
  - the global vector.

**Heads.**
- **Policy:** MLP(kind, embedding of anchor `a`, of anchor `b`, parameter, global). Anchors are port or
  component embeddings, zeroed for fields a kind does not read (`KirbyEnv.action_fields()`).
- **Value:** MLP(mean and max pool over ports and over components).

**Invariance.** No crossing or face index is ever a feature. `tests/test_rl.py` renumbers the crossings of
real observations and checks that values and every candidate's logit are unchanged (float64, 1e-9).

**Cost.** Dense padded gathers and `index_add_` scatters: O(B · (P + A) · H) per layer. No
torch-geometric; it runs on CUDA, MPS or CPU. The default (H = 128, 4 layers) has 2.4M parameters.

## PPO

Clipped surrogate and value loss, entropy bonus, GAE(λ), per-minibatch advantage normalisation, global
gradient clipping, linear learning-rate decay and early stopping on approximate KL (k3 estimator). Vector
envs autoreset in the same step (`AutoresetMode.SAME_STEP`). A time-limit truncation adds
`γ · V(final_obs)` to its reward, so cut episodes are not treated as failures.

## Reward (open; `potential` by default)

`potential` = `sparse` (+1 solved, −0.01 per step) + `γΦ(s') − Φ(s)` with
`Φ = −(0.1 · #components + 0.01 · #crossings)`. This is potential-based shaping, so the optimal policies
match those of the sparse reward. Φ(empty) = 0, and `gamma` defaults to the algorithm's.

## First measurement

60k steps on a laptop CPU in about 6 minutes: 8 async envs, H = 64, 3 layers, 4-manifold calculus,
defaults otherwise.

| Steps | Training success (recent episodes) | Scramble range | Greedy evaluation |
|---|---|---|---|
| 5k | 0.01 | 1–3 | 0.13 |
| 10k | 0.28 | 1–3 | 0.69 |
| 20k | 0.97 | 1–8 | 0.97 |
| 25k | 0.93 | 1–12 (cap) | 0.94 |
| 60k | 0.99 | 1–12 | 1.00 (32 episodes) |

On fresh scrambles with greedy actions:

| Checkpoint | Scramble length | Success | Mean steps when solved |
|---|---|---|---|
| `latest.pt` (60,416 steps) | 1–12 | 97.6% (208 episodes) | 4.8 |
| `latest.pt` | exactly 12 | 85.7% (112 episodes) | 7.3 |
| `best.pt` (56,320 steps) | 1–12 | 96.4% (112 episodes) | 4.5 |

`best.pt` is the checkpoint with the best evaluation score: the hardest level reached, then the success
there. Its 32-episode evaluation hit 100%; the larger test above puts the two checkpoints within noise
of each other. Evaluation runs the same number of episodes in every env, so it reports a multiple of
the env count (e.g. 112 for 100 requested).

Solutions are often shorter than the scramble. Pair creations followed by slides and kinks are
undone by removing the kinks and cancelling the pairs directly.

Throughput is 150–330 env steps/s on CPU, dominated by listing and encoding candidates in Python.
On MPS the default network ran at ~50 steps/s, so `device: auto` picks CUDA or the CPU, never MPS.

## Next

- **Faster candidates:** encode them with numpy, or cache them per state; this is the throughput limit.
- **Behaviour cloning:** a scramble's inverse moves are demonstrations, so pre-train the policy on them.
- **Harder starts:** known B⁴ and S³ diagrams beyond scrambles (Akbulut–Kirby, Gompf's examples), and
  starts that are not obviously solvable.
- **Search:** MCTS with the value network (AlphaZero-style) via `KirbyEnv.restore`.
