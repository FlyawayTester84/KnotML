# ppo-small-cpu-kind

A small PPO agent that reduces 4-manifold Kirby diagrams (`Handlebody`) to the empty diagram,
using kind-first move sampling (`policy.factorization: kind`). It was trained on 2026-10-11 for
60,416 environment steps, about 6 minutes on a laptop CPU.

It uses the same settings and seed as [`ppo-small-cpu`](../ppo-small-cpu/README.md); only the
factorisation differs. Kind-first moved up the curriculum 2–3.5× faster, and both agents end within
noise of each other (see `planning/rl-design.md`).

| File | Contents |
|---|---|
| `config.yaml` | every hyperparameter of the run (same as the config inside the checkpoints) |
| `checkpoints/best.pt` | weights at step 15,360, the best evaluation (100% on 32 episodes at scramble 1–12) |
| `checkpoints/latest.pt` | weights at the end, step 60,416 |
| `metrics.csv` | training metrics per iteration (success, return, losses, curriculum level, ...) |
| `eval.csv` | greedy evaluations during training, every 10 iterations |

Each checkpoint also holds the optimizer state, curriculum level and step count, so training can
resume from it.

## Setup

- **Network:** `gnn`, hidden 64, 3 layers, kind-first head (489k parameters).
  - P(kind) is a softmax over the kinds with a legal move.
  - P(move | kind) is a softmax over that kind's moves.
  - The entropy bonus is balanced across kinds.
- **Algorithm:** PPO with 8 async envs, rollouts of 64 steps, 4 minibatches, 4 epochs, learning
  rate 3e-4 annealed to 0.
- **Environment:** 4-manifold calculus with capacity 32 crossings, 6 components, 40 steps per
  episode and 512 candidate actions.
- **Reward:** `potential`, i.e. +1 solved, −0.01 per step, and shaping toward fewer components
  (weight 0.1) and crossings (weight 0.01).
- **Curriculum:** scrambles of 1–3 construction moves to start, raised one at a time once 80% of
  100 episodes succeeded. It reached 1–12, the cap, at step 10,752.

## Results (greedy actions, fresh scrambles, 208 episodes each)

| Checkpoint | Scramble length | Success | Mean steps when solved |
|---|---|---|---|
| `best.pt` (15,360 steps) | 1–12 | 95.7% | 4.8 |
| `best.pt` | exactly 12 | 88.5% | 8.2 |
| `latest.pt` (60,416 steps) | 1–12 | 96.2% | 5.0 |
| `latest.pt` | exactly 12 | 89.9% | 8.1 |

## Use

```bash
python scripts/solve_link.py --agent ppo-small-cpu-kind --knotinfo "L2a1{1}" --framings dot 3
python scripts/evaluate_agent.py models/ppo-small-cpu-kind/checkpoints/best.pt --scramble 1 12 --episodes 200
python scripts/train_agent.py --config models/ppo-small-cpu-kind/config.yaml        # retrain into runs/
python scripts/train_agent.py --resume models/ppo-small-cpu-kind/checkpoints/latest.pt --set run.total_timesteps=200000
```

`--resume` continues training in this folder: it appends to the CSVs and overwrites `latest.pt`
and `best.pt`. Copy the folder first to keep this version.
