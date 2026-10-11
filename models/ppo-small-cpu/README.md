# ppo-small-cpu

A small PPO agent that reduces 4-manifold Kirby diagrams (`Handlebody`) to the empty diagram.
It was trained on 2026-10-11 for 60,416 environment steps, about 6 minutes on a laptop CPU.

| File | Contents |
|---|---|
| `config.yaml` | every hyperparameter of the run (same as the config inside the checkpoints) |
| `checkpoints/best.pt` | weights at step 56,320, the best evaluation (100% on 32 episodes at scramble 1–12) |
| `checkpoints/latest.pt` | weights at the end, step 60,416 |
| `metrics.csv` | training metrics per iteration (success, return, losses, curriculum level, ...) |
| `eval.csv` | greedy evaluations during training, every 10 iterations |

Each checkpoint also holds the optimizer state, curriculum level and step count, so training can
resume from it.

## Setup

- **Network:** `gnn`, hidden 64, 3 layers (481k parameters).
- **Algorithm:** PPO with 8 async envs, rollouts of 64 steps, 4 minibatches, 4 epochs, learning
  rate 3e-4 annealed to 0.
- **Environment:** 4-manifold calculus with capacity 32 crossings, 6 components, 40 steps per
  episode and 512 candidate actions.
- **Reward:** `potential`, i.e. +1 solved, −0.01 per step, and shaping toward fewer components
  (weight 0.1) and crossings (weight 0.01).
- **Curriculum:** scrambles of 1–3 construction moves to start, raised one at a time to 1–12 once
  80% of 100 episodes succeeded.

## Results (greedy actions, fresh scrambles)

| Checkpoint | Scramble length | Success | Mean steps when solved |
|---|---|---|---|
| `best.pt` | 1–12 | 96.4% (112 episodes) | 4.5 |
| `latest.pt` | 1–12 | 97.6% (208 episodes) | 4.8 |
| `latest.pt` | exactly 12 | 85.7% (112 episodes) | 7.3 |

## Use

```bash
python scripts/evaluate_agent.py models/ppo-small-cpu/checkpoints/best.pt --scramble 1 12 --episodes 100
python scripts/evaluate_agent.py models/ppo-small-cpu/checkpoints/best.pt --html outputs/agent.html --record 3
python scripts/train_agent.py --config models/ppo-small-cpu/config.yaml        # retrain from scratch into runs/
python scripts/train_agent.py --resume models/ppo-small-cpu/checkpoints/latest.pt --set run.total_timesteps=200000
```

`--resume` continues training in this folder: it appends to the CSVs and overwrites `latest.pt`
and `best.pt`. Copy the folder first to keep this version.

In Python, `knotml.rl.load_policy("models/ppo-small-cpu/checkpoints/best.pt")` returns
`(policy, config)`.
