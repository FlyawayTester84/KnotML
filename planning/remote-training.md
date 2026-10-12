# Training on a remote GPU server

*2026-10-11. A checklist for long runs of `scripts/train_agent.py`. The code needs no changes; only
the configuration does, through `--set` or a copy of `configs/ppo.yaml`. Training needs only the
`rl` extra: SnapPy and Regina are for drawing pages and verification, not training.*

## 1. Set up the server (once)

```bash
git clone git@github.com:FlyawayTester84/KnotML.git && cd KnotML
python3 -m venv venv && source venv/bin/activate                          # needs Python >= 3.10
nvidia-smi                                                                 # note the CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu124      # use the command for your CUDA version from pytorch.org
pip install -e ".[rl,dev]" tensorboard
python -c "import torch; print(torch.cuda.is_available())"                # must print True
python -m pytest tests/test_rl.py -q                                        # quick check
```

## 2. Smoke test (about a minute)

```bash
python scripts/train_agent.py --set run.name=smoke run.total_timesteps=20000
```

The first line printed should say `device cuda`. `run.device: auto` picks the GPU by itself. To use
another GPU, set `run.device=cuda:1`, or `CUDA_VISIBLE_DEVICES=1`.

## 3. Settings to change for a long run

| Setting | Default | On the server | Why |
|---|---|---|---|
| `run.name` | `kirby-ppo` | e.g. `kind-gpu-1` | the run is saved in `runs/<name>/` |
| `run.total_timesteps` | 2,000,000 | 10–50M | the laptop agents used 60k |
| `env.num_envs` | 16 | about `nproc` − 2 | env stepping (pure Python, one process per env) limits speed, not the GPU |
| `algorithm.num_steps` | 128 | so that `num_steps × num_envs` ≈ 4k–16k | the batch per update |
| `algorithm.num_minibatches` | 8 | about batch ÷ 1024 | minibatches of ~1k samples suit a GPU |
| `env.max_crossings` | 32 | 64 | room for longer scrambles (see section 4) |
| `env.max_components` | 6 | 8–10 | each 1/2 pair adds two components |
| `env.max_steps` | 40 | 80 | about 2–3 × the expected solution length |
| `run.torch_threads` | 0 (all cores) | 4 | stops PyTorch competing with the env processes for CPU |
| `run.compile` | false | true (optional) | `torch.compile` on CUDA; the first iterations are slower |

The policy (`gnn`, hidden 128, 4 layers, `factorization: kind`), the reward (`potential`, or
`my_reward` from `src/knotml/envs/my_rewards.py`) and the PPO hyperparameters can stay as they are.

## 4. Curriculum

The curriculum runs on its own, but three settings should change for a long run:

| Setting | Default | On the server | Why |
|---|---|---|---|
| `curriculum.max_scramble` | 12 | 25–40 | the laptop runs reached 12 within ~11k steps; past the cap, extra compute repeats the same difficulty |
| `curriculum.window` | 200 | ~1000 | with 30 envs × 256 steps and short episodes, an iteration finishes hundreds to over a thousand episodes; a bigger window gives a steadier success estimate before each promotion |
| `curriculum.promote_at` | 0.8 | 0.85–0.9 | see the caveat below |

Keep `enabled: true`, `step: 1` and the starting range `env.scramble: [1, 3]`.

**Raise the capacity with `max_scramble`.**
- **Crossings and components:** a scramble never exceeds `env.max_crossings` or
  `env.max_components`; moves that do not fit are skipped. Small limits therefore cap the difficulty
  whatever `max_scramble` is. For example, 6 components allow only 3 handle pairs.
- **Episode length:** keep `env.max_steps` at about 2–3 × the expected solution length. Scrambles
  of 12 moves were solved in about 8.

**Caveat: the minimum stays at 1.**
- **What happens:** the scramble range grows as 1–4, 1–5, …, 1–30, with lengths drawn uniformly.
  Promotion uses the mean success over all episodes started at the current level, and most of
  them are much shorter than the maximum.
- **Consequence:** the agent can be promoted while still weak on the longest scrambles, because the
  easy ones lift the average. A higher `promote_at` reduces this; it does not remove it.
- **Fixes not yet implemented:** promoting on the success of episodes whose scramble was near the
  maximum, or a sliding range `[hi − width + 1, hi]` with some short scrambles mixed in.

**Evaluation.**
- **Default:** `eval.scramble: null` evaluates on the current training range. It follows the
  curriculum, but the eval curve is not comparable over time.
- **Fixed benchmark:** for a comparable curve, set e.g. `eval.scramble: [10, 10]`. This also makes
  `best.pt` the best success at that fixed length.

## 5. Start the run, detached from SSH

```bash
tmux new -s train          # detach: Ctrl-b d; come back: tmux attach -t train
python scripts/train_agent.py --set run.name=kind-gpu-1 run.total_timesteps=20000000 \
  env.num_envs=30 algorithm.num_steps=256 algorithm.num_minibatches=8 \
  env.max_crossings=64 env.max_components=10 env.max_steps=80 \
  curriculum.max_scramble=30 curriculum.window=1000 curriculum.promote_at=0.9 \
  run.torch_threads=4 2>&1 | tee train.log
```

- **Other ways to run it:** `nohup … > train.log 2>&1 &` also works. On SLURM, put the same command
  in an `sbatch` script that requests 1 GPU and about 32 CPUs.
- **Quoting:** list values in `--set` need quotes in the shell, e.g. `'eval.scramble=[10, 10]'`.
- **Keeping the settings in a file:** copy `configs/ppo.yaml` to `configs/ppo_gpu.yaml`, edit the
  values from sections 3–4, and run `python scripts/train_agent.py --config configs/ppo_gpu.yaml`.
  The run's full configuration is saved in `runs/<name>/config.yaml` either way.

## 6. While it runs

- **Progress:** in a terminal (tmux/screen), a progress bar shows steps, speed, time left and the
  current scramble range, success and return. A log line follows every `run.log_every` (10)
  iterations, and an evaluation every `eval.every` (25). With `| tee train.log` or `nohup` the bar
  switches itself off and only the log lines are written, so `tail -f train.log` stays readable.
  `runs/<name>/metrics.csv` records every iteration, and `eval.csv` every evaluation.
- **TensorBoard:** run `tensorboard --logdir runs --port 6006` on the server. On your laptop, run
  `ssh -L 6006:localhost:6006 you@server` and open `http://localhost:6006`.
- **What to watch:**
  - `scramble_hi`, the curriculum level, should keep rising;
  - `success`;
  - `approx_kl`, which should stay below `target_kl` = 0.03;
  - `explained_variance`, which should be positive and rising.
- **If the run stops:** resume with
  `python scripts/train_agent.py --resume runs/kind-gpu-1/checkpoints/latest.pt`. Add `--set
  run.total_timesteps=…` to extend it.
- **Disk:** checkpoints are kept as `latest.pt`, `best.pt` and the last `run.keep_checkpoints` (3)
  periodic `step_<n>.pt`. With hidden 128 and 4 layers each checkpoint is about 30 MB, weights plus
  optimizer state.

## 7. Bring the agent back

```bash
rsync -av --exclude 'step_*.pt' --exclude tb you@server:KnotML/runs/kind-gpu-1/ models/kind-gpu-1/
python scripts/evaluate_agent.py models/kind-gpu-1/checkpoints/best.pt --scramble 1 30 --episodes 500
python scripts/solve_link.py --agent kind-gpu-1 --knotinfo "L2a1{1}" --framings dot 3
```

`models/<name>/` follows the layout of `models/ppo-small-cpu-kind/`. A short README there should
record the setup and results.

## Checklist

- [ ] `torch.cuda.is_available()` is True; `tests/test_rl.py` passes
- [ ] the smoke run prints `device cuda`
- [ ] `run.name` is new (an existing name gets a `-2` suffix rather than being overwritten)
- [ ] `env.num_envs` matches the CPU cores; `run.torch_threads` is small
- [ ] `curriculum.max_scramble` is raised, with `env.max_crossings`, `env.max_components` and
      `env.max_steps` raised to match
- [ ] `curriculum.window` and `curriculum.promote_at` suit the episode throughput
- [ ] the reward is the intended one (`reward.name`)
- [ ] the run is inside tmux/screen/nohup/sbatch, logging to a file
