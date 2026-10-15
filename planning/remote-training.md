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
- **When the limits fill up** (handle slides multiply crossings, so it happens early). Mean
  crossings of fresh scrambles at each length:

  | Limits (crossings / components) | 16 moves | 32 | 64 | 128 |
  |---|---|---|---|---|
  | 32 / 6 | 31 | 32 (full from ~20) | 32 | 32 |
  | 64 / 10 | 44 | 64 (full) | 64 | 64 |
  | 128 / 16 | 53 | 104 | 128 (full) | 128 |
  | 256 / 24 | 54 | 131 | 240 | 256 (full) |

  So a curriculum up to 32 needs about 64 / 10, up to 64 needs 128 / 16, and up to 128 needs
  256 / 24.
- **The cost of bigger limits:**
  - **More legal moves:** an unfilled 256-crossing diagram can have ~15,000 legal moves to list
    each step, before 512 are chosen.
  - **Slower resets:** a 128-move start takes ~1 s to build (on a laptop), against 50 ms at 32
    moves.
  - **Slow promotion near the top:** each curriculum level needs `window` episodes at the
    promotion rate, so the last levels take longest.
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

## On FASRC Cannon (Slurm)

On Harvard's FASRC Cannon cluster, jobs go through Slurm
(see [Running jobs](https://docs.rc.fas.harvard.edu/kb/running-jobs/)). `scripts/slurm/` has two
batch scripts, run from the repository root:

| Script | Partition | What it does |
|---|---|---|
| `smoke.sbatch` | `gpu_test` (≤ 8 CPUs and 64 GB per MIG GPU, 2 jobs, 12 h) | checks CUDA, runs `tests/test_rl.py` and a 20k-step run (≈ minutes) |
| `train.sbatch` | `gpu` (A100, 3 days, not preempted); also works with `-p gpu_requeue` | 1 GPU, 32 CPUs (30 env processes), 128 GB, 3 days; curriculum to 128 with limits of 256 crossings, 24 components and 320 steps (section 4) |

**Set up once, with uv.** FASRC's own docs use mamba; a uv environment works just as well. It is
self-contained and brings its own Python, so no `python` module is needed. Follow FASRC's advice
on where things go:

- **Build on a compute node,** not a login node (login nodes also have no GPU, which the smoke test
  checks).
- **Keep the repository and its `.venv` on your lab's storage.** Home directories are slow for
  environments with many small files, such as PyTorch's.
- **Keep uv's cache off your home quota.** The CUDA PyTorch wheels alone are several GB.

```bash
salloc -p test -c 4 --mem=16G -t 0-02:00                      # a compute node for the install
curl -LsSf https://astral.sh/uv/install.sh | sh               # uv into ~/.local/bin (not pip --user)
# in ~/.bashrc, so every shell and job sees them (look up your lab's paths: ls /n/holylabs/LABS, ls /n/netscratch):
export UV_CACHE_DIR=<scratch or lab storage>/uv-cache
export UV_PYTHON_INSTALL_DIR=<lab storage>/uv-python
export UV_LINK_MODE=copy                                      # cache and environment sit on different filesystems
cd <lab storage> && git clone -b worker/runze git@github.com:FlyawayTester84/KnotML.git && cd KnotML
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124   # ships CUDA
uv pip install --python .venv/bin/python -e ".[rl,dev]"
mkdir -p logs                                                 # Slurm writes logs/<job name>_<job id>.out here
exit
```

The scripts read their defaults from these variables:

- `VENV` (`.venv`, relative to the repository root);
- `PY_MODULE` (none; set it only for an environment built on a module's Python);
- for `train.sbatch` also:
  - `RUN_NAME` (`kind-gpu`) and `TOTAL_STEPS` (20M);
  - `MAX_SCRAMBLE` (128), `MAX_CROSSINGS` (256), `MAX_COMPONENTS` (24) and `MAX_STEPS` (320), the
    limits of a new run. A continued run keeps the limits it started with;
  - `EXTRA_SET`, which holds extra `--set` overrides.

  A smaller setup with cheaper steps:
  `MAX_SCRAMBLE=64 MAX_CROSSINGS=128 MAX_COMPONENTS=16 MAX_STEPS=160 sbatch -c 16 --mem=64G scripts/slurm/train.sbatch`.

**Smoke test, then the run.**

```bash
sbatch scripts/slurm/smoke.sbatch                  # wait for "smoke test passed" in logs/knotml-smoke_<id>.out
RUN_NAME=kind-gpu-1 sbatch scripts/slurm/train.sbatch
sacct -u $USER                                     # state of your jobs
scontrol show job <id>                             # why it is pending, and its estimated start
tail -f logs/knotml-train_<id>.out                 # a line every 10 iterations, an evaluation every 25
jobstats <id>                                      # CPU/GPU/memory use: the 32 CPUs should be busy
scancel <id>                                       # stop it (the run saves latest.pt first)
```

**Continuing a run.**

- **Time limits:** 180 s before the limit, Slurm sends SIGTERM (`--signal=B:TERM@180`). The
  trainer saves `latest.pt` and exits.
- **Resubmitting:** the same command, with the same `RUN_NAME`, continues from `latest.pt`. To go
  further, raise the goal: `RUN_NAME=kind-gpu-1 TOTAL_STEPS=50000000 sbatch scripts/slurm/train.sbatch`.
- **A fresh run** needs a new `RUN_NAME`. Run one job per `RUN_NAME` at a time.

**Preemptible partition.**

- **Submitting:** `sbatch -p gpu_requeue scripts/slurm/train.sbatch` may start sooner, but the job
  can be preempted.
- **What happens on preemption:** the trainer saves on SIGTERM, Slurm requeues the job, and the
  requeued job finds `latest.pt` and continues.
- **What can be lost:** at most the steps since the last checkpoint. The script sets
  `run.checkpoint_every=10`, which is every 10 iterations, about 70k steps.

**Other notes.**

- Add `#SBATCH --account=<LAB>` (or `sbatch --account=<LAB> …`) to charge a specific lab's
  fairshare.
- Change resources on the command line, e.g. `sbatch -c 48 --mem=192G scripts/slurm/train.sbatch`.
  The number of env processes follows `-c`. `gpu` nodes have 64 cores and 4 GPUs, so asking for
  more than 16 cores per GPU may wait longer in the queue.

**Bring the agent back** from your laptop:

```bash
rsync -av --exclude 'step_*.pt' --exclude tb <you>@login.rc.fas.harvard.edu:KnotML/runs/kind-gpu-1/ models/kind-gpu-1/
```

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
- [ ] on Cannon: `logs/` exists, the smoke job passed, and `RUN_NAME` is new (or deliberately reused
      to continue a run)
