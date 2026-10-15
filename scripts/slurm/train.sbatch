#!/bin/bash
#SBATCH -J knotml-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 3-00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
# Train a KnotML agent on the FASRC Cannon cluster (see planning/remote-training.md).
#
#   mkdir -p logs && sbatch scripts/slurm/train.sbatch                    # partition gpu, run "kind-gpu"
#   RUN_NAME=kind-gpu-2 TOTAL_STEPS=50000000 sbatch scripts/slurm/train.sbatch
#   sbatch -p gpu_requeue scripts/slurm/train.sbatch                      # may be preempted, then requeued
#   EXTRA_SET="policy.hidden=256 algorithm.learning_rate=1e-4" sbatch scripts/slurm/train.sbatch
#   MAX_SCRAMBLE=64 MAX_CROSSINGS=128 MAX_COMPONENTS=16 MAX_STEPS=160 sbatch -c 16 --mem=64G scripts/slurm/train.sbatch
#
# Environment limits: a scramble never grows past env.max_crossings / env.max_components (moves that
# would are skipped), so the limits must grow with the curriculum's MAX_SCRAMBLE, or the longest
# scrambles are no harder than shorter ones. Measured: 64 crossings / 10 components are full by
# ~32 moves, 128 / 16 by ~64, 256 / 24 by ~128 (the defaults). MAX_STEPS bounds an episode.
#
# Submit from the repository root. The run lives in runs/$RUN_NAME/.
#  - If runs/$RUN_NAME/checkpoints/latest.pt exists, the job continues that run (after a time limit,
#    a requeue or a resubmission), with run.total_timesteps raised to TOTAL_STEPS.
#  - Otherwise it starts a new run. Use a new RUN_NAME for a fresh run, and one job per RUN_NAME.
#  - 180 s before the time limit Slurm sends SIGTERM; the trainer saves latest.pt and exits, so
#    resubmitting the same command continues the run. Preemption on gpu_requeue works the same way.
#
# Settings: the environment variables below (defaults after :-), plus `--set` overrides in EXTRA_SET.
# EXTRA_SET is applied when continuing too, so there it must not change the network (policy.*) or the
# env capacity (env.max_*), which the checkpoint's weights and buffers were built for. When continuing,
# the run keeps the limits it started with; MAX_* only apply to a new run.

set -euo pipefail
RUN_NAME=${RUN_NAME:-kind-gpu}
TOTAL_STEPS=${TOTAL_STEPS:-20000000}
VENV=${VENV:-.venv}          # the uv environment, relative to the repository root
PY_MODULE=${PY_MODULE:-}     # a module to load first (not needed for a uv environment)
EXTRA_SET=${EXTRA_SET:-}
MAX_SCRAMBLE=${MAX_SCRAMBLE:-128}       # curriculum.max_scramble: longest scramble the curriculum reaches
MAX_CROSSINGS=${MAX_CROSSINGS:-256}     # env.max_crossings
MAX_COMPONENTS=${MAX_COMPONENTS:-24}    # env.max_components
MAX_STEPS=${MAX_STEPS:-320}             # env.max_steps: moves per episode before truncation

cd "${SLURM_SUBMIT_DIR:-.}"
if [[ -n "$PY_MODULE" ]]; then module load "$PY_MODULE"; fi
source "$VENV/bin/activate"

CPUS=${SLURM_CPUS_PER_TASK:-32}
NUM_ENVS=$(( CPUS > 4 ? CPUS - 2 : 2 ))  # env processes; the rest is for the trainer itself
SETTINGS=(
  run.total_timesteps="$TOTAL_STEPS" run.torch_threads=2 run.checkpoint_every=10
  env.num_envs="$NUM_ENVS" algorithm.num_steps=512 algorithm.num_minibatches=8
  env.max_crossings="$MAX_CROSSINGS" env.max_components="$MAX_COMPONENTS" env.max_steps="$MAX_STEPS"
  curriculum.max_scramble="$MAX_SCRAMBLE" curriculum.window=1000 curriculum.promote_at=0.9
)
LATEST="runs/$RUN_NAME/checkpoints/latest.pt"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $CPUS CPUs, GPU: ${GPU:-none}, run runs/$RUN_NAME"

if [[ -f "$LATEST" ]]; then
  echo "continuing from $LATEST"
  # shellcheck disable=SC2086  # EXTRA_SET is a list of key=value words
  exec python scripts/train_agent.py --resume "$LATEST" --set run.total_timesteps="$TOTAL_STEPS" $EXTRA_SET
fi
if [[ -d "runs/$RUN_NAME" ]]; then  # an earlier attempt stopped before its first checkpoint: keep it aside
  mv "runs/$RUN_NAME" "runs/$RUN_NAME.incomplete-${SLURM_JOB_ID:-$$}"
fi
# shellcheck disable=SC2086
exec python scripts/train_agent.py --config configs/ppo.yaml --set run.name="$RUN_NAME" "${SETTINGS[@]}" $EXTRA_SET
