#!/bin/bash
#SBATCH -J knotml-smoke
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 0-00:30
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# A few minutes on gpu_test: the GPU is visible, the RL tests pass and a 20k-step run trains on CUDA.
#
#   mkdir -p logs && sbatch scripts/slurm/smoke.sbatch && tail -f logs/knotml-smoke_<JOBID>.out
#
# gpu_test allows at most 8 CPUs and 64 GB per (MIG) GPU and 2 jobs at a time.

set -euo pipefail
VENV=${VENV:-.venv}          # the uv environment, relative to the repository root
PY_MODULE=${PY_MODULE:-}     # a module to load first (not needed for a uv environment)
cd "${SLURM_SUBMIT_DIR:-.}"
if [[ -n "$PY_MODULE" ]]; then module load "$PY_MODULE"; fi
source "$VENV/bin/activate"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
python -m pytest tests/test_rl.py -q
python scripts/train_agent.py --set run.name="smoke-${SLURM_JOB_ID:-local}" run.total_timesteps=20000 \
  env.num_envs=4 run.output_dir=runs/smoke
echo "smoke test passed"
