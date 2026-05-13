#!/bin/bash
#SBATCH --job-name=train_sthelar
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit this script from the repository root, because Slurm log paths are relative.
set -euo pipefail

# Infer repository root from the location of this script (<repo>/ruche/slurm_train.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config if no argument is provided.
CONFIG="${1:-configs/training_sthelar.yaml}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG"
  exit 1
fi


# Environment.
# Override from command line if needed, e.g.
# sbatch --export=ALL,CONDA_ENV=myenv ruche/slurm_train.sh configs/training_sthelar.yaml
MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs
module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "Repo:   $REPO"
echo "Config: $CONFIG"
echo "Python: $(which python)"
python -V
echo "===================="

echo "===== START TRAINING ====="
python cell_segmentation/run_cellvit.py --config "$CONFIG"
echo "===== TRAINING DONE ====="