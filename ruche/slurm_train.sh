#!/bin/bash
#SBATCH --job-name=train_sthelar
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/workdir/taddeial/workspace/CellViT_for_STHELAR/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/taddeial/workspace/CellViT_for_STHELAR/logs/%x_%j.err

set -euo pipefail

REPO=/gpfs/workdir/taddeial/workspace/CellViT_for_STHELAR

if [ $# -lt 1 ]; then
  echo "Usage: sbatch $0 <training_config.yaml> [epochs_override]"
  exit 1
fi

CONFIG="$1"
EPOCHS_OVERRIDE="${2:-}"

module purge
module load miniconda3/25.5.1/none-none
source activate cellvit39

mkdir -p "$REPO/logs"

cd "$REPO"

echo "===== JOB INFO ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "pwd: $(pwd)"
echo "python: $(which python)"
python -V
echo "CONFIG=$CONFIG"
ls -lh "$CONFIG"

RUN_CONFIG="$CONFIG"

if [ -n "$EPOCHS_OVERRIDE" ]; then
  RUN_CONFIG="/scratch/${USER}-${SLURM_JOB_ID}/$(basename "$CONFIG" .yaml)_e${EPOCHS_OVERRIDE}.yaml"
  mkdir -p "$(dirname "$RUN_CONFIG")"

  python - <<PY
import yaml
from pathlib import Path

src = Path("$CONFIG")
dst = Path("$RUN_CONFIG")
epochs = int("$EPOCHS_OVERRIDE")

cfg = yaml.safe_load(open(src))
cfg["training"]["epochs"] = epochs

# Update logging names so the 1-epoch run does not overwrite the 3-epoch run.
if "logging" in cfg:
    old_comment = str(cfg["logging"].get("log_comment", "run"))
    cfg["logging"]["log_comment"] = old_comment + f"_override_e{epochs}"

    if "wandb_dir" in cfg["logging"]:
        cfg["logging"]["wandb_dir"] = str(cfg["logging"]["wandb_dir"]).rstrip("/") + f"_override_e{epochs}/wandb"
    if "log_dir" in cfg["logging"]:
        cfg["logging"]["log_dir"] = str(cfg["logging"]["log_dir"]).rstrip("/") + f"_override_e{epochs}/log"

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Created temporary config:", dst)
print("epochs:", cfg["training"]["epochs"])
print("log_comment:", cfg.get("logging", {}).get("log_comment"))
PY
fi

echo "===== CONFIG USED ====="
cat "$RUN_CONFIG"
echo "======================="

python - <<PY
import yaml
cfg = yaml.safe_load(open("$RUN_CONFIG"))
print("dataset_path:", cfg["data"]["dataset_path"])
print("train_folds:", cfg["data"]["train_folds"])
print("val_folds:", cfg["data"]["val_folds"])
print("test_folds:", cfg["data"]["test_folds"])
print("num_nuclei_classes:", cfg["data"]["num_nuclei_classes"])
print("epochs:", cfg["training"]["epochs"])
print("pretrained:", cfg["model"].get("pretrained"))
print("pretrained_encoder:", cfg["model"].get("pretrained_encoder"))
PY

echo "===== START TRAINING ====="
python cell_segmentation/run_cellvit.py --config "$RUN_CONFIG"
echo "===== TRAINING DONE ====="