#!/bin/bash
#SBATCH --job-name=prep_sthelar
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
  echo "Usage: sbatch $0 <preprocessing_config.yaml>"
  exit 1
fi

CONFIG="$1"

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

echo "===== CONFIG HEAD ====="
head -n 120 "$CONFIG"
echo "======================="

python - <<PY
import yaml
from pathlib import Path

p = Path("$CONFIG")
print("YAML exists:", p.exists())
cfg = yaml.safe_load(open(p))
print("output_root:", cfg.get("output_root"))
print("label_mode:", cfg.get("label_mode"))
print("label_column:", cfg.get("label_column"))
print("ignore_labels:", cfg.get("ignore_labels"))
print("nuclei_types:", cfg.get("nuclei_types"))
print("n_slide_ids:", len(cfg.get("slide_ids", [])))
PY

echo "===== START PREPROCESSING ====="
python preprocessing/sthelar/convert_hf_to_cellvit.py --config "$CONFIG"
echo "===== PREPROCESSING DONE ====="

echo "===== OUTPUT CHECK ====="
OUT=$(python - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["output_root"])
PY
)

ls -lh "$OUT"
wc -l "$OUT"/cell_count_*.csv
echo "--- dataset_config.yaml ---"
cat "$OUT/dataset_config.yaml"
echo "--- split_manifest.yaml ---"
cat "$OUT/split_manifest.yaml"
echo "========================"