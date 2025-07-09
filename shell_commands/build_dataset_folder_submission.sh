#!/usr/bin/env bash

# --------------------------------------------------------------------------
# Create a BioStudies-friendly folder hierarchy made of symbolic links only.
# --------------------------------------------------------------------------
set -euo pipefail

# ==== 1. CONFIGURE ROOT PATHS ===============================================
STH="/Volumes/DD1_FGS/MICS/data_HE2CellType/STHELAR"     # destination root
SRC_BASE="/Volumes/DD1_FGS/MICS/data_HE2CellType"        # shorten typing

# helper → mkdir + link, with sanity check
link() {
    local tgt_dir="$1" tgt_name="$2" src="$3"
    mkdir -p "$tgt_dir"
    if [[ -e "$src" ]]; then
        ln -sfn "$src" "$tgt_dir/$tgt_name"
    else
        echo "WARNING – source not found: $src" >&2
    fi
}

# ==== 2. sdata_slides =======================================================
# echo "Step 1  – sdata_slides"
# link "$STH" "sdata_slides" \
#      "$SRC_BASE/CT_DS/sdata_final"
# Commented to use instead zip_zarr_final_dataset.sh to get zipped zarr instead

# ==== 3. data_40x/data  (single files) ======================================
echo "Step 2  – data_40x/data core files"
DATA40="$STH/data_40x/data"
declare -a core40=(
    "HE2CT/prepared_datasets_cat/images.zip"
    "HE2CT/prepared_datasets_cat/masks_cell_ids_nuclei.zip"
    "HE2CT/prepared_datasets_cat/ct_1/ALL/labels.zip"
    "HE2CT/prepared_datasets_cat/ct_1/ALL/cell_count.csv"
    "HE2CT/prepared_datasets_cat/ct_1/ALL/patch_metrics.csv"
    "HE2CT/prepared_datasets_cat/ct_1/ALL/types.csv"
    "CT_DS/annots/annot_dicts_ct_1"
)
for rel in "${core40[@]}"; do
    base=$(basename "$rel")
    link "$DATA40" "$base" "$SRC_BASE/$rel"
done

# ==== 4. data_40x/data/masks_slides  (per-slide mask.npz) ====================
echo "Step 3  – masks_slides"
MSRC="$SRC_BASE/CT_DS/ds_slides_cat/ct_1"
MDST="$DATA40/masks_slides"
for slide in "$MSRC"/*; do
    [[ -d "$slide" ]] || continue
    sid=$(basename "$slide")
    link "$MDST" "masks_${sid}.npz" "$slide/masks.npz"
done

# ==== 5. fine-tuning – detailed =============================================
echo "Step 4  – finetuning_CellViT_detailed"
FTD="$STH/data_40x/finetuning_CellViT_detailed"
declare -a ftd_files=(
    "HE2CT/training_datasets/ds_1/cell_count_test.csv"
    "HE2CT/training_datasets/ds_1/cell_count_train.csv"
    "HE2CT/training_datasets/ds_1/cell_count_valid.csv"
    "HE2CT/training_datasets/ds_1/dataset_config.yaml"
    "HE2CT/training_datasets/ds_1/weight_config.yaml"
)
for rel in "${ftd_files[@]}"; do
    link "$FTD" "$(basename "$rel")" "$SRC_BASE/$rel"
done

# ---- 5a. detailed → output logs --------------------------------------------
echo "Step 5  – detailed output logs"
OUT27="$SRC_BASE/HE2CT/trainings/training_27/training/log/2025-03-25T100556_training_27"
for fname in config.yaml inference_results.json inference.log \
             logs.log logs.log.{1..5} checkpoints/checkpoint_40.pth; do
    link "$FTD/output" "$(basename "$fname")" "$OUT27/$fname"
done

# ---- 5b. detailed → per-slide model analysis --------------------------------
echo "Step 6  – detailed per-slide analysis"
ANA27_SRC="$SRC_BASE/CT_DS/analyze_trained_model/training_27/output_model"
ANA27_DST="$FTD/finetuned_model_analysis"
for slide in "$ANA27_SRC"/*; do
    [[ -d "$slide" ]] || continue
    sid=$(basename "$slide")
    for f in cell_features_cellvit.npy config.yaml \
             inference_instance_map_predictions.h5 inference_results.json inference.log \
             pannuke_labels_gt.pth; do
        new=$f
        [[ "$f" == "pannuke_labels_gt.pth" ]] && new="pixel_class_gt_mask.pth"
        link "$ANA27_DST/$sid" "$new" "$slide/$f"
    done
done

# ==== 6. fine-tuning – grouped ==============================================
echo "Step 7  – finetuning_CellViT_grouped"
FTG="$STH/data_40x/finetuning_CellViT_grouped"
declare -a ftg_files=(
    "HE2CT/training_datasets/ds_4/cell_count_test.csv"
    "HE2CT/training_datasets/ds_4/cell_count_train.csv"
    "HE2CT/training_datasets/ds_4/cell_count_valid.csv"
    "HE2CT/training_datasets/ds_4/dataset_config.yaml"
    "HE2CT/training_datasets/ds_4/weight_config.yaml"
)
for rel in "${ftg_files[@]}"; do
    link "$FTG" "$(basename "$rel")" "$SRC_BASE/$rel"
done

# ---- 6a. grouped → output logs ---------------------------------------------
echo "Step 8  – grouped output logs"
OUT28="$SRC_BASE/HE2CT/trainings/training_28/training/log/2025-03-26T130326_training_28"
for fname in config.yaml inference_results.json inference.log \
             logs.log logs.log.{1..5} checkpoints/checkpoint_32.pth; do
    link "$FTG/output" "$(basename "$fname")" "$OUT28/$fname"
done

# ---- 6b. grouped → per-slide model analysis ---------------------------------
echo "Step 9  – grouped per-slide analysis"
ANA28_SRC="$SRC_BASE/CT_DS/analyze_trained_model/training_28/output_model"
ANA28_DST="$FTG/finetuned_model_analysis"
for slide in "$ANA28_SRC"/*; do
    [[ -d "$slide" ]] || continue
    sid=$(basename "$slide")
    for f in cell_features_cellvit.npy config.yaml \
             inference_instance_map_predictions.h5 inference_results.json inference.log \
             pannuke_labels_gt.pth; do
        new=$f
        [[ "$f" == "pannuke_labels_gt.pth" ]] && new="pixel_class_gt_mask.pth"
        link "$ANA28_DST/$sid" "$new" "$slide/$f"
    done
done

# ==== 7. data_20x/data =======================================================
echo "Step 10 – data_20x/data core files"
DATA20="$STH/data_20x/data"
declare -a core20=(
    "HE2CT/prepared_datasets_cat_20x/images.zip"
    "HE2CT/prepared_datasets_cat_20x/masks_cell_ids_nuclei.zip"
    "HE2CT/prepared_datasets_cat_20x/ct_1/labels.zip"
    "HE2CT/prepared_datasets_cat_20x/ct_1/cell_count.csv"
    "HE2CT/prepared_datasets_cat_20x/patch_metrics.csv"
    "HE2CT/prepared_datasets_cat_20x/types.csv"
)
for rel in "${core20[@]}"; do
    link "$DATA20" "$(basename "$rel")" "$SRC_BASE/$rel"
done

# ==== 8. pretrained CellViT predictions (per slide, 40×) =====================
echo "Step 11 – pretrained CellViT per-slide preds"
PRED_SRC="$SRC_BASE/CT_DS/check_align_patches/apply_cellvit/output_cellvit"
PRED_DST="$DATA40/pretrained_CellViT_mask_preds"
for slide in "$PRED_SRC"/*; do
    [[ -d "$slide" ]] || continue
    sid=$(basename "$slide")
    link "$PRED_DST" "instance_map_predictions_${sid}.h5" \
         "$slide/inference_instance_map_predictions.h5"
done

# ==== 9. pretrained CellViT combined predictions (20×) =======================
echo "Step 12 – pretrained CellViT combined preds"
link "$DATA20" "instance_map_predictions_all_slides.h5" \
     "$SRC_BASE/HE2CT/prepared_datasets_cat_20x/do_patch_eval/inference_instance_map_predictions.h5"

echo "All symbolic links created.  Verify warnings above (if any)."