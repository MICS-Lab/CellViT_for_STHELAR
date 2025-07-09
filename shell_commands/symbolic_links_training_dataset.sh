#!/bin/bash

# Variables
cell_cat_id="ct_1"
training_dataset_id="ds_4"

# Base directories
source_dir="/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat"
target_dir="/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/training_datasets/${training_dataset_id}"

# Files to link
files=(
    "${source_dir}/images.zip"
    "${source_dir}/${cell_cat_id}/ALL/labels.zip"
    "${source_dir}/${cell_cat_id}/ALL/types.csv"
    "${source_dir}/adatas"
    "${source_dir}/masks_cell_ids_nuclei.zip"
    "${source_dir}/patch_bboxes.parquet"
    "${source_dir}/nucleus_centroids"
)

# Ensure target directory exists
mkdir -p "${target_dir}"

# Create symbolic links
for file in "${files[@]}"; do
    filename=$(basename "${file}")
    target="${target_dir}/${filename}"

    if [ ! -e "${target}" ]; then
        ln -s "${file}" "${target}"
        echo "Created symbolic link: ${target} -> ${file}"
    else
        echo "Link already exists: ${target}"
    fi
done


# NB: to run the script:
# chmod +x shell_commands/symbolic_links_training_dataset.sh
# ./shell_commands/symbolic_links_training_dataset.sh