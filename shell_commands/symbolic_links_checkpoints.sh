#!/bin/bash

### This script creates symbolic links to the checkpoint file in each slide_id folder, adding also the checkpoints folder if it doesn't exist. ###

# Path to the file to link
source_file="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/checkpoints/CellViT-SAM-H-x40.pth"

# Parent directory containing slide_id folders
slide_id_dir="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/output_cellvit"

# Slide ID to exclude
exclude_slide="att"

# List of slide IDs
slide_ids=(heart_s0)

# Loop through each slide ID
for slide_id in "${slide_ids[@]}"; do
    # Skip the excluded slide ID
    if [ "$slide_id" == "$exclude_slide" ]; then
        echo "Skipping $slide_id"
        continue
    fi

    # Full path to the slide_id folder
    slide_folder="$slide_id_dir/$slide_id"

    # Create the slide_id directory if it doesn't exist
    if [ ! -d "$slide_folder" ]; then
        mkdir -p "$slide_folder"
        echo "Created slide_id directory: $slide_folder"
    fi

    # Path to the checkpoints folder within the current slide_id folder
    checkpoints_dir="$slide_folder/checkpoints"

    # Create the checkpoints directory if it doesn't exist
    mkdir -p "$checkpoints_dir"

    # Create the symbolic link to the source file inside the checkpoints directory
    ln -sf "$source_file" "$checkpoints_dir"

    echo "Added symbolic link in: $checkpoints_dir"
done
