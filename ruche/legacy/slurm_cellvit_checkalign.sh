#!/bin/bash
#SBATCH --job-name=checkalign
#SBATCH --output=%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --partition=gpua100
#SBATCH --mail-type=ALL

# Module load
module load gcc/9.2.0/gcc-4.8.5
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Activate anaconda environment code
source activate cellvit_fgs

# Fix for openslide
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib:$HOME/usr/lib64/

# List of slide IDs
slide_ids=(breast_s1 breast_s0 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 brain_s0 bone_marrow_s0 bone_marrow_s1 bone_s0 prostate_s0 cervix_s0)

# Loop through each slide ID
for slide_id in "${slide_ids[@]}"; do
    echo "Processing slide: $slide_id"

    # Set the run directory based on the slide ID
    run_dir="/gpfs/workdir/user/HE2CellType/CT_DS/check_align_patches/apply_cellvit/output_cellvit/$slide_id"

    # Run the inference command
    time python /gpfs/users/user/HE2CellType/HE2CT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py \
        --run_dir "$run_dir" \
        --checkpoint_name CellViT-SAM-H-x40.pth \
        --gpu 0 \
        --magnification 40 \
        --cell_tokens nucleus

    echo "Finished processing slide: $slide_id"
    echo "======================================================="
    echo " "
done