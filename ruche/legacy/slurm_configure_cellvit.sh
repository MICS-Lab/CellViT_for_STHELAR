#!/bin/bash
#SBATCH --job-name=configure_cellvit
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_test

# Setup conda env - ensure your .conda dir is located on your workir, and move it if not
[ -L ~/.conda ] && unlink ~/.conda
[ -d ~/.conda ] && mv -v ~/.conda $WORKDIR
[ ! -d $WORKDIR/.conda ] && mkdir $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda

# Module load
module load gcc/9.2.0/gcc-4.8.5
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2

# Create conda environment
conda env create -f HE2CellType/HE2CT/ruche/configs/environment_cellvit.yml --force
# ou juste create et pip install à la main

# Save environment description
#source activate scgpt
#conda env export > config/environment_cellvit.yml
