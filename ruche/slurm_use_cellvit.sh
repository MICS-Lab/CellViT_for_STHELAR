#!/bin/bash
#SBATCH --job-name=train_breast_s0_test5
#SBATCH --output=%x.o%j
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
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

# Train the network
time python /gpfs/users/user/HE2CellType/HE2CT/cell_segmentation/run_cellvit.py --config /gpfs/users/user/HE2CellType/HE2CT/configs/in_use/train_cellvit_breast_s0.yaml
#time python /gpfs/users/user/HE2CellType/HE2CT/cell_segmentation/run_cellvit.py --config /gpfs/users/user/HE2CellType/HE2CT/configs/in_use/train_cellvit_breast_s0_ruche.yaml --checkpoint /gpfs/workdir/user/HE2CellType/HE2CT/run/breast_s0_test4/log/2024-05-18T172309_breast_s0_test4/checkpoints/checkpoint_43.pth
#time python /gpfs/users/user/HE2CellType/HE2CT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py --run_dir /gpfs/workdir/user/HE2CellType/HE2CT/run/breast_s0_test4/log/2024-05-18T172309_breast_s0_test4 --checkpoint_name checkpoint_55.pth --gpu 0 --magnification 40
