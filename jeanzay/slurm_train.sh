#!/bin/bash
#SBATCH --job-name=training_11_3         # nom du job
#SBATCH --output=./output/%x_%j.out     # fichier de sortie (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100               # demander des GPU A100 80 Go ou des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 nœuds
#SBATCH --ntasks=1                      # reserver 1 taches (ou processus)
#SBATCH --gres=gpu:1                    # reserver 1 GPU par noeud
#SBATCH --cpus-per-task=16               # reserver 4 CPU par tache (IMPORTANT pour la memoire dispo)
#SBATCH --time=20:00:00                 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread            # desactiver l’hyperthreading
#SBATCH --account=user@h100              # comptabilite A100 ou H100
#SBATCH --mail-type=ALL                 # When to send email notifications
#SBATCH --mail-user=mail  # Your email address

# Nettoyer les modules herites par defaut
module purge 

# Selectionner les modules compiles pour les A100 ou H100
module load arch/h100 

# Charger les modules
# No module to load for cuda or cudnn because already included in the conda env
module load miniforge/24.9.0

# Desactiver les environnements herites par defaut
conda deactivate 

# Activer environnement conda
conda activate cellvit_env

# Executer son script

# Start training
#srun /linkhome/rech/genrce01/ubu16ws/HE2CellType/HE2CT/cell_segmentation/run_cellvit.py --config /lustre/fswork/projects/rech/user/ubu16ws/HE2CellType/HE2CT/trainings/training_13/training_13_cellvit.yaml

# Restart training from a previous checkpoint
srun /linkhome/rech/genrce01/ubu16ws/HE2CellType/HE2CT/cell_segmentation/run_cellvit.py --config /lustre/fswork/projects/rech/user/ubu16ws/HE2CellType/HE2CT/trainings/training_11/training_11_cellvit.yaml --checkpoint /lustre/fswork/projects/rech/user/ubu16ws/HE2CellType/HE2CT/trainings/training_11/log/2025-01-31T180049_training_11/checkpoints/checkpoint_32.pth