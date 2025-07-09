#!/bin/bash
#SBATCH --job-name=checkalign_dataset20x         # nom du job
#SBATCH --output=./output/%x_%j.out     # fichier de sortie (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100               # demander des GPU A100 80 Go ou des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 nœuds
#SBATCH --ntasks=1                      # reserver 1 taches (ou processus)
#SBATCH --gres=gpu:1                    # reserver 1 GPU par noeud
#SBATCH --cpus-per-task=23               # reserver 4 CPU par tache (IMPORTANT pour la memoire dispo)
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

# Executer script inference
srun /linkhome/rech/genrce01/ubu16ws/HE2CellType/HE2CT/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py --run_dir /lustre/fswork/projects/rech/user/ubu16ws/HE2CellType/HE2CT/prepared_datasets_cat_20x/do_patch_eval --checkpoint_name CellViT-SAM-H-x20.pth --gpu 0 --magnification 20