module load gcc/9.2.0/gcc-4.8.5
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/intel-20.0.2
conda env create -f ./users/user/HE2CellType/HE2CT/environment.yml
source activate cellvit
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
conda env export > ./users/user/HE2CellType/HE2CT/ruche/config/environment_cellvit.yml # save conda environment description
