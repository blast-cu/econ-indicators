#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --partition=atesting_a100
#SBATCH --gres=gpu:1
#SBATCH --output=mask_tune_test-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
source activate ./venv

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/

module load cuda
module load cudnn

python -m models.roberta_classifier.masked_tune
