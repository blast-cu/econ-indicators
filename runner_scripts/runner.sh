#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=quant_prediction-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

mkdir -p logs
nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
source activate ../econ-indicators-copy/venv

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/

module load cuda
module load cudnn

python -m models.roberta_classifier.predict_qual --ns 16
