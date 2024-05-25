#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=48:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mistral-2-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

mkdir -p logs
nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
source activate ./venv
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export 'PYTORCH_NO_CUDA_MEMORY_CACHING=1'

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/

module load cuda
module load cudnn

python -m models.mistral.qual --s 2