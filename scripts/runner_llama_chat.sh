#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=llama-chat-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="dasr8731@colorado.edu"

module load cuda
module load cudnn

nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
source activate /rc_scratch/dasr8731/llama_env

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/


python -m models.llama_chat.llama_chat_classifier