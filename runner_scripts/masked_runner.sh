#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=masked_tune-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

mkdir -p logs
nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
conda activate /rc_scratch/alle5715/argmin-sharedtask/venv

mkdir -p metadata
mkdir -p outputs

export HF_HOME=metadata/
export PYTHONPATH=/rc_scratch/alle5715/econ-indicators

module load cuda
module load cudnn

python3 models/roberta_classifier/dapt.py --o models/roberta_classifier/tuned_models/roberta_base_dapt_512 --c roberta-base --s 512
