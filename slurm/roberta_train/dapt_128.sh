#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dapt_128-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

cd /scratch/alpine/alle5715/econ-indicators

module load anaconda
module load cuda/12.1.1

mkdir -p logs
nvidia-smi >> logs/nvidia-smi.out

conda activate econ-indicators

mkdir -p metadata
mkdir -p outputs

export HF_HOME=metadata/
export PYTHONPATH=/scratch/alpine/alle5715/econ-indicators

python3 models/roberta_classifier/dapt.py --o data/models/roberta_base_dapt_128 --c roberta-base --s 128
