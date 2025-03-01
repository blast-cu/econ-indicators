#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/base-quant-train-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

cd /scratch/alpine/alle5715/econ-indicators

module load anaconda
module load cuda/12.1.1

mkdir -p logs
nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
conda activate econ-indicators

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/
export PYTHONPATH=/scratch/alpine/alle5715/econ-indicators

python -m models.roberta_classifier.train_quant --m base
# python -m models.roberta_classifier.train_quant --m base --n all
# python -m models.roberta_classifier.train_quant --m base --n best
