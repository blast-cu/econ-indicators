#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=128G
#SBATCH --partition=aa100
#SBATCH --time=24:00:00
#SBATCH --output=logs/read_new_data-%j.out
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

export PYTHONPATH=/scratch/alpine/alle5715/econ-indicators

python -m data_utils.collection.add_data
