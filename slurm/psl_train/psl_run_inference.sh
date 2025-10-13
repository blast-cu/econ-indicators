#! /bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --output=logs/psl_testing.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="alle5715@colorado.edu"

cd /scratch/alpine/alle5715/econ-indicators

module load anaconda
module load cuda/12.1.1

mkdir -p logs
nvidia-smi >> logs/nvidia-smi.out

source /home/${USER}/.bashrc
conda activate econ-indicators
source env-psl/bin/activate

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/
export PYTHONPATH=/scratch/alpine/alle5715/econ-indicators


# loop over rules
for rule in no_inter precedes excerpt_article inter_article
do
    echo "Running inference for rule: $rule"
    python3 models/psl/run_inference.py --s $rule
done


# python3 models/psl/run_inference.py --s combos

