#! /bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --output=logs/psl_eval.out
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

for rule in no_inter precedes excerpt_article inter_article
do
    echo "Evaluating inference for rule: $rule"
    python3 models/psl/eval/evaluate_inference.py --s $rule
    python3 models/psl/eval/generate_setting_rule_table.py --s $rule
done

# python3 models/psl/eval/evaluate_inference.py --s combos
# python3 models/psl/eval/generate_setting_rule_table.py --s combos

