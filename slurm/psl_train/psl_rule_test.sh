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

mkdir -p metadata
mkdir -p outputs

export TRANSFORMERS_CACHE=metadata/
export PYTHONPATH=/scratch/alpine/alle5715/econ-indicators

python3 models/psl/run_inference.py --s no_inter
# python3 models/psl/eval/evaluate_inference.py --s no_inter
# python3 models/psl/eval/generate_setting_rule_table.py --s no_inter

# python3 models/psl/run_inference.py --s no_inter --no_inter
# python3 models/psl/eval/evaluate_inference.py --s no_inter
# python3 models/psl/eval/generate_setting_rule_table.py --s no_inter

# python3 models/psl/run_inference.py --s precedes
# python3 models/psl/eval/evaluate_inference.py --s precedes
# python3 models/psl/eval/generate_setting_rule_table.py --s precedes

# python3 models/psl/run_inference.py --s excerpt_article
# python3 models/psl/eval/evaluate_inference.py --s excerpt_article
# python3 models/psl/eval/generate_setting_rule_table.py --s excerpt_article

# python3 models/psl/run_inference.py --s inter_article
# python3 models/psl/eval/evaluate_inference.py --s inter_article
# python3 models/psl/eval/generate_setting_rule_table.py --s inter_article

# python3 models/psl/run_inference.py --s combos
# python3 models/psl/eval/evaluate_inference.py --s combos
# python3 models/psl/eval/generate_setting_rule_table.py --s combos

