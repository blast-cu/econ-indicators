#! /bin/bash

#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=168:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --output=logs/psl_final_inference0-%j.out
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

python -m models.psl.split_data

for i in {0..9}; do
    python3 -m models.psl.run_inference --s best_2025-05 --final_split $i
done

