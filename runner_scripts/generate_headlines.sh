#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=168:00:00
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/generate-headlines-%j.out
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

declare -a sources=("apnews" "breitbart" "cnn" "foxnews" "huffpost" "nytimes" "reuters" "theguardian" "usatoday" "washingtonpost" "wsj")

for source in "${sources[@]}"
do
    echo "Generating headlines for $source"
    python -m data_utils.collection.generate_headlines --dataset data/2024_dump/text/$source/articles.json
done
