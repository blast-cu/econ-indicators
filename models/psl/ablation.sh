#!/bin/bash

cd /home/aleto/research/econ-indicators
source env/bin/activate

python3 models/psl/generate_rules.py

for split in {0..4}
do 
python3 models/psl/run_inference.py --split $split
done

python3 models/psl/evaluate_inference.py