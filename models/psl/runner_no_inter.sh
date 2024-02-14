#!/bin/bash

python3 models/psl/run_inference.py --s inter_article
python3 models/psl/evaluate_inference.py --s inter_article
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir inter_article
