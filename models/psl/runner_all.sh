#!/bin/bash

python3 models/psl/run_inference.py --s no_inter
python3 models/psl/evaluate_inference.py --s no_inter
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir no_inter

python3 models/psl/run_inference.py --s precedes
python3 models/psl/evaluate_inference.py --s precedes
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir precedes

python3 models/psl/run_inference.py --s excerpt_article
python3 models/psl/evaluate_inference.py --s excerpt_article
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir excerpt_article

python3 models/psl/run_inference.py --s inter_article
python3 models/psl/evaluate_inference.py --s inter_article
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir inter_article
