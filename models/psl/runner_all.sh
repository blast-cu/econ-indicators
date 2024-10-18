#!/bin/bash

# python3 models/psl/run_inference.py --s no_inter
# python3 models/psl/eval/evaluate_inference.py --s no_inter
# python3 models/psl/eval/generate_setting_rule_table.py --s no_inter

# python3 models/psl/run_inference.py --s precedes
# python3 models/psl/eval/evaluate_inference.py --s precedes
# python3 models/psl/eval/generate_setting_rule_table.py --s precedes

python3 models/psl/run_inference.py --s excerpt_article
python3 models/psl/eval/evaluate_inference.py --s excerpt_article
python3 models/psl/eval/generate_setting_rule_table.py --s excerpt_article

python3 models/psl/run_inference.py --s inter_article
python3 models/psl/eval/evaluate_inference.py --s inter_article
python3 models/psl/eval/generate_setting_rule_table.py --s inter_article
