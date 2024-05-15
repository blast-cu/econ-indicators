#!/bin/bash

python3 models/psl/run_inference.py --s best_rules
python3 models/psl/eval/evaluate_inference.py --s best_rules
python3 models/psl/eval/generate_setting_rule_table.py --s best_rules
