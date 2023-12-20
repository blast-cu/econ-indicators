
Options for different ablation studies are included in SETTINGS.py. For example, to run the neighbors ablation study we test each rule set
included in data/rules/neighbors individually.

First we run inference with each rule file: 

````console
python3 models/psl/run_inference.py --s neighbors
````

Then, we generate a classification report for each rule set and for each annotation component in the corresponding data/split{split_num}/ 
directory. We also generate classification reports over all splits for each annotation component in data/results with:  

````console
python3 models/psl/evaluate_inference.py --s neighbors
````

Finally, we can create a table with the weighted and macro F1 for each annotation component and rule file in the same csv:

````console
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir neighbors
````