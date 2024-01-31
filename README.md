# econ-indicators
Narratives around Economic Indicators in News Media

## Fine Tune RoBERTa for Predicting Annotations

To fine tune a model for each qualitative component, from the top level directory, run 

````console
python3 models/roberta_classifier/train_qual.py --db [path_to_db]
````

A classification report over all folds of the data is generated in *models/roberta_classifier/results/qual* 
for each annotation component. Further, the model with the best macro F1 score for each component is saved into 
*models/roberta_classifier/best_models/qual* along with a classification report corresponding to this data fold. 

To fine tune a model for a collection of tasks corresponding to qualitative annotations, run 

````console
python3 models/roberta_classifier/train_quant.py --db [path_to_db]
````

Note that the tasks are outlined in detail in *train_quant.py*


## PSL for improving RoBERTa predictions

To generate data needed for all rule settings into the appropriate split subdirectories of the data directory 
(see data/split{}/eval and data/split{}/learn)

````console
python3 models/psl/generate_data.py
````
Note that this will take time to complete because RoBERTa predictions are obtained during this process. To generate
rule files into the data/rules directory, run 

````console
python3 models/psl/generate_rules.py
````

This will create necessary files for running all ablaion studies. To (finally) run inference with the desired setting: 

````console
python3 models/psl/run_inference.py --s SETTING
````
See SETTINGS.py for rule setting options. To evaluate, run 

````console
python3 models/psl/evaluate_inference.py --s SETTING
````

This creates eval tables for each annotation component and each rule setting in data/results. To generate a table that 
includes the macro and weighted f1 for each component, run 

````console
python3 models/psl/generate_eval_tables/generate_setting_rule_table.py --dir data_dir
````
Where data_dir is the path to the directory containing all eval tables created in the previous step. 

## Get Article Samples with Predicted Annotations

To generate a csv file of randomly selected articles and their predicted qualitative annotations, from the top 
level directory, run 

````console
python3 models/roberta_classifier/predict_qual.py --db [path_to_db] --ns [number_of_samples]
````

A csv with the article id and the corresponding annotation predictions will be generated in 
*models/roberta_classifier/samples/qual_samples.csv*