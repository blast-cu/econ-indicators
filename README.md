### Framing in the Presence of Supporting Data: A Case Study in U.S. Economic News

## Quick Start
To begin, clone the repo and create a directory in the top level of the repository called *data*. 
This directory will be ignored by github. Dowload the [economic news article dataset]{} and place it in this directory. 
See [dataset_schema]{dataset_schema.md} for format of expected dataset.


## Model Training and Testing
# Fine Tune RoBERTa for Predicting Annotations

To fine tune a classification model for article-level annotations, from the top level directory, run 

````console
python3 models/roberta_classifier/train_qual.py --m [base, large, dapt] --n [best, all]
````

A model and test results for each fold are generated in *models/roberta_classifier/tuned_models/*

To fine tune a model for a collection of tasks corresponding to qualitative annotations, run 

````console
python3 models/roberta_classifier/train_quant.py --m [base, large, dapt] --n [best, all]
````

Note that the tasks are outlined in detail in *train_quant.py*.


# PSL for improving RoBERTa predictions

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

# Get Article Samples with Predicted Annotations

To generate a csv file of randomly selected articles and their predicted qualitative annotations, from the top 
level directory, run 

````console
python3 models/roberta_classifier/predict_qual.py --db [path_to_db] --ns [number_of_samples]
````

A csv with the article id and the corresponding annotation predictions will be generated in 
*models/roberta_classifier/samples/qual_samples.csv*


# Potato Annotator Testing

To test the quantitative annotator install potato annotator 

````console
pip install potato-annotation
````

Then, run 

````console
python3 potato start potato_annotation/quant_annotate
````

and visit [http://localhost:8000/?PROLIFIC_PID=user](http://localhost:8000/?PROLIFIC_PID=user)

