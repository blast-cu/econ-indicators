# econ-indicators
Code for the paper [Framing in the Presence of Supporting Data: A Case Study in U.S. Economic News](https://arxiv.org/abs/2402.14224)

## Abstract
The mainstream media has much leeway in what it chooses to cover and how it covers it. These choices have real-world consequences on what people know and their subsequent behaviors. However, the lack of objective measures to evaluate editorial choices makes research in this area particularly difficult. In this paper, we argue that there are newsworthy topics where objective measures exist in the form of supporting data and propose a computational framework to analyze editorial choices in this setup. We focus on the economy because the reporting of economic indicators presents us with a relatively easy way to determine both the selection and framing of various publications. Their values provide a ground truth of how the economy is doing relative to how the publications choose to cover it. To do this, we define frame prediction as a set of interdependent tasks. At the article level, we learn to identify the reported stance towards the general state of the economy. Then, for every numerical quantity reported in the article, we learn to identify whether it corresponds to an economic indicator and whether it is being reported in a positive or negative way. To perform our analysis, we track six American publishers and each article that appeared in the top 10 slots of their landing page between 2015 and 2023. 


## Quick Start
To begin, clone the repo and create a directory in the top level of the repository called *data*. This directory will be ignored by github. Dowload the [economic news article dataset]() and place it in this directory. See [Schema](#schema) for format of expected dataset.

## Generate Data Splits

## Model Training and Testing
### Fine Tune RoBERTa for Predicting Annotations

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


### PSL for improving RoBERTa predictions

To generate data needed for all rule settings into the appropriate split subdirectories of the data directory (see data/split{}/eval and data/split{}/learn)

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

## Using Models to Get Predictions
### Get Article Samples with Predicted Annotations

To generate a csv file of randomly selected articles and their predicted qualitative annotations, from the top 
level directory, run 

````console
python3 models/roberta_classifier/predict_qual.py --db [path_to_db] --ns [number_of_samples]
````

A csv with the article id and the corresponding annotation predictions will be generated in 
*models/roberta_classifier/samples/qual_samples.csv*

## Data Visualization 
### Tables in Paper
### Figures in Paper

## Dataset Info
### Schema

#### user
|  Field   |     Type     | Nullable | Key | Default | Index | Extra |
| -------- | ------------ | -------- | --- | ------- | :---: | ----- |
| id       | INTEGER      | NO       |     |         |       |       |
| email    | VARCHAR(100) | YES      |     | NULL    |       |       |
| password | VARCHAR(100) | YES      |     | NULL    |       |       |
| admin    | BOOLEAN      | NO       |     |         |       |       |
| PRIMARY  | KEY          | YES      |     | NULL    |       |       |
| UNIQUE   | (email)      | YES      |     | NULL    |       |       |

#### topic
|    Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ----------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id          | INTEGER       | NO       |     |         |       |       |
| description | VARCHAR(1000) | YES      |     | NULL    |       |       |
| relevant    | BOOLEAN       | NO       |     |         |       |       |
| name        | VARCHAR(1000) | NO       |     |         |       |       |
| PRIMARY     | KEY           | YES      |     | NULL    |       |       |
| UNIQUE      | (name)        | YES      |     | NULL    |       |       |

#### cluster
|    Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ----------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id          | INTEGER       | NO       |     |         |       |       |
| name        | VARCHAR(1000) | NO       |     |         |       |       |
| explanation | VARCHAR(1000) | YES      |     | NULL    |       |       |
| PRIMARY     | KEY           | YES      |     | NULL    |       |       |
| UNIQUE      | (name)        | YES      |     | NULL    |       |       |

#### clusters
|   Field    |  Type   | Nullable | Key | Default | Index | Extra |
| ---------- | ------- | -------- | --- | ------- | :---: | ----- |
| cluster_id | INTEGER | NO       |     |         |       |       |
| user_id    | INTEGER | NO       |     |         |       |       |
| PRIMARY    | KEY     | YES      |     | NULL    |       |       |
| user_id)   |         | YES      |     | NULL    |       |       |

#### article
|    Field     |     Type      | Nullable | Key | Default | Index | Extra |
| ------------ | ------------- | -------- | --- | ------- | :---: | ----- |
| id           | INTEGER       | NO       |     |         |       |       |
| headline     | VARCHAR(1000) | NO       |     |         |       |       |
| source       | VARCHAR(100)  | NO       |     |         |       |       |
| keywords     | VARCHAR(100)  | NO       |     |         |       |       |
| num_keywords | INTEGER       | NO       |     |         |       |       |
| relevance    | FLOAT         | NO       |     |         |       |       |
| text         | VARCHAR(5000) | NO       |     |         |       |       |
| distance     | FLOAT         | YES      |     | NULL    |       |       |
| date         | DATETIME      | NO       |     |         |       |       |
| url          | VARCHAR(100)  | NO       |     |         |       |       |
| cluster_id   | INTEGER       | YES      |     | NULL    |       |       |
| PRIMARY      | KEY           | YES      |     | NULL    |       |       |

#### articleann
|    Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ----------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id          | INTEGER       | NO       |     |         |       |       |
| user_id     | INTEGER       | NO       |     |         |       |       |
| article_id  | INTEGER       | NO       |     |         |       |       |
| frame       | VARCHAR       | YES      |     | NULL    |       |       |
| econ_rate   | VARCHAR       | YES      |     | NULL    |       |       |
| econ_change | VARCHAR       | YES      |     | NULL    |       |       |
| comments    | VARCHAR(1000) | YES      |     | NULL    |       |       |
| text        | VARCHAR(5000) | YES      |     | NULL    |       |       |
| PRIMARY     | KEY           | YES      |     | NULL    |       |       |

#### quantity
|   Field    |     Type      | Nullable | Key | Default | Index | Extra |
| ---------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id         | VARCHAR(1000) | NO       |     |         |       |       |
| local_id   | INTEGER       | NO       |     |         |       |       |
| article_id | INTEGER       | NO       |     |         |       |       |
| PRIMARY    | KEY           | YES      |     | NULL    |       |       |

##### topics
|     Field      |  Type   | Nullable | Key | Default | Index | Extra |
| -------------- | ------- | -------- | --- | ------- | :---: | ----- |
| topic_id       | INTEGER | NO       |     |         |       |       |
| articleann_id  | INTEGER | NO       |     |         |       |       |
| PRIMARY        | KEY     | YES      |     | NULL    |       |       |
| articleann_id) |         | YES      |     | NULL    |       |       |

##### quantityann
|      Field       |     Type      | Nullable | Key | Default | Index | Extra |
| ---------------- | ------------- | -------- | --- | ------- | :---: | ----- |
| id               | INTEGER       | NO       |     |         |       |       |
| user_id          | INTEGER       | NO       |     |         |       |       |
| quantity_id      | INTEGER       | NO       |     |         |       |       |
| type             | VARCHAR(1000) | YES      |     | NULL    |       |       |
| macro_type       | VARCHAR(1000) | YES      |     | NULL    |       |       |
| industry_type    | VARCHAR(1000) | YES      |     | NULL    |       |       |
| gov_level        | VARCHAR(1000) | YES      |     | NULL    |       |       |
| gov_type         | VARCHAR(1000) | YES      |     | NULL    |       |       |
| expenditure_type | VARCHAR(1000) | YES      |     | NULL    |       |       |
| revenue_type     | VARCHAR(1000) | YES      |     | NULL    |       |       |
| comments         | VARCHAR(1000) | YES      |     | NULL    |       |       |
| spin             | VARCHAR(1000) | YES      |     | NULL    |       |       |
| PRIMARY          | KEY           | YES      |     | NULL    |       |       |

#### articles
|   Field    |  Type   | Nullable | Key | Default | Index | Extra |
| ---------- | ------- | -------- | --- | ------- | :---: | ----- |
| article_id | INTEGER | NO       |     |         |       |       |
| user_id    | INTEGER | NO       |     |         |       |       |
| PRIMARY    | KEY     | YES      |     | NULL    |       |       |
| user_id)   |         | YES      |     | NULL    |       |       |

## Potato Annotator Testing

To test the quantitative annotator install potato annotator 

````console
pip install potato-annotation
````

Then, run 

````console
python3 potato start potato_annotation/quant_annotate
````

and visit [http://localhost:8000/?PROLIFIC_PID=user](http://localhost:8000/?PROLIFIC_PID=user)


## Cite us 
If you find our dataset or classifiers helpful, please cite us as: 
````console
@misc{leto2024framing,
      title={Framing in the Presence of Supporting Data: A Case Study in U.S. Economic News}, 
      author={Alexandria Leto and Elliot Pickens and Coen D. Needell and David Rothschild and Maria Leonor Pacheco},
      year={2024},
      eprint={2402.14224},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
````
