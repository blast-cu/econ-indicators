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


## Get Article Samples with Predicted Annotations

To generate a csv file of randomly selected articles and their predicted qualitative annotations, from the top 
level directory, run 

````console
python3 models/roberta_classifier/predict_qual.py --db [path_to_db] --ns [number_of_samples]
````

A csv with the article id and the corresponding annotation predictions will be generated in 
*models/roberta_classifier/samples/qual_samples.csv*