from sklearn.metrics import classification_report, f1_score
import pandas as pd

"""
Methods for reporting model performance
"""


def to_csv(annotation_component: str,
           labels: list,
           predicted: list,
           destination: str = "models/roberta/results"):
    """
    Write classification report to a CSV file.

    Parameters:
    - annotation_component (str): Name of the annotation component.
    - labels (list): List of true labels.
    - predicted (list): List of predicted labels.
    - destination (str, optional): Path to save the CSV file. Defaults to "models/roberta/results".
    """
    report = classification_report(labels,
                                   predicted,
                                   output_dict=True,
                                   zero_division=0)

    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{destination}/{annotation_component}_classification_report.csv")


def to_f1_csv(results: dict,
              destination: str,
              f1: str):
    """
    Convert F1 scores from `results` dictionary to a CSV file.

    Parameters:
    - results (dict): A dictionary containing F1 scores for different tasks.
        The keys of the dictionary represent the task names, and the values
        are dictionaries with 'labels' and 'predictions' keys.
    - destination (str): The destination path where the CSV file will be saved.
    - f1 (str): The averaging method to be used for calculating F1 score.
        Possible values are 'micro', 'macro', 'weighted', or 'samples'.

    Returns:
    - None: This function does not return anything. It saves the F1 scores
    as a CSV file at the specified destination.
    """
    destination = destination + f1 + "_f1_report.csv"

    results_formatted = {}
    for task in results.keys():
        results_formatted[task] = []
        labels = results[task]['labels']
        predictions = results[task]['predictions']
        score = f1_score(labels, predictions, average=f1)
        score = round(score, 3)
        results_formatted[task].append(score)

    df = pd.DataFrame(results_formatted)
    df.to_csv(destination)
