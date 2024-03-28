import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
import data_utils.model_utils.dataset as d
import models.roberta_classifier.utils.quant as qu
import argparse

MISTRAL_RESULTS_DIR = "data/mistral_results"

def no_shot_prompt(text, task):
    """
    Generates a prompt for the Mistral model.
    """
    messages = []
    preamble = "You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:"
    postamble = "Please answer with a single letter without explanations. If you are unsure, please guess."

    if task == "type":
        messages = [
            {"role": "user",
            "content": f"{preamble}\n"
                f"excerpt: {text[0]}\n"
                f"context: {text[1]}\n"
                "multiple choice question: The excerpt should contain an economic indicator value. Based on the context, what type of indicator is it?\n"
                "A. Macroeconomic / General Economic Conditions\n"
                "B. Firm-specific\n"
                "C. Industry-specific\n"
                "D. Government revenue and expenses\n"
                "E. Other\n"
                "F. None of the above\n"
                f"{postamble}"}
        ]
    elif task == "macro_type":
        messages = [
            {"role": "user",
            "content": f"{preamble}\n"
                f"excerpt: {text[0]}\n"
                f"context: {text[1]}\n"
                "multiple choice question: The excerpt should contain an economic indicator value. " 
                "If the indicator's general type is 'Macroeconomic / General Economic Conditions', what specific type of indicator is it? " 
                "Select 'None of the above' if the quantity is not relevant to the U.S. economy or is not a Macroeconomic / General Economic Conditions type.\n"
                "A. Job Numbers\n"
                "B. Market Numbers\n"
                "C. Housing\n"
                "D. Macroeconomy\n"
                "E. Wages\n"
                "F. Prices\n"
                "G. Confidence\n"
                "H. Retail Sales\n"
                "I. Interest Rates\n"
                "J. Currency Values\n"
                "K. Energy Prices\n"
                "L. None of the above\n"
                f"{postamble}"}
        ]
    elif task == "spin":
        messages = [
            {"role": "user",
             "content": f"{preamble}\n"
                f"excerpt: {text[0]}\n"
                f"context: {text[1]}\n"
                "multiple choice question: The excerpt should contain an economic indicator value. " 
                "If the quantity's general type is 'Macroeconomic / General Economic Conditions', what spin does the writer of the excerpt put on the indicator? " 
                "Select 'None of the above' if the quantity is not relevant to the U.S. economy or is not a Macroeconomic / General Economic Conditions type.\n"
                "A. Positive\n"
                "B. Negative\n"
                "C. Neutral\n"
                "F. None of the above\n"
                f"{postamble}"}
        ]
    else:
        raise ValueError(f"Task {task} not recognized.")

    return messages


def shot_prompt(text, train, shots):
    """
    Generates a prompt for the Mistral model.
    """
    prompt = "My favourite condiment is"
    return prompt


def get_prompts(train, test, task, shots=0):
    """
    Returns a list of prompts for the Mistral model.
    """
    test_text = test[0]

    prompts = []
    for text in test_text:
        if shots == 0:
            prompt = no_shot_prompt(text, task)
        else:
            prompt = shot_prompt(text, train, shots)

        prompts.append(prompt)
    
    return prompts

def main(args):

    try:
        SHOTS = int(args.s)
    except ValueError:
        raise ValueError("Number of shots must be an integer.")

    os.makedirs(MISTRAL_RESULTS_DIR, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    splits_dict = pickle.load(open(d.SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(d.SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(d.SPLIT_DIR + 'quant_dict', 'rb'))

    # dict for tracking results across folds
    results = {}
    for task in d.quant_label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():

        print()
        print("Fold " + str(k+1) + " of 5")

        split_train_ids = split['train']
        split_test_ids = split['test']

        for task in list(d.quant_label_maps.keys()):

            print("     Task: " + task)

            ann_component = task.split('-')[0]

            train = \
                qu.get_texts(ann_component,
                             task,
                             qual_dict,
                             quant_dict,
                             split_train_ids)

            test = \
                qu.get_texts(ann_component,
                             task,
                             qual_dict,
                             quant_dict,
                             split_test_ids)

            results[task]['labels'] += test[1]
            prompts = get_prompts(train, test, task, shots=SHOTS)
            for p in prompts:
                # print(p)

                model_inputs = tokenizer.apply_chat_template(p, return_tensors="pt").to("cuda")

                generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
                response = tokenizer.batch_decode(generated_ids)[0]
                results[task]['predictions'].append(response)

    pickle.dump(results, open(f'{MISTRAL_RESULTS_DIR}/{SHOTS}_shot_results', 'wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--s', required=True, help='Number of shots')
    args = parser.parse_args()
    main(args)