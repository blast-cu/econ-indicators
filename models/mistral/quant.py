import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
import data_utils.model_utils.dataset as d
import models.roberta_classifier.utils.quant as qu
import argparse
import random

MISTRAL_RESULTS_DIR = "data/mistral_results"
PREAMBLE = "You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:"
POSTAMBLE = "Please answer with a single letter without explanations. If you are unsure, please guess."

def_map = {
    'type': {
                0: "A. Macroeconomic / General Economic Conditions",  # macro
                1: "B. Industry-specific", #'industry',
                2: "C. Government revenue and expenses",  #'government',
                3: "D. Personal", #'personal',
                4: "E. Firm-specific",  #'business',
                5: "F. None of the above" #'other'
            },
    'macro_type': {
                0: "A. Job Numbers",
                1: "B. Retail Sales",
                2: "C. Interest Rates",
                3: "D. Prices",
                4: "E. Energy Prices",
                5: "F. Wages",
                6: "G. Macroeconomy",
                7: "H. Market Numbers",
                8: "I. Currency Values",
                9: "J. Housing",
                10: "K. Other",
                11: "L. None of the above"
            },
    'spin': {
                0: "A. Positive",
                1: "B. Negative",
                2: "C. Neutral",
                3: "D. None of the above"  # fixme
    }
}

questions = {
    'type': "The excerpt should contain an economic indicator value. Based on the context, what type of indicator is it?",
    'macro_type': "The excerpt should contain an economic indicator value. " 
                    "If the indicator's general type is 'Macroeconomic / General Economic Conditions', what specific type of indicator is it? "
                    "Select 'None of the above' if the quantity is not relevant to the U.S. economy or is not a Macroeconomic / General Economic Conditions type.",
    'spin': "If the quantity's general type is 'Macroeconomic / General Economic Conditions', what spin does the writer of the excerpt put on the indicator? " 
                    "Select 'None of the above' if the quantity is not relevant to the U.S. economy or is not a Macroeconomic / General Economic Conditions type."
}


def no_shot_prompt(text, task):
    """
    Generates a prompt for the Mistral model.
    """
    messages = []
    content_str = f"{PREAMBLE}\n excerpt: {text[0]}\n context: {text[1]}\n multiple choice question: {questions[task]}\n"

    for options in def_map[task].values():
        content_str += f"{options}\n"

    content_str += f"{POSTAMBLE}"

    messages = [
        {"role": "user",
         "content": content_str}
    ]
    return messages


def shot_prompt(text, train, task, shots):
    """
    Generates a prompt for the Mistral model.
    """
    train_texts = train[0]
    train_labels = train[1]
    options = list(range(len(train_texts)))
    idx_choices = random.sample(options, shots)

    messages = []
    
    for i in idx_choices:
        if i == 0:
            content_str = f"{PREAMBLE}\n"
        else:
            content_str = ""

        indicator_text = train_texts[i][0]
        context = train_texts[i][1]
        content_str += f"So for instance the following:\n excerpt: {indicator_text}\n context: {context}\n multiple choice question: {questions[task]}\n"
        for options in def_map[task].values():
            content_str += f"{options}\n"
        example_dict = {"role": "user", "content": content_str}
        messages.append(example_dict)

        # example answer
        answer_str = def_map[task][train_labels[i]]
        answer_dict = {"role": "assistant", "content": answer_str}
        messages.append(answer_dict)

    # final prompt
    content_str = f"excerpt: {text[0]}\n context: {text[1]}\n multiple choice question: {questions[task]}\n"
    for options in def_map[task].values():
        content_str += f"{options}\n"
    content_str += f"{POSTAMBLE}"
    messages.append({"role": "user", "content": content_str})

    return messages


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
            prompt = shot_prompt(text, train, task, shots)

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