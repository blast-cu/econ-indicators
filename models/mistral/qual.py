import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
import data_utils.model_utils.dataset as d
import models.roberta_classifier.utils.qual as qlu
import argparse
import random



MISTRAL_RESULTS_DIR = "data/mistral_results"
PREAMBLE = "You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:"
POSTAMBLE = "Please answer with a single letter without explanations. If you are unsure, please guess."

def_map = {
    'frame': {
            0: "A. Firm-specific", # 'business',
            1: "B. Industry-specific", # 'industry',
            2: "C. Macroeconomic / General Economic Conditions", #'macro',
            3: "D. Government revenue and expenses", # 'government',
            4: "E. None of the above"},
    'econ_rate': {
            0: "A. Good",  # 'good',
            1: "B. Poor", # 'poor',
            2: "C. No opinion", # 'none',
            3: "D. None of the above"  # 'irrelevant'
            },
    'econ_change': {
            0: "A. Getting better", # 'better',
            1: "B. Getting worse", #'worse',
            2: "C. Staying the same", # 'same',
            3: "D. No opinion", #'none',
            4: "D. None of the above"  #'irrelevant'
            }
}

questions = {
    'frame': "What is the main type of economic information covered in this article?",
    'econ_rate': "If this article pertains to Macro-economic / General Economic Conditions, how does it rate economic conditions in the US? "
                    "Select 'None of the above' if the article is not relevant to the U.S. economy or does not pertain to Macroeconomic / General Economic Conditions.",
    'econ_change': "If this article pertains to Macro-economic / General Economic Conditions, does it state/imply that economic conditions in the US as a whole are. . . ?" 
                    "Select 'None of the above' if the article is not relevant to the U.S. economy or does not pertain to Macroeconomic / General Economic Conditions."
}


def no_shot_prompt(text, task):
    """
    Generates a prompt for the Mistral model.
    """
    messages = []
    content_str = f"{PREAMBLE}\n excerpt: {text}\n multiple choice question: {questions[task]}\n"

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
    
    for counter, i in enumerate(idx_choices):
        if counter == 0:
            content_str = f"{PREAMBLE}\n"
        else:
            content_str = ""

        
        excerpt = train_texts[i]
        excerpt = excerpt.replace("\n", " ")
        if len(excerpt) > 2048:
            excerpt = excerpt[:2048]
        content_str += f"So for instance the following:\n excerpt: {excerpt}\n multiple choice question: {questions[task]}\n"
        for options in def_map[task].values():
            content_str += f"{options}\n"
        example_dict = {"role": "user", "content": content_str}
        messages.append(example_dict)

        # example answer
        answer_str = def_map[task][train_labels[i]]
        answer_dict = {"role": "assistant", "content": answer_str}
        messages.append(answer_dict)

    # final prompt
    content_str = f"excerpt: {text}\n multiple choice question: {questions[task]}\n"
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
        text = text.replace("\n", " ")
        if len(text) > 2048:
            text = text[:2048]
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

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    random.seed(42)
    splits_dict = pickle.load(open(d.SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(d.SPLIT_DIR + 'qual_dict', 'rb'))

    # dict for tracking results across folds
    results = {}
    for task in d.qual_label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():

        print()
        print("Fold " + str(k+1) + " of 5")

        split_train_ids = split['train']
        split_test_ids = split['test']

        for task in list(d.qual_label_maps.keys()):

            print("     Task: " + task)

            ann_component = task.split('-')[0]

            test = \
                qlu.get_texts(d.DB_FILENAME,
                                ann_component,
                                task,
                                qual_dict,
                                split_test_ids)
            train = \
                qlu.get_texts(d.DB_FILENAME,
                                ann_component,
                                task,
                                qual_dict,
                                split_train_ids)


            results[task]['labels'] += test[1]
            prompts = get_prompts(train, test, task, shots=SHOTS)
            for p in prompts:
                model_inputs = tokenizer.apply_chat_template(p, return_tensors="pt").to("cuda")


                generated_ids = model.generate(model_inputs, max_new_tokens=40, do_sample=True)
                response = tokenizer.batch_decode(generated_ids)[0]
                results[task]['predictions'].append(response)


    pickle.dump(results, open(f'{MISTRAL_RESULTS_DIR}/qual_{SHOTS}_shot_results', 'wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--s', required=True, help='Number of shots')
    args = parser.parse_args()
    main(args)