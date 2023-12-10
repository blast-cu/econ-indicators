
'''
1. Declare Variables
'''

####~~~~ Llama Related Variables ~~~~~####
LLAMA_MODULE_PATH = "/rc_scratch/dasr8731/llama"
LLAMA_CKPT_DIR = "/rc_scratch/dasr8731/llama-2-models/llama-2-7b-chat"
# LLAMA_MODELS = ["/rc_scratch"]


####~~~~ Data Related Variables ~~~~~####
path_to_splits_dict = 'data/splits_dict'
path_to_quant_dict = "data/quant_dict"
path_to_save_predictions = "data/preds_dict"
train_split_to_use_for_prompt_example = 0
test_splits_to_predict = 0

####~~~~ HyperParameters ~~~~~####
max_seq_len = 1024
max_gen_len = 16
batch_size = 16
temperature = 0.6 
top_p = 0.9

'''
2. Module Imports
'''
####~~~~ Add Llama module to PATH for import ~~~~~####

import sys 
sys.path.insert(0, LLAMA_MODULE_PATH)


####~~~~ Regular imports ~~~~~####

import fire
import os
import random
from itertools import product

from models.llama_chat.data_handler import LlamaChatPromptGenerator

from llama import Llama


'''
3. Setting up data
'''

prompts = LlamaChatPromptGenerator()


'''
4. Initializing Models
'''

LLAMA_TOKENIZER_PATH = os.path.join(LLAMA_MODULE_PATH , "tokenizer.model")

generator = Llama.build(
    ckpt_dir=LLAMA_CKPT_DIR,
    tokenizer_path=LLAMA_TOKENIZER_PATH,
    max_seq_len=max_seq_len,
    max_batch_size=batch_size,
)


'''
5. Running generator, saving data
'''

prompt_generator = prompts.generator(test_splits_to_predict)

for article_id, excerpt_id, prompt, label in prompt_generator : 

    print(f"ARTICLE ID : {article_id}")
    print(f"EXCERPT ID : {excerpt_id}")
    print(f"PROMPT : {prompt}")
    print(f"LABEL : {label}")


    prediction = generator.text_completion(
            prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

    print(f"PREDICTION : {prediction}")

    prompts.save_prediction(prediction)

prompts.flush_predictions()

