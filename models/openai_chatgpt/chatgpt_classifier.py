import openai
import pickle
import time

from openai import OpenAI
from models.llama_chat.data_handler import ExcerptTypePromptHandler
from models.openai_chatgpt.subtype_data_handler import ExcerptSubTypePromptHandler


'''
Setting up prompt handlers
'''
excerpt_type_handler = ExcerptTypePromptHandler(path_to_quant_dict='data/quant_dict_clean')
excerpt_subtype_handler = ExcerptSubTypePromptHandler(path_to_quant_dict='data/quant_dict_clean')


'''
Setting up GPT stuff
'''
client = OpenAI(
    api_key=open('data/openai.key').read().strip(),
)

openai.api_key = open('data/openai.key').read().strip()
MODEL_NAME = "gpt-4"
temperature = 0.2
max_tokens = 4
sleep_time = 20 

'''
Function Definitions
'''

def gpt4_prediction(prompt) : 

    messages = [{"role": "system", "content": "You are a helpful assistant.",},
                {"role" : "user","content" : prompt }]

    predicted_label = client.chat.completions.create(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model=MODEL_NAME,
    )

    time.sleep(sleep_time)

    return predicted_label.choices[0].message.content

'''
Main Loop
'''
test_splits = [0,1,2,3,4]

for test_split in test_splits : 

    print("Working on Test Split : " , test_split)

    macro_accuracy_counter = 0
    subtype_accuracy_counter = 0
    oracle_subtype_accuracy_counter = 0
    save_path = f"data/preds_clean_split{test_split}_dict"

    for i, p in enumerate(excerpt_type_handler.generator(test_split)) : 

        article_id, e_id , prompt, label = p
        excerpt = excerpt_type_handler.excerpts[e_id]["excerpt"]
        subtype = excerpt_type_handler.excerpts[e_id]["macro_type"]
        print(f"Processing : {i+1 } | Test Split : {test_split}")

        # Predicting Type
        predicted_type = gpt4_prediction(prompt).strip()
        excerpt_type_handler.save_prediction(e_id, 'type', predicted_type)
        
        if predicted_type == label : macro_accuracy_counter += 1 
        print(f"TYPE : PREDICTED : {predicted_type} , GT : {label}")

        # Predicting Subtype
        if predicted_type == "macro" : 
            subtype_prompt = excerpt_subtype_handler.create_prompt(excerpt)
            predicted_subtype = gpt4_prediction(subtype_prompt).strip()            
        else : 
            predicted_subtype = 'did_not_predict_as_macro'
        
        excerpt_type_handler.save_prediction(e_id, 'subtype', 'did_not_predict_as_macro')
        if predicted_subtype == subtype : subtype_accuracy_counter += 1
        print(f"SUBTYPE : PREDICTED : {predicted_subtype} , GT : {subtype}")
        
        # Made this complex if else block
        # so that we don't make multiple calls to GPT. 

        # Predict Subtype in Oracle Setting
        if (label == "macro") and (predicted_type == label) : 
            # The excerpt is of type macro, and GPT already 
            # predicted ''macro'' correctly. 
            predicted_oracle_subtype = predicted_subtype

        elif (label == "macro") and (predicted_type != label) : 
            # The excerpt is of type macro, and GPT 
            # predicted ''macro'' INCORRECTLY. 
            subtype_prompt = excerpt_subtype_handler.create_prompt(excerpt)
            predicted_oracle_subtype = gpt4_prediction(subtype_prompt).strip()
            
        else : 
            # The excerpt is not of type macro, and GPT also did
            # not predict ''macro'' correctly. 
            predicted_oracle_subtype = 'is_not_of_type_macro'
        
        excerpt_type_handler.save_prediction(e_id, 'oracle_subtype', predicted_oracle_subtype)
        if predicted_oracle_subtype == subtype : oracle_subtype_accuracy_counter += 1
        print(f"SUBTYPE (oracle setting) : PREDICTED : {predicted_oracle_subtype} , GT : {subtype}")
            

        # Saving 
        if i%10 == 0 :

            print("Saving Test Split : " , test_split)
            print(f"Current type accuracy : {macro_accuracy_counter}/{i+1}")
            print(f"Current subtype accuracy : {subtype_accuracy_counter}/{i+1}")
            print(f"Current subtype accuracy (oracle setting) : {oracle_subtype_accuracy_counter}/{i+1}")

            with open(save_path, 'wb') as f : 
                pickle.dump(excerpt_type_handler.predicted_excerpts, f)

    with open(save_path, 'wb') as f : 
        pickle.dump(excerpt_type_handler.predicted_excerpts, f)
