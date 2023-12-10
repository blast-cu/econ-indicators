import pickle
import random

class LlamaChatPromptGenerator(object) :

    prompt = '''
Take a look at the following definitions of economic domains that newspaper excerpts cover.

[DEFINITIONS]

Based on the definitions, you can assign a label to a newspaper excerpt. For example:

[EXAMPLES]

Using the definitions of these economic indicators, what is the label for the following newspaper excerpt:

[EXCERPT_TO_PREDICT]
    '''

    label_definitions = '''
macro: macro-economic indicators aggregate statistics and data that reflect the economic\
circumstances of a particular country or region.

business: A business-specific data point is a data point associated with a particular firm or company.

industry: Industry level data points describe an entire industry rather than individual businesses

government: Any value that describes how a government earned or spent its income falls into this category.

personal: If data point focuses on the economic condition of a single person, or a group of individuals that \
    is not large enough to represent an entire demographic then we consider it personal
    '''

    def __init__(self, path_to_splits_dict:str="data/splits_dict", 
                 path_to_quant_dict:str="data/quant_dict",
                 path_to_save_predictions:str="data/preds_dict",
                 train_split_to_use_for_prompt_example:int=0) -> None:
        
        self.splits = pickle.load(open(path_to_splits_dict , "rb"))
        self.excerpts = pickle.load(open(path_to_quant_dict , "rb"))
        self.path_to_save_predictions = path_to_save_predictions
        self.predicted_excerpts = self.excerpts
        self.train_split_to_use_for_prompt_example = train_split_to_use_for_prompt_example
        self.examples_for_each_label = self.get_examples_for_each_label()

    def generator(self, test_split:int) : 

        excerpt_ids = self.splits[test_split]["test"]
        id2excerpts = self.get_id2excerpts(excerpt_ids)

        for article_id, excerpt_id in id2excerpts.items() : 
            for e_id in excerpt_id : 
                prompt = self.create_prompt(self.excerpts[e_id]['excerpt'])
                label = self.excerpts[e_id]['type']
                yield (article_id, e_id , prompt, label)

    def save_prediction(self, e_id, predicted_label) : 
        self.predicted_excerpts[e_id]['predicted_type'] = predicted_label

            

    def flush_predictions(self) : 
        with open(self.path_to_save_predictions, 'wb') as f : 
                pickle.dump(self.predicted_excerpts, f)


        
    def create_prompt(self, excerpt_to_predict) : 

        example_texts = []

        for excerpt_type in self.examples_for_each_label.keys() : 
            example_text = ""
            excerpt_example = random.sample(self.examples_for_each_label[excerpt_type] , 1)[0]

            example_text += f"Newspaper Excerpt: {excerpt_example}\n"
            example_text += f"Label: {excerpt_type}\n"

            example_texts.append(example_text)

        processed_prompt = self.prompt.replace("[DEFINITIONS]" , 
                                               self.label_definitions)
        
        processed_prompt = processed_prompt.replace("[EXAMPLES]" , "\n".join(example_texts))

        processed_prompt = processed_prompt.replace("[EXCERPT_TO_PREDICT]" , 
                                                    f"Newspaper Excerpt: {excerpt_to_predict}\nLabel:")

        return processed_prompt


    def get_examples_for_each_label(self) : 

        examples = {}
        sample_ids_for_examples = self.splits[self.train_split_to_use_for_prompt_example]["train"]
        sample_ids_2_excerpt_ids = self.get_id2excerpts(sample_ids_for_examples)
        excerpt_ids_for_examples = []

        for sample_id, excerpt_ids in sample_ids_2_excerpt_ids.items() : 
            excerpt_ids_for_examples.extend(excerpt_ids)
        
        excerpts_for_examples = {excerpt_id:self.excerpts[excerpt_id] for excerpt_id in excerpt_ids_for_examples}

        for excerpt_id, excerpt in excerpts_for_examples.items() : 

            excerpt_label = excerpt["type"]
            if excerpt_label not in examples : 

                examples[excerpt_label] = [excerpt["excerpt"]]
            else : 
                examples[excerpt_label].append(excerpt["excerpt"])

        return examples
    
    def get_id2excerpts(self, excerpt_ids) : 

        id2excerpts = {}

        for id in excerpt_ids : 
            id = str(id)
            for excerpt_sentence_id, excerpt_data in self.excerpts.items() : 
                if id == excerpt_sentence_id.split('_')[0] : 
                    if id not in id2excerpts : 
                        id2excerpts[id] = [excerpt_sentence_id]
                    else : 
                        id2excerpts[id].append(excerpt_sentence_id)

        return id2excerpts
    


            

        