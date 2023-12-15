from models.llama_chat.data_handler import ExcerptTypePromptHandler


class ExcerptSubTypePromptHandler(ExcerptTypePromptHandler) : 

    prompt = '''
Take a look at the following definitions of macro-economic categories that newspaper excerpts cover.

[DEFINITIONS]

Based on the definitions, you can assign a label to a newspaper excerpt. For example:

[EXAMPLES]
Using the definitions of these macro-economic categories, what is the label for the following newspaper excerpt:

[EXCERPT_TO_PREDICT]
    '''

    label_definitions = '''
jobs: Numbers describing the state of the job market.\
These numbers are focused on employment and payroll statistics. \
In particular, this category tracks what percentage of the workforce is currently \
(un)employed and how many jobs are added/lost.

market: This category contains numbers that describe the state of various “markets” \
around the globe. For example, numbers describing the state of the S&P 500 would fall into \
this category, because they track overall market activity in the United States. Other \
values that could fall into this category include global commodity prices and \
national debt prices. 

housing: Numbers describing the state of the housing market including the  average cost of a new home, the \
number of new homes being built, and the volume of home sales during a given period

macro: This category tracks the aggregate size of the economy, and the wealth of its residents. \
GDP and GDP per capita fall into this category.

    
wages: Metrics that track wages fall into this category. The most common quantity here is inflation \
adjusted wage  growth, but other measures that track shifts in wages (at a national level) should 
be included here.

prices: This category includes metrics that track how the prices of goods and the cost of living has shifted over \
time. For the most part, the metrics in this category are different ways to monitor inflation.


retail: Tracks national level retail sales. This measures how much people are purchasing, \
and whether the amount people are purchasing is going up or down.


interest: The rates set by institutions like the federal reserve. These rates determine the cost of borrowing.


currency: The prices of currencies on the open market, and exchange rates between currencies.


energy: Quantities that describe the cost of energy. The prices of a barrel of crude oil, or a \
cubic meter of  LNG  both fall into this category. Renewable/green energy prices could also \
be in this category.

none: If none of the above categories apply, or this is not a macro-economic indicator, it can be marked as none.
    '''

    acceptable_labels = ['interest', 'prices', 'macro', 'market', 
                         'none', 'jobs', 'wages', 'energy', 
                         'retail', 'other', 'currency', 'housing']

    def __init__(self, 
                 path_to_splits_dict: str = "data/splits_dict", 
                 path_to_quant_dict: str = "data/quant_dict", 
                 path_to_save_predictions: str = "data/preds_dict", 
                 train_split_to_use_for_prompt_example: int = 0) -> None:
        
        super().__init__(path_to_splits_dict, 
                         path_to_quant_dict, 
                         path_to_save_predictions, 
                         train_split_to_use_for_prompt_example)

        self.examples_for_each_label = self.get_examples_for_each_label()
        # print(self.examples_for_each_label)

    def save_prediction(self, e_id, predicted_label) : 
        self.predicted_excerpts[e_id]['predicted_subtype'] = predicted_label
        
        
        
    def get_examples_for_each_label(self) : 

        examples = {}
        sample_ids_for_examples = self.splits[self.train_split_to_use_for_prompt_example]["train"]
        sample_ids_2_excerpt_ids = self.get_id2excerpts(sample_ids_for_examples)
        excerpt_ids_for_examples = []

        for sample_id, excerpt_ids in sample_ids_2_excerpt_ids.items() : 
            excerpt_ids_for_examples.extend(excerpt_ids)
        
        excerpts_for_examples = {excerpt_id:self.excerpts[excerpt_id] for excerpt_id in excerpt_ids_for_examples}

        for excerpt_id, excerpt in excerpts_for_examples.items() : 

            if excerpt["type"] == "macro" : 
                excerpt_label = excerpt["macro_type"]
                if excerpt_label in self.acceptable_labels :
                    if excerpt_label not in examples :  

                        examples[excerpt_label] = [excerpt["excerpt"]]
                    else : 
                        examples[excerpt_label].append(excerpt["excerpt"])

            if len(examples) == len(self.acceptable_labels) : 
                return examples

        return examples
            
        
if __name__ == "__main__" : 

    subtype_prompt = ExcerptSubTypePromptHandler()

    for p in subtype_prompt.generator(0) :
        print(p[2])
        break