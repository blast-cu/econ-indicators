from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    AutoModelForMaskedLM, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import math
import random
# import collections
# import numpy as np

# from data_utils.get_annotation_stats import get_all_text
import sqlite3
import re

def extract_strings(dirty_str: str):
    clean = re.sub('<[^>]+>', '', dirty_str)
    return clean

def get_all_text(db_filename: str, clean: bool = True):
    """
    Retrieves the text data from the specified SQLite database file.

    Parameters:
    - db_filename (str): The path to the SQLite database file.
    - clean (bool): Flag indicating whether to clean the text data. Default is True.

    Returns:
    - text (list): The list of text data retrieved from the database.
    """
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()

    query = 'SELECT text FROM article;'
    res = cur.execute(query)
    text = [t[0] for t in res.fetchall()]

    if clean:
        text = [extract_strings(t) for t in text]

    conn.close()
    return text 


def group_texts(examples):
    """
    Group the texts into chunks of a specified size.

    Args:
        examples (dict): A dictionary containing the input examples.

    Returns:
        dict: A dictionary containing the grouped texts.

    """
    chunk_size = 128

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # Split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    print(f'Created {len(result["input_ids"])} chunks of size {chunk_size}')

    return result


def tokenize_function(text, tokenizer):
    """
    Tokenizes the given text using the provided tokenizer.

    Args:
        text (str): The input text to be tokenized.
        tokenizer: The tokenizer object to be used for tokenization.

    Returns:
        dict: A dictionary containing the tokenized output, including the word IDs.
    """
    result = tokenizer(text)
    result['word_ids'] = result.word_ids()
    return result


class EconomicArticlesDatataset(Dataset):
    """
    Dataset class for economic articles.

    Args:
        texts (list): List of texts.
        tokenizer: Tokenizer object.

    Attributes:
        input_ids (list): List of input IDs.
        attention_mask (list): List of attention masks.
        word_ids (list): List of word IDs.
        labels (list): List of labels.
    """

    def __init__(self, texts, tokenizer):

        # group list of texts into chunks w embeddings of size 128
        inputs = {k: [] for k in ['input_ids', 'attention_mask', 'word_ids']}
        for text in texts:
            inputs_ = tokenize_function(text, tokenizer)
            for k in inputs:
                inputs[k].append(inputs_[k])

        chunks = group_texts(inputs)

        self.input_ids = chunks['input_ids']
        self.attention_mask = chunks['attention_mask']
        self.word_ids = chunks['word_ids']
        self.labels = chunks['labels']

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):

        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'word_ids': self.word_ids[idx],
            'labels': self.labels[idx]
        }


# def whole_word_masking_data_collator(features):
#     tokenizer = tokenize_function
#     wwm_probability = 0.2
#     for feature in features:
#         word_ids = feature.pop("word_ids")

#         # Create a map between words and corresponding token indices
#         mapping = collections.defaultdict(list)
#         current_word_index = -1
#         current_word = None
#         for idx, word_id in enumerate(word_ids):
#             if word_id is not None:
#                 if word_id != current_word:
#                     current_word = word_id
#                     current_word_index += 1
#                 mapping[current_word_index].append(idx)

#         # Randomly mask words
#         mask = np.random.binomial(1, wwm_probability, (len(mapping),))
#         input_ids = feature["input_ids"]
#         labels = feature["labels"]
#         new_labels = [-100] * len(labels)
#         for word_id in np.where(mask)[0]:
#             word_id = word_id.item()
#             for idx in mapping[word_id]:
#                 new_labels[idx] = labels[idx]
#                 input_ids[idx] = tokenizer.mask_token_id
#         feature["labels"] = new_labels

#     return default_data_collator(features)

def load_dataset(db_filename: str):
    """
    Load dataset from a database file.

    Args:
    - db_filename (str): The filename of the database.

    Returns:
    - list: A list of strings representing the loaded articles from the db.
    """

    text = get_all_text(db_filename, clean=True)  # list of strings
    return text


# def main(args):
"""
Main function for training and evaluating a masked language model using RoBERTa.

Args:
    args: Command-line arguments passed to the script.
"""

# Load pretrained model and tokenizer
model_checkpoint = "roberta-base"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Get list of all articles in db, split into train and val
db_filename = "data/data.db"
texts = load_dataset(db_filename)
# texts = random.sample(texts, 10)
train_texts, val_texts = train_test_split(
    texts,
    test_size=0.15,
    random_state=42
)

print(f'>>> Loaded {len(train_texts)} training texts')
print(f'>>> Loaded {len(val_texts)} validation texts')

print('>>> Tokenizing train texts')
train_dataset = EconomicArticlesDatataset(
    train_texts,
    tokenizer
)
print('>>> Tokenizing val texts')
val_dataset = EconomicArticlesDatataset(
    val_texts,
    tokenizer
)

# create data collator for adding masks to input
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
)

batch_size = 64
logging_steps = len(train_dataset) // batch_size

training_args = TrainingArguments(
    output_dir=f"models/roberta_classifier/{model_checkpoint}-finetuned-masked-2",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

eval_results = trainer.evaluate()
print(f">>> Initial Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

eval_results = trainer.evaluate()
print(f">>> Final Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.save_model()
# trainer.save_metrics()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--db", required=True, help="path to database")
#     args = parser.parse_args()
#     main(args)
