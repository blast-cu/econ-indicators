from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    AutoModelForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import math
import os
import pickle
import sqlite3
import random
import argparse

import data_utils.get_annotation_stats as gs


def group_texts(examples, chunk_size=128):
    """
    Group the texts into chunks of a specified size.

    Args:
        examples (dict): A dictionary containing the input examples.

    Returns:
        dict: A dictionary containing the grouped texts.

    """

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

    def __init__(self, texts, tokenizer, chunk_size=128):

        # group list of texts into chunks w embeddings of size 128
        inputs = {k: [] for k in ['input_ids', 'attention_mask', 'word_ids']}
        for text in texts:
            inputs_ = tokenize_function(text, tokenizer)
            for k in inputs:
                inputs[k].append(inputs_[k])

        chunks = group_texts(inputs, chunk_size)

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


def load_dataset(db_filename: str, remove_labelled: bool = False):

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

    if remove_labelled:
        labelled_ids = list(pickle.load(open('data/clean/qual_dict', 'rb')).keys())
        query = 'SELECT id, text FROM article;'
        res = cur.execute(query)
        result = res.fetchall()
        text = [t[1] for t in result if t[0] not in labelled_ids]
    else:
        query = 'SELECT text FROM article;'
        res = cur.execute(query)
        text = [t[0] for t in res.fetchall()]

    text = [gs.extract_strings(t) for t in text]

    conn.close()

    return text


def main(args):
    """
    Main function for training and evaluating a masked language model using RoBERTa.

    Args:
        args: Command-line arguments passed to the script.
    """

    CHUNK_SIZE = int(args.s)
    if CHUNK_SIZE % 128 != 0:
        raise ValueError("Chunk size must be a multiple of 128")
    else:
        BATCH_SIZE = 64 // (CHUNK_SIZE // 128)

    OUT_DIR = args.o
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load pretrained model and tokenizer
    model_checkpoint = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint).to('cuda')
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
        tokenizer,
        chunk_size=CHUNK_SIZE
    )

    filename = os.path.join(OUT_DIR, f"train_dataset_{CHUNK_SIZE}")
    # train_dataset = pickle.load(open(filename, 'rb'))
    f = open(filename, 'wb')
    pickle.dump(train_dataset, f)


    print('>>> Tokenizing val texts')
    val_dataset = EconomicArticlesDatataset(
        val_texts,
        tokenizer,
        chunk_size=CHUNK_SIZE
    )

    filename = os.path.join(OUT_DIR, f"val_dataset_{CHUNK_SIZE}")
    # val_dataset = pickle.load(open(filename, 'rb'))
    f = open(filename, 'wb')
    pickle.dump(val_dataset, f)

    # create data collator for adding masks to input
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    print(len(train_dataset))
    print(len(val_dataset))

    logging_steps = len(train_dataset) // BATCH_SIZE

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--o", required=True, help="out directory")
    parser.add_argument("--c", default='roberta-base', help="checkpoint, default roberta-base")
    parser.add_argument("--s", default=128, help="chunk size, default 128")
    args = parser.parse_args()
    main(args)
