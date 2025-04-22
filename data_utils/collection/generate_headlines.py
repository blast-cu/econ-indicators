import numpy as np
import json
import argparse
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from tqdm import tqdm
from torch.utils import data
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(data.Dataset):
    def __init__(self, input_ids, attention_masks, article_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.article_ids = article_ids

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        input_ids_i = self.input_ids[index]
        attention_mask_i = self.attention_masks[index]
        article_id_i = self.article_ids[index]
        return input_ids_i, attention_mask_i, article_id_i


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
    tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
    model = model.to(device)

    in_path = args.dataset

    for publisher in os.listdir(in_path):

        pub_path = os.path.join(in_path, publisher)
        article_json_path = os.path.join(pub_path, 'articles.json')

        # if articles_gen_headlines.json.json already exists, skip this publisher
        if os.path.exists(os.path.join(pub_path, 'articles_gen_headlines.json')):
            logger.info(f"Skipping publisher '{publisher}' as 'articles_gen_headlines.json' already exists.")
            continue
            
        logger.info(f"Processing publisher '{publisher}'...")

        # Load article texts
        dataset = json.load(open(article_json_path))
        for art, _ in enumerate(dataset):
            if type(dataset[art]['headline']) is float:
                dataset[art]['headline'] = None

        pbar = tqdm(total=len(dataset), desc='tokenizing')

        input_ids_batch = []
        attention_mask_batch = []
        articles_batch = []

        for art, _ in enumerate(dataset):
            if (dataset[art]['headline'] is None or len(dataset[art]['headline'].split()) <= 4) and 'gen-headline' not in dataset[art]:
                article = dataset[art]['text']
                text = "headline: " + article
                max_len = 1024
                encoding = tokenizer(text, return_tensors = "pt", max_length=max_len, truncation=True, padding='max_length')
                input_ids = encoding["input_ids"].to(device)
                attention_masks = encoding["attention_mask"].to(device)
                input_ids_batch.append(input_ids)
                attention_mask_batch.append(attention_masks)
                articles_batch.append(art)
            pbar.update(1)
        pbar.close()

        input_ids = torch.vstack(input_ids_batch)
        attention_masks = torch.vstack(attention_mask_batch)

        # print("input_ids", input_ids.shape)
        # print("attention_masks", attention_masks.shape)

        infer_dataset = TextDataset(input_ids, attention_masks, articles_batch)
        infer_dataloader = data.DataLoader(infer_dataset, batch_size=16)

        pbar = tqdm(total=len(infer_dataloader), desc='generating headlines')
        for input_ids_b, attention_masks_b, article_ids_b in infer_dataloader:

            beam_outputs = model.generate(
                input_ids=input_ids_b,
                attention_mask=attention_masks_b,
                max_length=64,
                num_beams=3,
                early_stopping=True,
            )

            # print(beam_outputs.shape, len(article_ids_b))

            for art, beam_output in zip(article_ids_b, beam_outputs):
                # decode and clean up result.
                result = tokenizer.decode(beam_output)
                headline = result.replace('<pad>', '')
                headline = headline.replace('</s>', '')
                headline = headline.strip()

                dataset[art]['gen-headline'] = headline

            pbar.update(1)
        pbar.close()

        out_path = args.dataset.replace('.json', '_gen_headlines.json')
        with open(out_path, 'w') as fp:
            json.dump(dataset, fp, indent=4)
        logger.info(f"Saved generated headlines to '{out_path}' for publisher '{publisher}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    main(args)
