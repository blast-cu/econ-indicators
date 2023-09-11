import numpy as np
import json
import argparse
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from tqdm import tqdm
from torch.utils import data


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

    # Load article texts
    dataset = json.load(open(args.dataset))
    pbar = tqdm(total=len(dataset), desc='tokenizing')

    input_ids_batch = []
    attention_mask_batch = []
    articles_batch = []

    for art in dataset:
        if (dataset[art]['headline'] is None or len(dataset[art]['headline'].split()) <= 4) and 'gen-headline' not in dataset[art]:
            article = dataset[art]['text']
            text =  "headline: " + article

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

    print("input_ids", input_ids.shape)
    print("attention_masks", attention_masks.shape)

    infer_dataset = TextDataset(input_ids, attention_masks, articles_batch)
    infer_dataloader = data.DataLoader(infer_dataset, batch_size=16)
     
    pbar = tqdm(total=len(infer_dataloader), desc='generating headlines')
    for input_ids_b, attention_masks_b, article_ids_b in infer_dataloader:

        beam_outputs = model.generate(
            input_ids = input_ids_b,
            attention_mask = attention_masks_b,
            max_length = 64,
            num_beams = 3,
            early_stopping = True,
        )
      
        #print(beam_outputs.shape, len(article_ids_b))

        for art, beam_output in zip(article_ids_b, beam_outputs):
            result = tokenizer.decode(beam_output)
            dataset[art]['gen-headline'] = result
        
        pbar.update(1)
    pbar.close()

    with open(args.dataset, 'w') as fp:
        json.dump(dataset, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    main(args)
