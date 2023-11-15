import fire
import sys
sys.path.insert(0, "/home/aleto/research/llama_temp/")
from llama import Llama
from typing import List
import random

import models.llama_classifier.shared as shared
import models.utils.dataset as d


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """
    sys.path.insert(0, '/home/aleto/research/llama/')

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    texts, labels = d.load_qual_dataset(
        db_filename='/home/aleto/research/econ_indicators/data/data.db',
        annotation_component ='frame'
    )

    # Randomly sample 5 texts and labels for testing
    zipped = list(zip(texts, labels))

    random.seed(42)
    random.shuffle(zipped)

    zipped = zipped[:5]
    texts = [z[0] for z in zipped]
    labels = [z[1] for z in zipped]

    # pre_prompt = 'This is a multiple choice question. Answer with a single letter. News article about the economy: \n'
    pre_prompt = '<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.'
    post_prompt = '\n\nIs the above article talking about the economy in the context of \n A. business' + \
        ' \n B. industry \n C. macro \n D. government \n E. other'
    
    prompts = shared.form_prompts(texts, pre_prompt, post_prompt)
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
