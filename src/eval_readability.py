#!/usr/bin/env python3

import logging
from pathlib import Path
import dacy
from rouge_score import tokenizers
from tqdm import tqdm
import jsonlines
import json



class DacyTokenizer(tokenizers.Tokenizer):
    def __init__(self):
        self.nlp = dacy.load("large")

    def tokenize(self, text):
        return [x for x in self.nlp(text)]


def lix(text, tokenizer = DacyTokenizer()):
    if text.strip()[-1] not in ".!?":
        text += "."
    tokens = tokenizer.tokenize(text)
    n_sentences = text.count(".") + text.count("?") + text.count("!")
    n_tokens = len(tokens)
    n_long_words = len([x for x in tokens if len(x) > 6])

    lix = (n_tokens / n_sentences) + (n_long_words * 100 / n_tokens)
    return lix


# def spellcheck(text):
#     spellcheck_results = None
#     n_spelling_errors = len(spellcheck_results)
#     return n_spelling_errors


if __name__ in "__main__":
    # jsondata = load_loop()
    root_dir = Path(__file__).parents[1]

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "generated"

    # all files with generated answers
    files = [f for f in generated_path.iterdir() if f.suffix == ".jsonl"]



    results = {}

    for gen_file in files:
        logging.info(f"Calculating LIX for {str(gen_file)}")
        with jsonlines.open(gen_file) as f:
            # all generated answers
            generated_answers = [answer["answer"] for answer in f]

        lix_results = {
            "lix_answer": [lix(text) for text in tqdm(generated_answers)]
            }
        results[gen_file.stem] = lix_results


    # save to json
    with open(results_path / "readability.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
