#!/usr/bin/env python3

from pathlib import Path
import dacy
from rouge_score import tokenizers
from tqdm import tqdm
import jsonlines
import json
import aspell

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class DacyTokenizer(tokenizers.Tokenizer):
    def __init__(self):
        self.nlp = dacy.load("large")

    def tokenize(self, text):
        return [str(x) for x in self.nlp.tokenizer(text)]


def add_ending_punctuation(text: str, punct: str = ".!?") -> str:
    if text.strip()[-1] not in punct:
        text += "."
    return text

def tokenize(text: str, tokenizer = DacyTokenizer()) -> list[str]:
    tokens = tokenizer.tokenize(text)
    return tokens


def lix(text, tokens):
    n_sentences = text.count(".") + text.count("?") + text.count("!")
    n_tokens = len(tokens)
    n_long_words = len([x for x in tokens if len(x) > 6])

    lix = (n_tokens / n_sentences) + (n_long_words * 100 / n_tokens)
    return lix


def spellcheck(tokens, spellchecker = aspell.Speller(("lang", "da"), ("run-together", "true"))):
    spellcheck_results = [spellchecker.check(str(t)) for t in tokens if t] # dont test empty strings
    n_spelling_errors = len([t for t in spellcheck_results if t]) # how many are true
    return n_spelling_errors / len(spellcheck_results)


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
        with jsonlines.open(gen_file) as f:
            logging.info(f"Tokenizing answers from {str(gen_file)}")
            # all generated answers
            generated_answers = [add_ending_punctuation(answer["answer"]) for answer in f]
            tokens = [tokenize(text) for text in tqdm(generated_answers)]


        logging.info(f"Calculating LIX for {str(gen_file)}")

        lix_results = {
            "lix_answer": [lix(text,token) for text,token in tqdm(zip(generated_answers, tokens))]
            }
        results[gen_file.stem] = lix_results


        logging.info(f"Counting spelling mistakes for {str(gen_file)}")

        spellcheck_results = {
            "spellcheck_answer": [spellcheck(token) for token in tqdm(tokens)]
            }
        results[gen_file.stem].update(spellcheck_results)


    # save to json
    with open(results_path / "readability.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
