# from transformers import AutoModelForCausalLM, AutoTokenizer
#from ctransformers import AutoModelForCausalLM
#import torch
from pathlib import Path
from data_load import load_loop, map_filter

import os
#import torch

import subprocess

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



def make_input_mixtral(question: str) -> str:
    system = """Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kort og præcist på dansk, og giver dit bedste bud også selv om du er usikker.
    Hvis ikke du kender svaret, er det okay, og så siger du bare det.
    Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt.
    """

    prompt = f"""
    [INST] Prompt: {system} [/INST]
    Okay, jeg er klar på at svare på dit spørgsmål
    [INST] Spørgsmål: {question} [/INST]
    """

    return prompt





model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

root_path = Path.cwd()
if root_path.name == "src": root_path = root_path.parent

root_path = Path("/home/maltelau/cogsci") / "nlp" / "exam" / "aarhus_rag_eval"
model_path = root_path / "models" / "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"



def load_mixtral(model_path: Path = model_path):
    logging.info(f"Loading model {model_path}")
    def Mixtral(text: str) -> str:
        # logging.debug(f"Mixtral input:{text}")
        result = subprocess.run(["llamacpp",
                                 "-m", str(model_path),
                                 "-p", text,
                                 "-ngl", "6", # number of layers to offload to GPU
                                 "-s", "-1",  # random seed? -1 = random
                                 "-n", "512", # max tokens to produce
                                 "-t", "8",   # threads
                                 ], text=True, check=True, capture_output = True)

        logging.debug(f"Mixtral output:{result.stdout}")
        return result.stdout
    return Mixtral




if __name__ == '__main__':

    model = load_mixtral(model_path)
    for doc in load_loop()[:5]:
        text = doc['question']
        outputs = model(make_input_mixtral(text))
    # print(outputs)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
