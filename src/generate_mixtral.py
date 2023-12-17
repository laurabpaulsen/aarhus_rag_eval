# from transformers import AutoModelForCausalLM, AutoTokenizer
#from ctransformers import AutoModelForCausalLM
#import torch
from pathlib import Path
from data_load import load_loop, map_filter, map_questions

from llama_cpp import Llama

import os
#import torch

import subprocess
from tqdm import tqdm
import json

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



# def make_input_mixtral(question: str) -> str:

#     system = """
# Du er en sprogmodel som forstår og taler kompetent dansk.
# Du svarer kort og præcist på dansk, og giver dit bedste bud.
# Hvis du er usikker, skal du
# Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt.
# """

#     prompt = f"""
# [INST] Prompt: {system} [/INST]
# Okay, jeg er klar på at svare på dit spørgsmål
# [INST] Spørgsmål: {question} [/INST]
# """

#     return prompt


def make_input(question, system, prompt) -> str:
    return prompt.format(system=system, question=question)


def make_input_mixtral_noprompt(question):
    return make_input(question, "", "[INST]{question}[/INST]")

def make_input_mixtral_okprompt(question):

    system = """
Du er en effektiv sprogmodel som hjælper professionelle medarbejdere i kommunen med at svare på faglige spørgsmål ud fra de dokumenter om regler og vejledninger du har til rådighed.
Du forstår fuldstændigt alle former for dansk, og svarer altid kun på kompetent dansk.
Giv først et kort og direkte svar på spørgsmålet. Derefter skal du kort forklare og begrunde svaret.
Skriv dit svar som det ideelle svar til en professionel medarbejder i sundheds- og omsorgssektoren i Aarhus Kommune.
Hvis du ikke kan finde et relevant svar i dokumenterne, skal du forsøge at svare så godt du kan ud fra din professionelle baggrundsviden.
Hvis du ikke kender svaret, skal du sige "Jeg kender ikke svaret, vent venligst på et svar fra vores redaktører"
"""

    prompt = """
[INST]{system}[/INST]
Okay, jeg er klar på at svare på dit spørgsmål
[INST]Spørgsmål: {question}[/INST]
"""
    return make_input(question, system, prompt)


model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

root_path = Path.cwd()
if root_path.name == "src": root_path = root_path.parent

root_path = Path("/home/maltelau/cogsci") / "nlp" / "exam" / "aarhus_rag_eval"
model_path = root_path / "models" / "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
output_dir = root_path / "generated"


def load_mixtral(model_path: Path = model_path):
    logging.info(f"Loading model {model_path}")
    llm = Llama(
        model_path=str(model_path),  # Download the model file first
        n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=6         # The number of layers to offload to GPU, if you have GPU acceleration available
    )
    def text_only(prompt):
        return llm(prompt, max_tokens = 4000)['choices'][0]['text']
    return text_only



# def load_mixtral(model_path: Path = model_path):
#     logging.info(f"Loading model {model_path}")
#     def Mixtral(text: str) -> str:
#         # logging.debug(f"Mixtral input:{text}")
#         result = subprocess.run(["llamacpp",
#                                  "-m", str(model_path),
#                                  "-p", f'"{text}"',
#                                  "-ngl", "6", # number of layers to offload to GPU
#                                  "-s", "-1",  # random seed? -1 = random
#                                  "-n", "512", # max tokens to produce
#                                  "-t", "8",   # threads
#                                  ], text=True, check=True, capture_output = True)

#         logging.debug(f"Mixtral output:{result.stdout}")
#         return result.stdout
#     return Mixtral


def map_questions_save_generations(model_name: str, model: object, make_input_func: object):
    logging.info(f"Starting generation for {model_name}")
    # output_data = []
    for i, question in tqdm(enumerate(map_questions(load_loop())), desc=f"Generating answers for {model_name}"):
        data = {}


        if not question:
            logging.debug(f"Question [{i}] not found")
            continue


        data["question"] = question
        data["prompt"] = make_input_func(question)
        data["answer"] = model(data["prompt"])
        # output_data.append(data)

        # save to json
        with open(output_dir / f"{model_name}.json", "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info(f"Finished generation for {model_name}")

if __name__ == '__main__':


    model = load_mixtral(model_path)
    # map_questions_save_generations("mixtral-no-prompt", model, make_input_mixtral_noprompt)
    # map_questions_save_generations("mixtral-simple-prompt", model, make_input_mixtral_okprompt)
