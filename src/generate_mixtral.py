# from transformers import AutoModelForCausalLM, AutoTokenizer
#from ctransformers import AutoModelForCausalLM
#import torch
from pathlib import Path
from data_load import load_loop, map_filter, map_questions, load_documents
from data_retsinformation import load_retsinformation
from langchain.vectorstores import FAISS
from create_vector_db import prep_embeddings

from scipy.special import expit
import numpy as np

from llama_cpp import Llama

import os
#import torch

import subprocess
from tqdm import tqdm
import jsonlines

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from functools import partial
from pathlib import Path

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

def make_input_mixtral_askllm(system, score_name):
    prompt = """
    [INST]{system}[/INST]
    {score_name}: """
    return prompt.format(system=system, score_name=score_name)

def make_input_rag(question:str, db, full_docs:dict, k:int) -> str:
    docs = retrieve_prep_documents(question, db, full_docs, k)
    system = """
Du er en effektiv sprogmodel som hjælper professionelle medarbejdere i kommunen med at svare på faglige spørgsmål ud fra de dokumenter om regler og vejledninger du har til rådighed.
Du forstår fuldstændigt alle former for dansk, og svarer altid kun på kompetent dansk.
Giv først et kort og direkte svar på spørgsmålet. Derefter skal du forklare og begrunde svaret. Hvis du har citeret et dokument skal du henvise korrekt til din kilde.
Skriv dit svar som det ideelle svar til en professionel medarbejder i sundheds- og omsorgssektoren.
Hvis du ikke kan finde et relevant svar i dokumenterne, skal du forsøge at svare så godt du kan ud fra din professionelle baggrundsviden.
Hvis du ikke kender svaret, skal du sige "Jeg kender ikke svaret, vent venligst på et svar fra vores redaktører"
"""
    
    return f"""[INST] {system} [/INST]
Hvilke dokumenter er relevante for at besvare spørgsmålet?
[INST] Dokumenter: {docs} [/INST]
Okay, jeg er klar på at svare på dit spørgsmål
[INST] Spørgsmål: {question} [/INST]
"""

def retrieve_prep_documents(question:str, db, full_docs:dict, k=5) -> list:
    retrieved_documents = db.similarity_search_with_score(question, k=k)
    
    # get the titles and scores of the retrieved documents
    retrieved_titles = [(doc.metadata["title"], score) for doc, score in retrieved_documents]

    # get the full documents and keep the best score if the same document is retrieved multiple times
    docs = []
    for title, score in retrieved_titles:
        if title not in [doc[0] for doc in docs]: # if the title is not already in the list
            #save title, score and text
            docs.append((title, score, full_docs[title]))

        else: # if the title is already in the list
            # check if the current score is better than the one in the list
            for i, doc in enumerate(docs):
                if doc[0] == title:
                    if score < doc[1]:
                        docs[i] = (title, score, full_docs[title])

    
    documents = [f'Titel: {doc[0]} \nScore: {expit(np.log(1/doc[1])):.4f}, \nTekst: {doc[2]}' for doc in docs]
    documents = "\n".join(documents)
    
    return documents




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
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
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
    for i, question in tqdm(enumerate(map_questions(load_loop()[87:])), desc=f"Generating answers for {model_name}"):
        data = {}


        if not question:
            logging.debug(f"Question [{i}] not found")
            continue


        data["question"] = question
        data["prompt"] = make_input_func(question)
        data["answer"] = model(data["prompt"])
        # output_data.append(data)

        # save to json
        # with jsonlines.open('output.jsonl', mode='w') as writer:
        #     writer.write(...)
        with jsonlines.open(output_dir / f"{model_name}.json", "a") as f:
            f.write(data)

    logging.info(f"Finished generation for {model_name}")

if __name__ == '__main__':
    path = Path(__file__).parents[1]

    model = load_mixtral(model_path)
    # map_questions_save_generations("mixtral-no-prompt", model, make_input_mixtral_noprompt)


    RAG = True # change to false if you only want to run the above
    if RAG:
        # RAG model
        db_path = path / "data" / "vector_db"
        
        embeddings = prep_embeddings()

        # Load the vector store from disk
        db = FAISS.load_local(db_path, embeddings)

        # load documents for RAG model
        full_docs_loop = load_documents()
        full_docs_ri = load_retsinformation(paragraph=False)

        full_docs = full_docs_loop + full_docs_ri
        docs_dict = {doc.metadata["title"]: doc.page_content for doc in full_docs}

        partial_make_input_rag = partial(
            make_input_rag, 
            db = db,
            full_docs = docs_dict, 
            k=5 # number of documents to retrieve
            )
        
        map_questions_save_generations("mixtral-rag", model, partial_make_input_rag)


    # for doc in load_loop()[1:2]:
    #     text = doc['question']
    #     outputs = model(make_input_mixtral(text))


    # print(outputs)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # map_questions_save_generations("mixtral-no-prompt", model, make_input_mixtral_noprompt)
    # map_questions_save_generations("mixtral-simple-prompt", model, make_input_mixtral_okprompt)
