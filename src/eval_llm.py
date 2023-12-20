
from generate_mistral import load_mistral
from generate_mixtral import load_mixtral, make_input_mixtral_askllm
from data_load import load_loop, map_filter
import jsonlines
from pathlib import Path
import pandas as pd
import logging

import re

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def create_input_faithful(documents = None, proposed = None, **kwargs):
    """
    Creates the input for the language model used when evaluating faithfulness.
    """
    # logging.debug(f"Creating faithfullness prompt:{documents[:100]} {proposed[:100]}")
    return f"""
    Please give a faithfullness score indicating how well the following answer is supported by the following documents.

    The documents are:
    {documents}

    The answer is:
    {proposed}

    Please give a score between 0 and 10, where 0 means that the answer is not supported at all and 10 means that the answer is fully supported by the documents. Do not explain, just write the score.
    """


def create_input_correct(**kwargs):
    pass

def create_input_informativeness(question, answer):
    pass

def remove_document_scores(documents: str) -> str:
    regex = "\nScore:[ ]*\d+(?:\.\d+)?,?[ ]*\n"
    return re.compile(regex).sub("", documents)


def get_scores(
        model, 
        reference, 
        proposed, 
        question, 
        documents, 
        input_funcs: list[callable],
        score_names: list[str],
        savepath = None):
    """
    Uses a large language model to score the the generated answers against the reference answers taking the question into account.
    """
    scores = pd.DataFrame()
    for ref, prop, q, d in zip(reference, proposed, question, documents):
        tmp_data = {
            "question": q,
            "reference": ref,
            "proposed": prop,
            "documents": d
        }

        #logging.debug(f"Testing combination {tmp_data}")

        for input_func, score_name in zip(input_funcs, score_names):
            system = input_func(**tmp_data)
            input_mdl = make_input_mixtral_askllm(system, score_name)
            score = model(input_mdl)
            tmp_data[score_name] = score

        scores = pd.concat([scores, pd.DataFrame.from_dict(tmp_data, orient="index").T])

        if savepath:
            scores.to_csv(savepath)

def extract_docs_from_prompt(prompt):
    """
    Extracts the documents from the prompt.
    """
    # get location of second "[INST]"
    inst_loc = prompt.find("[INST]", prompt.find("[INST]") + 1)
    # get location of second "[/INST]"
    inst_end_loc = prompt.find("[/INST]", prompt.find("[/INST]") + 1)
    
    documents = prompt[inst_loc:inst_end_loc]
    documents = documents.replace("[INST]", "").replace("[/INST]", "")

    # filter out document scores from vector similarity
    documents = remove_document_scores(documents)

    return documents

if __name__ in "__main___":
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "generated"
    gen_file = generated_path / "mixtral-rag.jsonl"

    loop_answers = [answer for answer in map_filter(jsondata, "response") if answer is not None]
    loop_questions = [question for question, answer in zip(map_filter(jsondata, "question"), map_filter(jsondata, "response")) if answer is not None]

    # CHANGE TO MIXTRAL POTENTIALLY
    model = load_mixtral(grammar = root_dir / "score_0_10.gbnf")

    with jsonlines.open(gen_file) as f:
        generated_answers = list(f)


    documents = [extract_docs_from_prompt(answer["prompt"]) for answer in generated_answers]
    generated_answers = [answer["answer"] for answer in generated_answers]


    # testing on the first - REMEBER TO REMOVE [0]
    get_scores(
        model = model, 
        reference = loop_answers[:3],
        proposed = generated_answers[:3],
        question = loop_questions[:3],
        documents = documents[:3],
        input_funcs = [create_input_faithful],
        score_names = ["faithfulness"],
        savepath = results_path / f"{gen_file.stem}_llm.csv"
        )



            



