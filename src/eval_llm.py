
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
    assert documents is not None and proposed is not None
    # logging.debug(f"Creating faithfullness prompt:{documents[:100]} {proposed[:100]}")
    return f"""
    Please give a faithfulness score indicating how well the following answer is supported by the following documents.

    The documents are:
    {documents}

    The answer is:
    {proposed}

    Please give a score between 0 and 10, where 0 means that the answer is not supported at all and 10 means that the answer is fully supported by the documents. Do not explain, just write the score.
    """


def create_input_correctness(reference = None, proposed = None, **kwargs):
    if proposed is not None and reference is None:
        return None

    assert reference is not None and proposed is not None

    return f"""
    Please give a correctness score indicating how good the proposed answer is given the true gold standard answer.
    The gold standard answer has been verified to be correct by humans.

    True gold standard answer: {reference}

    Proposed answer: {proposed}

    Please give a score between 0 and 10, where 0 means that the proposed answer is not supported at all and 10 means that the answer is fully supported by the gold standard. Do not explain, just write the score.
    """

def create_input_informativeness_proposed(question = None, proposed = None, **kwargs):
    assert question is not None and proposed is not None

    return f"""
    Please give an informativeness score indicating how well the answer is related to the question. Has everything in the question been answered, and is the entire answer working towards answering the question?

    Question: {question}

    Answer: {proposed}

    Please give a score between 0 and 10, where 0 means the question has not been answered at all, and 10 means the question has been answered perfectly. Do not explain, just write the score.
    """


def create_input_readability_proposed(proposed = None, **kwargs):
    assert answer is not None

    return f"""
    Please give a readability score indicating the quality of the language used in the given answer. A good answer is short, precise, uses simple language to express its ideas, and is well structured to aid human understanding.

    Answer: {answer}

    Please give a score between 0 and 10 where 0 means its very badly worded, and 10 is the easiest and best sentence one could ever read. Do not explain, just write the score.
    """

def create_input_readability2_proposed(proposed = None, **kwargs):
    assert proposed is not None

    return f"""
    Please give a language score indicating that the answer is in Danish, and it is written in correct danish. A good answer uses no other languages, and has no spelling mistakes or grammatical errors.

    Answer: {proposed}

    Please give a score between 0 and 10 where 0 means the answer is in a different language from Danish or that it is full of language errors, and 10 means it is written in perfect Danish. Do not explain, just write the score.
    """

def create_input_knowledgeaccess(question = None, documents = None, **kwargs):
    assert question is not None and documents is not None

    return f"""
    Please give a knowledge access score indicating how appropriate the knowledge given in the documents is for answering the question.

    Question: {question}

    Documents:
    {documents}

    Please give a score between 0 and 10 where 0 means there is no relevant knowledge in the documents related to the question, and 10 means there is a very clear link from the knowledge in the documents to the question. Do not explain, just write the score.
    """

def create_input_alignment(proposed = None, **kwargs):
    assert proposed is not None

    return f"""
    Please give an alignment score indicating how appropriate the language use in the answer is for a professional context in a Danish Municipality.
    Aarhus kommune is a high-trust, flat-hierarchy organization with high professional standards for the quality of service provided. The answer has to be aligned with the values of the organization, but will not be visible to any external users or partners.

    Answer: {proposed}

    Please give a score between 0 and 10 where 0 means the answer is completely inappropriate, and 10 mean sthe answer fits perfectly in the context. Do not explain, just write the score.
    """

def create_input_metacognition(proposed = None, **kwargs):
    assert proposed is not None

    return f"""
    Please give a metacognition score indicating how much confidence the answer has in itself. A high score means the answer is very certain.

    Answer: {proposed}

    Please give a score between 0 and 10 where 0 means the answer has no confidence whatsoever, and 10 means it has 100% confidence. Do not explain, just write the score.
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
            if system is None:
                tmp_data[score_name] = None
                continue
            input_mdl = make_input_mixtral_askllm(system, score_name.split("_")[0])
            score = model(input_mdl)
            tmp_data[score_name] = score

        scores = pd.concat([scores, pd.DataFrame.from_dict(tmp_data, orient="index").T])

        if savepath:
            scores.to_csv(savepath, index = False)

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

    loop_answers   = map_filter(jsondata, "response")
    loop_questions = map_filter(jsondata, "question")

    # CHANGE TO MIXTRAL POTENTIALLY
    model = load_mixtral(grammar = root_dir / "score_0_10.gbnf")

    with jsonlines.open(gen_file) as f:
        generated_answers = list(f)


    documents = [extract_docs_from_prompt(answer["prompt"]) for answer in generated_answers]
    generated_answers = [answer["answer"] for answer in generated_answers]


    # testing on the first - REMEBER TO REMOVE [0]
    get_scores(
        model = model, 
        reference = loop_answers[75:78],
        proposed = generated_answers[75:78],
        question = loop_questions[75:78],
        documents = documents[75:78],
        input_funcs = [create_input_faithful, create_input_correctness, create_input_informativeness_proposed],
        score_names = ["faithfulness", "correctness", "informativeness"],
        # input_funcs = [create_input_metacognition],
        # score_names = ["confidence"],
        savepath = results_path / f"{gen_file.stem}_llm.csv"
        )



            



