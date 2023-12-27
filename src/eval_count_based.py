from pathlib import Path
from data_load import load_loop, map_filter
import json 
import dacy
from rouge_score import rouge_scorer
import jsonlines
from tqdm import tqdm
import pandas as pd
import re

def remove_document_scores(documents: str) -> str:
    regex = "\nScore:[ ]*\d+(?:\.\d+)?,?[ ]*\n"
    return re.compile(regex).sub("", documents)


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

def calculate_NER_overlap(reference:str, candidate:str, nlp:object):
    """
    Calculates the NER tag overlap between two texts based on the named entities.

    Parameters
    ----------
    text1 : str
        The first text.
    text2 : str
        The second text.
    nlp : object
        The spaCy model to use for NER tagging.
    """

    nlp1 = nlp(reference)
    nlp2 = nlp(candidate)

    entities_ref = [ent.text for ent in nlp1.ents]
    entities_can = [ent.text for ent in nlp2.ents]

    # compare the entities
    intersection_ner = set(entities_ref).intersection(set(entities_can))

    recall = len(intersection_ner) / max(len(entities_ref), 1)
    precision = len(intersection_ner) / max(len(entities_can), 1)

    return recall, precision
 

def get_all_scores(reference:list, candidate:list, nlp, scorer, savepath:Path = None) -> tuple:
    """
    Returns the average NER overlap, ROUGE-L and ROUGE-1 for pairs of texts.

    Parameters
    ----------
    texts1 : list of str
        First list of texts. 
    texts2 : list of str
        Second list of texts.
    nlp : object
        The spaCy model to use for NER tagging.
    scorer : object
        The rouge scorer.
    savepath : Path
        The path to save the individual scores to.
    """
    # overall results for all texts
    results = {}

    # individual scores for each pair of texts
    scores = pd.DataFrame()

    for ref, can in tqdm(zip(reference, candidate)):
        # check that non of the texts are none
        if ref is None or can is None:
            continue
        NER_recall, NER_precision = calculate_NER_overlap(ref, can, nlp)
        rouge = scorer.score(ref, can)

        tmp_dat = pd.DataFrame.from_dict({"reference": [ref], "candidate": [can], "ner_recall": [NER_recall], "ner_precision": [NER_precision], "rouge_l": [rouge["rougeL"].recall], "rouge_1": [rouge["rouge1"].recall], "rouge_l_precision": [rouge["rougeL"].precision], "rouge_1_precision": [rouge["rouge1"].precision]})
        scores = pd.concat([scores, tmp_dat], ignore_index=True)
        

    results["ner_recall"] = scores["ner_recall"].mean()
    results["ner_precision"] = scores["ner_precision"].mean()
    results["rouge_l"] = scores["rouge_l"].mean()
    results["rouge_1"] = scores["rouge_1"].mean()
    results["rouge_l_precision"] = scores["rouge_l_precision"].mean()
    results["rouge_1_precision"] = scores["rouge_1_precision"].mean()

    if savepath:
        scores.to_csv(savepath, index=False)

    return results


if __name__ in "__main__":
    
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "generated"

    # all files with generated answers
    files = [f for f in generated_path.iterdir() if f.suffix == ".jsonl"]

    # model for NER overlap
    nlp = dacy.load("large")
    nlp.add_pipe("dacy/ner-fine-grained", config={"size": "large"})

    # model for ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    loop_answers = map_filter(jsondata, "response")
    loop_questions = map_filter(jsondata, "question")

    results = {}
    
    for gen_file in files:
        with jsonlines.open(generated_path / gen_file) as f:
            # all generated answers
            generated_answers = list(f)

        model_response = [answer["answer"] for answer in generated_answers]

        gold_to_response_results = {
            "gold_to_response": get_all_scores(loop_answers, model_response, nlp, scorer, results_path / f"{gen_file.stem}_gold_to_response.csv")
            }
        results[gen_file.stem] = gold_to_response_results

        question_to_response_results = {
            "question_to_response": get_all_scores(loop_questions, model_response, nlp, scorer, results_path / f"{gen_file.stem}_question_to_response.csv")
        }
        results[gen_file.stem].update(question_to_response_results)

        question_to_gold_results = {
            "question_to_gold": get_all_scores(loop_questions, loop_answers, nlp, scorer, results_path / f"{gen_file.stem}_question_to_gold.csv")
        }
        results[gen_file.stem].update(question_to_gold_results)

        if "rag" in gen_file.stem:
            documents = [extract_docs_from_prompt(answer["prompt"]) for answer in generated_answers]

            document_to_response_results = {
                "document_to_response": get_all_scores(documents, model_response, nlp, scorer, results_path / f"{gen_file.stem}_document_to_response.csv")
            }
            results[gen_file.stem].update(document_to_response_results)

            document_to_gold_results = {
                "document_to_gold": get_all_scores(documents, loop_answers, nlp, scorer, results_path / f"{gen_file.stem}_document_to_gold.csv")
            }
            results[gen_file.stem].update(document_to_gold_results)

    # save to json
    with open(results_path / "count_based.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)





            


