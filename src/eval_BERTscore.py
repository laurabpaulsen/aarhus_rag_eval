from pathlib import Path
from data_load import load_loop, map_filter
import json 
import jsonlines
import pandas as pd
from eval_count_based import extract_docs_from_prompt
from tqdm import tqdm

# BERTscore
from bert_score import BERTScorer


def get_all_scores(reference:list, candidate:list, scorer, savepath:Path = None) -> tuple:
    """
    Returns the average NER overlap, ROUGE-L and ROUGE-1 for pairs of texts.

    Parameters
    ----------
    reference : list of str
        First list of texts. 
    candidate : list of str
        Second list of texts.
    scorer : object
        The BERT scorer to use.
    savepath : Path
        The path to save the individual scores to.
    """
    scores = pd.DataFrame()
    

    
    for ref, can in tqdm(zip(reference, candidate)):
        # check that non of the texts are none
        if ref is None or can is None:
            continue

        else:
            P, R, F1 = scorer.score([ref], [can])
            tmp_dat = pd.DataFrame.from_dict({"reference": [ref], "candidate": [can], "BERTscore_precision": [P.item()], "BERTscore_recall": [R.item()], "BERTscore_f1": [F1.item()]})
            scores = pd.concat([scores, tmp_dat], ignore_index=True)


    
    # overall results for all texts
    results = {
        "BERTscore_precision": scores["BERTscore_precision"].mean(),
        "BERTscore_recall": scores["BERTscore_recall"].mean(),
        "BERTscore_f1": scores["BERTscore_f1"].mean()
    }

    if savepath:
        scores.to_csv(savepath, index=False)

    return results


if __name__ in "__main__":


    root_dir = Path(__file__).parents[1]
    jsondata = load_loop(root_dir / "data" /"loop_q_and_a_w_meta.json")

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "generated"

    loop_answers = map_filter(jsondata, "response")
    loop_questions = map_filter(jsondata, "question")

    scorer = BERTScorer(model_type = "google/mt5-xl")

    results = {}
    
    for gen_file in ["mixtral-rag", "mixtral-no-prompt", "mixtral-simple-prompt"]:
        with jsonlines.open(generated_path / f"{gen_file}.jsonl") as f:
            # all generated answers
            generated_answers = list(f)

        model_response = [answer["answer"] for answer in generated_answers]
        gold_to_response_results = {
            "gold_to_response": get_all_scores(loop_answers, model_response, scorer, results_path / f"{gen_file}_gold_to_response_BERT.csv")
            }
        results[gen_file] = gold_to_response_results

        question_to_response_results = {
            "question_to_response": get_all_scores(loop_questions, model_response, results_path / f"{gen_file}_question_to_response_BERT.csv")
        }
        results[gen_file].update(question_to_response_results)

        question_to_gold_results = {
            "question_to_gold": get_all_scores(loop_questions, loop_answers, scorer, results_path / f"{gen_file}_question_to_gold_BERT.csv")
        }
        results[gen_file].update(question_to_gold_results)

        if "rag" in gen_file:
            documents = [extract_docs_from_prompt(answer["prompt"]) for answer in generated_answers]

            document_to_response_results = {
                "document_to_response": get_all_scores(documents, model_response, scorer, results_path / f"{gen_file}_document_to_response_BERT.csv")
            }
            results[gen_file].update(document_to_response_results)

            document_to_gold_results = {
                "document_to_gold": get_all_scores(documents, loop_answers, scorer, results_path / f"{gen_file}_document_to_gold_BERT.csv")
            }
            
            results[gen_file].update(document_to_gold_results)

    # save to json
    with open(results_path / "semantic_similarity.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)



            


