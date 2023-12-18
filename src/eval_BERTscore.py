from pathlib import Path
from data_load import load_loop, map_filter
import json 
import jsonlines
import pandas as pd

# BERTscore
from bert_score import score


def get_all_scores(texts1:list, texts2:list, savepath:Path = None) -> tuple:
    """
    Returns the average NER overlap, ROUGE-L and ROUGE-1 for pairs of texts.

    Parameters
    ----------
    texts1 : list of str
        First list of texts. 
    texts2 : list of str
        Second list of texts.
    savepath : Path
        The path to save the individual scores to.
    """


    # check that the lists are of equal length else shorten the longest
    if len(texts1) != len(texts2):
        if len(texts1) > len(texts2):
            texts1 = texts1[:len(texts2)]
        else:
            texts2 = texts2[:len(texts1)]

    P, R, F1 = score(texts1, texts2, lang="da", verbose=True)

    scores = pd.DataFrame.from_dict({"text1": texts1, "text2": texts2, "BERTscore_precision": P, "BERTscore_recall": R, "BERTscore_f1": F1})
        
    # overall results for all texts
    results = {}
    results["BERTscore_recall"] = scores["BERTscore_recall"].mean()
    results["BERTscore_precision"] = scores["BERTscore_precision"].mean()
    results["BERTscore_f1"] = scores["BERTscore_f1"].mean()


    if savepath:
        scores.to_csv(savepath)

    return results


if __name__ in "__main__":
    
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "data" / "generated"

    # all files with generated answers
    files = [f for f in generated_path.iterdir() if f.suffix == ".jsonl"]

    loop_answers = [answer for answer in map_filter(jsondata, "response") if answer is not None]
    loop_questions = [question for question, answer in zip(map_filter(jsondata, "question"), map_filter(jsondata, "response")) if answer is not None]

    results = {}
    
    for gen_file in files:
        with jsonlines.open(generated_path / gen_file) as f:
            # all generated answers
            generated_answers = list(f)

        generated_answers = [answer["answer"] for answer in generated_answers]

        answer_to_answer_results = {
            "answer_to_answer_gen": get_all_scores(loop_answers, generated_answers, results_path / f"{gen_file.stem}_answer_to_answer_BERT.csv")
            }
        results[gen_file.stem] = answer_to_answer_results
    
        question_to_answer_gen_results = {
            "question_to_answer_gen": get_all_scores(loop_questions, generated_answers, results_path / f"{gen_file.stem}_question_to_answer_BERT.csv")
        }
        results[gen_file.stem].update(question_to_answer_gen_results)

        #question_to_answer_loop_results = {
        #    "question_to_answer_loop": get_all_scores(loop_questions, loop_answers, bertscore, results_path / f"{gen_file.stem}_question_to_answer_loop_BERT.csv")
        #}

    # save to json
    with open(results_path / "semantic_similarity.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)





            


