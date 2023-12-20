from pathlib import Path
from data_load import load_loop, map_filter
import json 
import dacy
from rouge_score import rouge_scorer
import jsonlines
from tqdm import tqdm
import pandas as pd

def calculate_NER_overlap(text1:str, text2:str, nlp:object):
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

    nlp1 = nlp(text1)
    nlp2 = nlp(text2)

    entities1 = [ent.text for ent in nlp1.ents]
    entities2 = [ent.text for ent in nlp2.ents]

    # compare the entities
    overlapping_ents = [entity for entity in entities1 if entity in entities2]
    
    # total number of entities
    total_ents = len(entities1) + len(entities2)

    # percentage of overlapping entities
    if total_ents != 0:
        return len(overlapping_ents) / total_ents
    else:
        return 0
 

def get_all_scores(texts1:list, texts2:list, nlp, scorer, savepath:Path = None) -> tuple:
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
    scores = pd.DataFrame(columns=["text1", "text2", "ner_overlap", "rouge_l_recall", "rouge_1_recall", "rouge_l_precision", "rouge_1_precision"])

    for txt1, txt2 in tqdm(zip(texts1, texts2)):
        NER_overlap = calculate_NER_overlap(txt1, txt2, nlp)
        rouge = scorer.score(txt1, txt2)

        tmp_dat = pd.DataFrame.from_dict({"text1": [txt1], "text2": [txt2], "ner_overlap": [NER_overlap], "rouge_l_recall": [rouge["rougeL"].recall], "rouge_1_recall": [rouge["rouge1"].recall], "rouge_l_precision": [rouge["rougeL"].precision], "rouge_1_precision": [rouge["rouge1"].precision]})
        scores = pd.concat([scores, tmp_dat], ignore_index=True)
        

    results["ner_overlap"] = scores["ner_overlap"].mean()
    results["rouge_l_recall"] = scores["rouge_l_recall"].mean()
    results["rouge_1_recall"] = scores["rouge_1_recall"].mean()
    results["rouge_l_precision"] = scores["rouge_l_precision"].mean()
    results["rouge_1_precision"] = scores["rouge_1_precision"].mean()

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

    # model for NER overlap
    nlp = dacy.load("large")
    nlp.add_pipe("dacy/ner-fine-grained", config={"size": "large"})

    # model for ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    loop_answers = [answer for answer in map_filter(jsondata, "response") if answer is not None]
    loop_questions = [question for question, answer in zip(map_filter(jsondata, "question"), map_filter(jsondata, "response")) if answer is not None]

    results = {}
    
    for gen_file in files:
        with jsonlines.open(generated_path / gen_file) as f:
            # all generated answers
            generated_answers = list(f)

        generated_answers = [answer["answer"] for answer in generated_answers]

        answer_to_answer_results = {
            "answer_to_answer_gen": get_all_scores(loop_answers, generated_answers, nlp, scorer, results_path / f"{gen_file.stem}_answer_to_answer.csv")
            }
        results[gen_file.stem] = answer_to_answer_results
    
        question_to_answer_gen_results = {
            "question_to_answer_gen": get_all_scores(loop_questions, generated_answers, nlp, scorer, results_path / f"{gen_file.stem}_question_to_answer.csv")
        }
        results[gen_file.stem].update(question_to_answer_gen_results)

        #question_to_answer_loop_results = {
        #    "question_to_answer_loop": get_all_scores(loop_questions, loop_answers, nlp, scorer, results_path / f"{gen_file.stem}_question_to_answer_loop.csv")
        #}

    # save to json
    with open(results_path / "count_based.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)





            


