from pathlib import Path
from data_load import load_loop, map_filter
import json 
import dacy
from rouge_score import rouge_scorer
from evaluate import load

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


def get_all_scores(texts1:list, texts2:list, nlp, scorer) -> tuple:
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
    """

    results = {}

    NER_overlap = []
    ROUGE_l = []
    ROUGE_1 = []

    for txt1, txt2 in zip(texts1, texts2):
        NER_overlap.append(calculate_NER_overlap(txt1, txt2, nlp))
        rouge = scorer.score(txt1, txt2)
        ROUGE_l.append(rouge["rougeL"].recall)
        ROUGE_1.append(rouge["rouge1"].recall)


    results["ner_overlap"] = sum(NER_overlap) / len(NER_overlap)
    results["rouge_l"] = sum(ROUGE_l) / len(ROUGE_l)
    results["rouge_1"] = sum(ROUGE_1) / len(ROUGE_1)

    return results


if __name__ in "__main__":
    
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "data" / "generated"

    # model for NER overlap
    nlp = dacy.load("large")
    nlp.add_pipe("dacy/ner-fine-grained", config={"size": "large"})

    # model for ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # all files with generated answers
    files = [f for f in generated_path.iterdir() if f.suffix == ".json"]

    loop_answers = [answer for answer in map_filter(jsondata, "response") if answer is not None]
    loop_questions = [question for question in map_filter(jsondata, "question") if question is not None]

    results = {}
    
    for gen_file in files:
        with open(generated_path / gen_file, "r") as f:
            generated_answers = json.load(f)

        generated_answers = [answer["answer"] for answer in generated_answers]

        answer_to_answer_results = {
            "answer_to_answer_gen": get_all_scores(loop_answers, generated_answers, nlp, scorer)
            }
        results[gen_file.stem] = answer_to_answer_results
    
        question_to_answer_gen_results = {
            "question_to_answer_gen": get_all_scores(loop_questions, generated_answers, nlp, scorer)
        }
        results[gen_file.stem].update(question_to_answer_gen_results)

        #question_to_answer_loop_results = {
        #    "question_to_answer_loop": get_all_scores(loop_questions, loop_answers, nlp, scorer)
        #}

    # save to json
    with open(results_path / "count_based.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)





            


