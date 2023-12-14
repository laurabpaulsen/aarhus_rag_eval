from pathlib import Path
from data_load import load_loop, map_filter
import json 
import dacy


if __name__ in "__main__":
    
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]

    results_path = root_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    generated_path = root_dir / "data" / "generated"

    nlp = dacy.load("medium")

    # add ner to pipeline
    nlp.add_pipe("dacy/ner-fine-grained", config={"size": "medium"})

    # get all the generated answers
    files = [f for f in generated_path.iterdir() if f.suffix == ".json"]


    loop_answers = [answer for answer in map_filter(jsondata, "response") if answer is not None]

    results = {}
    for gen_file in files:
        with open(generated_path / gen_file, "r") as f:
            generated_answers = json.load(f)

        overlap = []
        
        
        generated_answers = [answer["answer"] for answer in generated_answers]
        for i, (answer_loop, answer_gen) in enumerate(zip(loop_answers, generated_answers)):

            # get the entities
            loop_nlp = nlp(answer_loop)
            gen_nlp = nlp(answer_gen)

            loop_entities = [ent.text for ent in loop_nlp.ents]
            gen_entities = [ent.text for ent in gen_nlp.ents]

            # compare the entities
            overlapping_ents = [entity for entity in loop_entities if entity in gen_entities]
            
            # total number of entities
            total_ents = len(loop_entities) + len(gen_entities)

            # percentage of overlapping entities
            try:
                overlap.append(len(overlapping_ents) / total_ents)
            except ZeroDivisionError:
                overlap.append(0)


        results[gen_file.stem] = {"ner_overlap": sum(overlap) / len(overlap)}
    
    # save to json
    with open(results_path / "ner_overlap.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)





            


