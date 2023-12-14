from pathlib import Path
from data_load import load_loop, map_filter
import json 
import dacy


if __name__ in "__main__":
    
    jsondata = load_loop()[:5]
    root_dir = Path(__file__).parents[1]

    nlp = dacy.load("medium")

    # add ner to pipeline
    nlp.add_pipe("dacy/ner-fine-grained", config={"size": "medium"})


    # load generated answers
    generated_path = root_dir / "data" / "generated" / "mistral_rag.json"
    with open(generated_path, "r") as f:
        generated_answers = json.load(f)

    # only keep the answers
    generated_answers = [answer["answer"] for answer in generated_answers]

    loop_answers = [answer for answer in map_filter(jsondata, "response") if answer is not None]

    overlap = []

    for i, (answer_loop, generated_answer_gen) in enumerate(zip(loop_answers, generated_answers)):

        # get the entities
        loop_nlp = nlp(answer_loop)
        gen_nlp = nlp(generated_answer_gen)

        loop_entities = [ent.text for ent in loop_nlp.ents]
        gen_entities = [ent.text for ent in gen_nlp.ents]

        # compare the entities
        overlapping_ents = [entity for entity in loop_entities if entity in gen_entities]
        
        # total number of entities
        total_ents = len(loop_entities) + len(gen_entities)

        # percentage of overlapping entities
        overlap.append(len(overlapping_ents) / total_ents)

        print(f"Loop: {loop_entities}")
        print(f"Generated: {gen_entities}")

        # parts of speech tags
        loop_pos = [token.pos_ for token in loop_nlp]
        gen_pos = [token.pos_ for token in gen_nlp]
        print ("------------------")
        #print(f"Loop: {loop_pos}")
        #print(f"Generated: {gen_pos}")
        #print ("------------------")


    print(f"Average overlap: {sum(overlap) / len(overlap)}")





        


