#!/usr/bin/env python3

from data_load import load_loop, map_filter
from ctransformers import AutoModelForCausalLM
from pathlib import Path
import os
import json
from tqdm import tqdm

def make_input_mistral(question: str) -> str:
    system = """Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kort og præcist på dansk, og giver dit bedste bud også selv om du er usikker.
    Hvis ikke du kender svaret, er det okay, og så siger du bare det.
    Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt.
    """

    prompt = f"""
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    return prompt

def load_mistral():
    if os.path.expanduser("~") == "/Users/laurapaulsen":
        gpu_layers = 50
    else:
        gpu_layers = 0


    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        model_file = "openhermes-2.5-mistral-7b.Q4_K_M.gguf", 
        model_type="mistral",
        context_length = 4000,
        gpu_layers=gpu_layers
        )
    
    
    return model


if __name__ == '__main__':
    jsondata = load_loop()[:5] # a list of dictionaries
    root_dir = Path(__file__).parents[1]
    output_dir = root_dir / "data" / "generated"

    # ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading mistral model!")
    model = load_mistral()

    output_data = []
    for question in tqdm(map_filter(jsondata, field = "question"), desc="Generating answers"):
        data = {}

        if not question:
            continue
        
        data["question"] = question
        data["prompt"] = make_input_mistral(question)
        data["answer"] = model(data["prompt"])

        output_data.append(data)

    # save to json
    with open(output_dir / "mistral.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
