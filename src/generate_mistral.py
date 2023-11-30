#!/usr/bin/env python3

from data_load import load_loop, map_questions
from ctransformers import AutoModelForCausalLM
from pathlib import Path

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

def load_mistral(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), 
        model_type="mistral",
        gpu_layers=50)
    
    return model


if __name__ == '__main__':
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]
    max_new_tokens = 50
    model = load_mistral(root_dir / "models" / "openhermes-2.5-mistral-7b.Q4_K_M.gguf")

    for question in map_questions(jsondata)[:2]:
        print(question)
        print(model(make_input_mistral(question)))

        print("----------------")
