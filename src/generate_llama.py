#!/usr/bin/env python3


from data_load import load_loop, map_questions

from ctransformers import AutoModelForCausalLM
from transformers import pipeline
from pathlib import Path

def make_input_llama(question: str) -> str:
    system = """Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kort og præcist på dansk, og giver dit bedste bud også selv om du er usikker.
    Hvis ikke du kender svaret, er det okay, og så siger du bare det.
    Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt.
    """

    prompt = f"""
    [INST] <<SYS>>
    {system}
    <</SYS>>
    {question}[/INST]
    """

    return prompt

def load_llama(model_path):
    model = AutoModelForCausalLM.from_pretrained(str(model_path), model_type="llama")
    return model


if __name__ == '__main__':
    jsondata = load_loop()
    root_dir = Path(__file__).parents[1]
    model = load_llama(root_dir / "models" / "llama-2-7b-chat.Q4_K_M.gguf")


    for question in map_questions(jsondata)[:2]:
        print(question)
        print(model(make_input_llama(question)))
        print("----------------")
