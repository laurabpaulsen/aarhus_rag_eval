from pathlib import Path
from data_load import load_loop, map_filter
from transformers import pipeline
import pandas as pd


if __name__ in "__main__":
    
    jsondata = load_loop()[:5]
    root_dir = Path(__file__).parents[1]

    ner = pipeline("ner", model="intfloat/multilingual-e5-large")


    for question, response in zip(map_filter(jsondata, field = "question"), map_filter(jsondata, "response")):
        if response is None:
            continue
            
        ner_response = ner(response)

        print(response)
        df_ner = pd.DataFrame.from_records(ner_response)
        print(df_ner)
        print("------------------")



        


