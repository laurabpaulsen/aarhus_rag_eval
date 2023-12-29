"""
Creates a new file with the scores of the different types as columns.

(still need to figure out how to merge with LLM scores)
"""

import pandas as pd
from pathlib import Path


if __name__ in "__main__":

    root_dir = Path(__file__).parents[1]
    results_path = root_dir / "results"
    outpath = root_dir / "results" / "merged"

    outpath.mkdir(parents=True, exist_ok=True)

    for model in ["mixtral-no-prompt", "mixtral-simple-prompt", "mixtral-rag"]:
        for compare in ["gold_to_response", "question_to_gold", "question_to_response", "document_to_gold", "document_to_response"]:
            files = [f for f in results_path.iterdir() if f.suffix == ".csv" and compare in f.stem and model in f.stem]
            
            if len(files) == 0: 
                continue

            dfs = [pd.read_csv(f) for f in files]

            # merge the dataframes by candidate and reference columns
            for i in range(len(dfs)):
                print(dfs[i].columns)
                if i == 0:
                    merged = dfs[i]
                else:
                    merged = merged.merge(dfs[i], on=["candidate", "reference"], how="outer")
            
            filename = f"{model}_{compare}.csv"
            merged.to_csv(outpath / filename, index=False)

            print(f"length of {filename}: {len(merged)}")