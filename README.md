# aarhus_rag_eval

Code for "Evaluation of Large Language Models from Human Goals" by Malte Lau Petersen and Laura Bock Paulsen.

Natural Language Processing at Msc Cognitive Science, Aarhus University

## Introduction
With recent developments within natural language processing, the potential use-
cases of large language models (LLM) have expanded by reframing any type of task
to a Natural Language Generation (NLG) task (Brown et al., 2020). Consequently,
there is a growing interest in using the models for internal and external use in
commercial business and other organisations. The quality of the output of the
model is crucial, but pinpointing which qualities to prioritise in a given context can
be difficult, and many of the proposed solutions that do exist are too generic to
address specific concerns. We propose evaluating models not just on preexisting
benchmarks and accuracy, but on contextually relevant qualities, with the aim to
nuance evaluation of LLMs and to emphasise that metrics should be meaningful in
relation to human goals.

## Data
Not shared yet.

## Code
All python files only run code in `if __name__ == "__main__"` so that we can freely impport functions without side effects.

```
src/data_*
```

Code to load the different data sources as langchain `Document`s. `data_retsinformation` downloads law texts from retsinformation.dk via gigaword and huggingface. `data_load` loads the internal data source with 700 questions and human answers.

```
src/generate_*
```

Generate proposals for answers. Not the focus of our work, but has to be done in order to have something to evaluate on. 
`generate_mixtral.py` has the generative language model and `generate_rag.py` has the rag pipeline. `create_vector_db.py` preprocesses the documents for retrieval.

```
src/eval_*
```

Run various evaluation metrics on the generated proposals. `eval_countbased.py` and `eval_BERTscore.py` do lexical and semantic similarity. `eval_llm.py` uses mixtral to generate various scores.

```
*.R
```
Data munging for combining data from different sources in `concate_scores.py` and `merge_results.R`. ggplot2 code for generating figure 1 and 2 as well as examples of the code used to filter and summarise the results for preparing the tables in `plot_results.R`

# TODO
- [x] Run generative pipeline
- [x] Run evaluation
- [x] Write the paper
- [ ] Get permission to share the dataset
