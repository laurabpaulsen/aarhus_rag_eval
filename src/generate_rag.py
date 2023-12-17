from pathlib import Path
from langchain.vectorstores import FAISS
from create_vector_db import prep_embeddings
from data_load import load_loop, map_filter
import logging
from tqdm import tqdm
import json
from generate_mistral import load_mistral
from data_load import load_documents
from data_retsinformation import load_retsinformation


def make_input_mistral_rag(question: str, documents:list) -> str:
    system = """Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kort og præcist på dansk, og giver dit bedste bud også selv om du er usikker.
    Hvis ikke du kender svaret, er det okay, og så siger du bare det.
    Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt. 

    Medarbejderen præsenterer dig for en række dokumenter som måske er relevante for at besvare deres spørgsmål. Hvert dokument har en title, en score og en tekst. En lav score er bedre end en høj en. 
    
    Giv et kort svar og henvis gerne til et dokument hvis det er relevant for spørgsmålet.
    """
    documents = [f'Titel: {doc[0]} \nScore: {doc[1]}, \nTekst: {doc[2]}' for doc in documents]

    documents = "\n".join(documents)

    prompt = f"""
    <|im_start|>system
    {system}

    Dokumenterne:\n {documents}
    <|im_end|>
    <|im_start|>user
    Spørgsmålet er:{question}
    <|im_end|>
    <|im_start|>assistant
    """

    return prompt

if __name__ in "__main__":
    logger = logging.getLogger(__name__)

    # based on this example: https://medium.com/international-school-of-ai-data-science/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7
    root_dir = Path(__file__).parents[1]
    db_path = root_dir / "data" / "vector_db"

    output_dir = root_dir / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = prep_embeddings()

    print("Loading vector store from disk")
    # Load the vector store from disk
    db = FAISS.load_local(db_path, embeddings)

    # load mistal
    print("Loading mistral model")
    model = load_mistral()  
  
    jsondata = load_loop()

    # load the unsplitted docs
    full_documents_loop = load_documents()
    full_documents_ri = load_retsinformation(paragraph=False)

    full_documents = full_documents_loop + full_documents_ri

    docs_dict = {doc.metadata["title"]: doc.page_content for doc in full_documents}


    output_data = []
    for question in tqdm(map_filter(jsondata, field = "question"), desc="Generating answers"):
        data = {}

        if not question:
            continue

        # get the top documents
        retrieved_documents = db.similarity_search_with_score(question, k=5)
        
        # get the titles and scores of the retrieved documents
        retrieved_titles = [(doc.metadata["title"], score) for doc, score in retrieved_documents]
        #print(retrieved_titles)
        
        # get the full documents and keep the best score if the same document is retrieved multiple times
        input_docs = []
        for title, score in retrieved_titles:
            if title not in [doc[0] for doc in input_docs]: # if the title is not already in the list
                #save title, score and text
                input_docs.append((title, score, docs_dict[title]))

            else: # if the title is already in the list
                # check if the current score is better than the one in the list
                for i, doc in enumerate(input_docs):
                    if doc[0] == title:
                        if score < doc[1]:
                            input_docs[i] = (title, score, docs_dict[title])

        # make the input
        input_mdl = make_input_mistral_rag(question, input_docs)
        
        data["num_docs"] = len(input_docs)
        data["title_docs"] = [doc[0] for doc in input_docs]
        data["question"] = question
        data["prompt"] = input_mdl
        #data["answer"] = model(input_mdl, top_p = 0.9)

        output_data.append(data)
    
    # for inspection of the input to the model
    #for data in output_data:
        #print(data["question"])
        #print(data["num_docs"])
        #data["title_docs"].sort()
        #print(data["title_docs"])
        #print(data["prompt"])

        #print(f"the prompt is {len(data['prompt'])} tokens long")
        #print("-----------------------")
    
    # save to json
    with open(output_dir / "mistral_rag.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
 


