from pathlib import Path
from langchain.vectorstores import FAISS
from create_vector_db import prep_embeddings # loading the embedding settings from create_vector_db.py to ensure consistency
from data_load import load_loop, map_filter
import logging
from tqdm import tqdm
import json
from generate_mistral import load_mistral


def make_input_mistral_rag(question: str, documents:list) -> str:
    system = """Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kort og præcist på dansk, og giver dit bedste bud også selv om du er usikker.
    Hvis ikke du kender svaret, er det okay, og så siger du bare det.
    Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt. 

    Medarbejderen præsenterer dig for en række dokumenter som måske er relevante for at besvare deres spørgsmål. Hvert dokument har en title, en score og en tekst. En lav score er bedre end en høj en. 
    Det bliver presenteret på følgende måde:
    Dokument 1: Titel: <titel> Score: <score> Tekst: <tekst>
    
    Giv et kort svar og henvis gerne til et dokument hvis det er relevant for spørgsmålet. Hvis du henviser til dokumentet så skriv titlen på dokumentet.
    """
    system_en = """
    You are a language model that understands and speaks competent Danish. 
    You answer briefly and precisely in Danish, and do your best even if you are unsure.
    If you don't know the answer, that's okay, and then you just say so.
    Your task is to help a municipal employee advise them to do their job right. 
    The employee presents you with a number of documents that may be relevant to answer their question.

    The documents will be presented to you first followed by the question.
    """

    documents = [f'Dokument {i+1}: Titel: {doc[0].metadata["title"]} \nScore: {doc[1]}, \nTekst: {doc[0].page_content}' for i, doc in enumerate(documents)]

    documents = "\n".join(documents)

    prompt = f"""
    <|im_start|>system
    {system}

    Dokumenterne:\n {documents}
    <|im_end|>
    <|im_start|>user
    Mit spørgsmål er:{question}

    Answer in the following language: Danish
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
  
    jsondata = load_loop()[:5] 

    output_data = []
    for question in tqdm(map_filter(jsondata, field = "question"), desc="Generating answers"):
        data = {}

        if not question:
            continue

        # get the top documents
        documents = db.similarity_search_with_score(question, k=2)

        # make the input
        input_mdl = make_input_mistral_rag(question, documents)
        
        data["question"] = question
        data["prompt"] = input_mdl
        data["generated"] = model(data["prompt"], top_p=0.9)

        output_data.append(data)

    # save to json
    with open(output_dir / "mistral_rag_documents_in_user_english_inst.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
 


