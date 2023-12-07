from pathlib import Path
from langchain.vectorstores import FAISS
from create_vector_db import prep_embeddings # loading the embedding settings from create_vector_db.py to ensure consistency
from ctransformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA

def make_input_mistral(question: str, documents:list) -> str:
    system = """Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kort og præcist på dansk, og giver dit bedste bud også selv om du er usikker.
    Hvis ikke du kender svaret, er det okay, og så siger du bare det.
    Din opgave er at hjælpe en medarbejder fra kommunen med at rådgive dem til at gøre deres arbejde rigtigt. 

    Du får et spørgsmål som du skal svare på ud fra din viden samt information fra relevante dokumenter som er præsenteret her. 
    """

    prompt = f"""
    <|im_start|>system
    {system}

    Dokumenterne: {documents}
    <|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """

    return prompt

if __name__ in "__main__":
    # based on this example: https://medium.com/international-school-of-ai-data-science/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7
    path = Path(__file__).parents[1]
    db_path = path / "data" / "vector_db"

    embeddings = prep_embeddings()

    # Load the vector store from disk
    db = FAISS.load_local(db_path, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})


    # load mistal
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        model_file = "openhermes-2.5-mistral-7b.Q4_K_M.gguf", 
        model_type="mistral"
    )   
    
    question = "Hvordan for man bevilling til kateterslange?"

    # get the top 4 documents
    documents = retriever.get_relevant_documents(question)

    # make the input
    input = make_input_mistral(question, documents)

    # generate the answer
    answer = model(input)
    print(answer)




