from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from data_load import load_documents


def prep_embeddings():
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    return embeddings

if __name__ in "__main__":
    path = Path(__file__).parents[1] 

    outpath = path / "data" / "vector_db"
    # load data
    docs = load_documents()

    embeddings = prep_embeddings()

    db = FAISS.from_documents(docs, embeddings)

    # Save the vector store to disk
    db.save_local(outpath)