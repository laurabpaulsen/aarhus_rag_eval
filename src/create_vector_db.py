"""
This script creates a vector store from a set of documents and saves it to disk.
"""

from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer
from data_load import load_documents


def prep_embeddings():
    # Define the path to the pre-trained model you want to use
    model_name = "KennethEnevoldsen/dfm-sentence-encoder-medium-3"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    return embeddings

if __name__ in "__main__":
    path = Path(__file__).parents[1] 

    outpath = path / "data" / "vector_db"
    # load data
    docs = load_documents()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("KennethEnevoldsen/dfm-sentence-encoder-medium-3")

    # split text into smaller pieces
    splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=512,
        chunk_overlap=50,
    )
    docs = splitter.split_documents(docs)

    embeddings = prep_embeddings()

    db = FAISS.from_documents(docs, embeddings)

    # Save the vector store to disk
    db.save_local(outpath)
