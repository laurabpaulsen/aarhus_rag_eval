"""
This script creates a vector store from a set of documents and saves it to disk.
"""

from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
from data_load import load_documents
from data_retsinformation import load_retsinformation

def prep_embeddings():
    # Define the path to the pre-trained model you want to use
    model_name = "KennethEnevoldsen/dfm-sentence-encoder-large-2"
    #model_name = "sentence-transformers/all-MiniLM-l6-v2"

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
    docs_loop = load_documents()
    print(type(docs_loop))

    # load retsinformation
    docs_ri = load_retsinformation(paragraph=False)
    print(type(docs_ri))

    # combine retsinformation and loop documents
    docs = docs_loop + docs_ri


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

    db = FAISS.from_documents(docs, embeddings, distance_strategy = DistanceStrategy.COSINE)

    # Save the vector store to disk
    db.save_local(outpath)
