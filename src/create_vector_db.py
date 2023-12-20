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

#from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
#from langchain.text_splitter import SpacyTextSplitter

def prep_embeddings():
    # Define the path to the pre-trained model you want to use
    # model_name = "Maltehb/danish-bert-botxo"
    model_name = "sentence-transformers/all-MiniLM-l6-v2"

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


    #embeddings = LlamaCppEmbeddings(model_path=str((Path(__file__).parents[1] / "models" / "mistral-7b-v0.1.Q4_K_M.gguf").absolute()))
    # embeddings = LlamaCppEmbeddings(model_path=str((Path(__file__).parents[1] / "models" / "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf").absolute()),  # Download the model file first
    #                                 n_ctx=8192,  # The max sequence length to use - note that longer sequence lengths require much more resources
    #                                 n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
    #                                 n_gpu_layers=6  )

    # embeddings = SpacyEmbeddings()
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


    # # load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("KennethEnevoldsen/dfm-sentence-encoder-medium-3")
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-l6-v2")

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
