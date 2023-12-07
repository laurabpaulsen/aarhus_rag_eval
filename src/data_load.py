#!/usr/bin/env python3

import json
from pathlib import Path
# from llama_index import download_loader
from langchain.document_loaders import JSONLoader

root_path = Path.cwd().parent
default_path = root_path / "data" / "loop_q_and_a_w_meta.json"
docs_path = root_path / "data" / "loop_documents_w_meta.json"

def load_loop(path: Path = default_path) -> list[dict]:
    with open(path) as f:
        jsondata = json.load(f)
    return jsondata

def load_loop_jsonl(path: Path = default_path) -> list[dict]:
    with open(path) as f:
        jsondata = [json.loads(line) for line in f.readlines()]
    return jsondata

def map_filter(jsondata: list[dict], field: str) -> list[str]:
    return [item[field] for item in jsondata]

def map_questions(jsondata: list[dict]) -> list[str]:
    return map_filter(jsondata, 'question')

def metadata_loop_document(record: dict, metadata: dict) -> dict:
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/json
    # extract the metadata from the documents that we want

    for key in ['nid', 'title','relative_url', 'concatenated_text']:
        metadata[key] = record.get(key)

    return metadata


def load_documents(doc_path: Path = docs_path):
    # JSONReader = download_loader("JSONReader")
    loader = JSONLoader(doc_path,
                        jq_schema=".[] | select(.document_body != null)",
                        content_key="document_body",
                        metadata_func = metadata_loop_document)
    documents = loader.load()
    return documents


class PlainTextTransformer(BaseDocumentTransformer):
    """
    Turn a document's page_content into plaintext by removing html tags
    see https://github.com/langchain-ai/langchain/discussions/7497
    """
    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) > Sequence[Document]:
        for document in documents:
            document.page_content = document.page_content

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Implement the asynchronous version of the method
        return self.transform_documents(documents, **kwargs)


if __name__ == '__main__':
    jsondata1 = load_loop()
    jsondata2 = load_loop_jsonl(root_path / "data" / "loop_q_and_a_w_ref_text_meta.jsonl")

    ## investigate entries with internal references
    questions = [doc['question'] for doc in jsondata2 if 'internal_reference_texts' in doc.keys()]
    responses = [doc['response'] for doc in jsondata2 if 'internal_reference_texts' in doc.keys()]

    documents = [doc['internal_reference_texts'] for doc in jsondata2 if 'internal_reference_texts' in doc.keys()]

    documents2 = load_documents()
    assert len(jsondata) == 700
