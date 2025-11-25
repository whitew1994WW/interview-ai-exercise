"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.loading.chunk_json import chunk_data
from ai_exercise.models import Document


def get_json_data(url: str) -> dict[str, Any]:
    # Send a GET request to the URL specified in SETTINGS.DOCS_URL
    response = requests.get(url, headers={"Accept": "application/json"})
    try:
        json_data = response.json()
    except json.JSONDecodeError:
        raise Exception(f"Error parsing JSON from {url} with status code {response.status_code}")

    response.raise_for_status()
    return json_data


def document_json_array(data: list[dict[str, Any]], source: str) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects."""
    return [
        Document(page_content=json.dumps(item), metadata={"source": source})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects."""
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)
        docs.extend(document_json_array(chunks, attribute))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def add_documents(collection: chromadb.Collection, docs: list[Document]) -> None:
    """Add documents to the collection"""
    collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata or {} for doc in docs],
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
