"""Retrieve relevant chunks from a vector store."""

import chromadb
from ai_exercise.models import SearchChunk


def get_relevant_chunks(
    collection: chromadb.Collection, query: str, k: int
) -> list:
    """Retrieve k most relevant chunks for the query with full context information.
    
    Returns a list of SearchChunk-compatible dictionaries.
    """


    results = collection.query(query_texts=[query], n_results=k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Build and return SearchChunk objects
    chunks = [
        SearchChunk(
            document=doc,
            metadata=meta if meta else None,
            id=chunk_id,
            distance=dist,
        )
        for doc, meta, chunk_id, dist in zip(
            documents,
            metadatas,
            ids,
            distances if distances else [None] * len(documents),
        )
    ]

    return chunks
