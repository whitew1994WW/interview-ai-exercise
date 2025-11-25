"""FastAPI app creation, main API routes."""

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import create_prompt, get_completion
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    add_documents,
    build_docs,
    get_json_data,
    split_docs,
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    LoadDocumentsOutput,
    SearchChunk,
    SearchOutput,
    SearchQuery,
)
from ai_exercise.retrieval.retrieval import get_relevant_chunks
from ai_exercise.retrieval.vector_store import create_collection


app = FastAPI()

collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to load documents into vector store."""
    global collection
    
    chroma_client.delete_collection(name=SETTINGS.collection_name)
    collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)

    for url in SETTINGS.json_urls:
        json_data = get_json_data(url)
        print(json_data)
        documents = build_docs(json_data, url)

        # split docs
        documents = split_docs(documents)

        # load documents into vector store
        add_documents(collection, documents)

    # check the number of documents in the collection
    print(f"Number of documents in collection: {collection.count()}")

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Get relevant chunks with full context information
    context_chunks = get_relevant_chunks(
        collection=collection, query=chat_query.query, k=SETTINGS.k_neighbors
    )

    # Extract document strings for the prompt
    documents = [chunk.document for chunk in context_chunks]

    # Create prompt with context
    prompt = create_prompt(query=chat_query.query, context=documents)

    print(f"Prompt: {prompt}")

    # Get completion from LLM
    result = get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )

    return ChatOutput(message=result, context=context_chunks)


@app.post("/search")
def search_route(search_query: SearchQuery) -> SearchOutput:
    """Search route to query the vector database directly and return matching chunks."""
    # Query the collection directly
    results = collection.query(
        query_texts=[search_query.query], n_results=search_query.k
    )

    # Extract results from the query response
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Build list of SearchChunk objects
    chunks = [
        SearchChunk(
            document=doc,
            metadata=meta if meta else None,
            id=chunk_id,
            distance=dist if distances else None,
        )
        for doc, meta, chunk_id, dist in zip(
            documents, metadatas, ids, distances if distances else [None] * len(documents)
        )
    ]

    return SearchOutput(chunks=chunks, total_results=len(chunks))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
