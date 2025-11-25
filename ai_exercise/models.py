"""Types for the API."""

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class Document:
    """A document to be added to the vector store."""

    page_content: str
    metadata: dict = None


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class LoadDocumentsOutput(BaseModel):
    """Model for the load documents route output."""

    status: str


class ChatQuery(BaseModel):
    """Model for the chat input."""

    query: str

class SearchQuery(BaseModel):
    """Model for the search input."""

    query: str
    k: int = 5

class SearchChunk(BaseModel):
    """Model for a single search result chunk."""

    document: str
    metadata: dict | None = None
    id: str
    distance: float | None = None

class ChatOutput(BaseModel):
    """Model for the chat route output."""
    message: str
    context: list[SearchChunk]

class SearchOutput(BaseModel):
    """Model for the search route output."""

    chunks: list[SearchChunk]
    total_results: int