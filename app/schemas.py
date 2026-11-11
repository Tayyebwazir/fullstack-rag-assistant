"""Shared Pydantic request schemas for the Multi-Tenant RAG Assistant API."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Payload for asking a question against a tenant's ingested documents."""

    question: str = Field(min_length=1, max_length=4000)
    source_filter: str | None = Field(default=None, description="Optional document filter")


class DocIdRequest(BaseModel):
    """Payload identifying a single document by UUID."""

    doc_id: str
