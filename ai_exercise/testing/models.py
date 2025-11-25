"""Models for evaluation and testing."""

from typing import List
from pydantic import BaseModel, Field
from ai_exercise.models import ChatOutput


class FaithfulnessEvaluation(BaseModel):
    """Structured output for faithfulness evaluation."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Brief explanation of the score")


class CorrectnessEvaluation(BaseModel):
    """Structured output for answer correctness evaluation."""
    
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Brief explanation of the score")


class TestQuestion(BaseModel):
    """A single test question with expected answer and metadata."""
    
    question: str
    answer: str  # Ground truth answer
    source_chunk_id: str  # The chunk ID that should be retrieved
    source_metadata: dict  # Additional metadata about the source


class TestCaseResults(BaseModel):
    """Metrics for evaluating a single test case."""
    
    # Retrieval metrics
    chunk_in_top_k: bool  # True if source_chunk_id in top-k, False otherwise
    chunk_rank: int | None  # Position of source_chunk_id in results (1-indexed), None if not found
    
    # Answer quality metrics
    faithfulness_score: float  # 0.0 to 1.0 - how well the answer is grounded in context
    answer_correctness_score: float  # 0.0 to 1.0 - similarity to ground truth
    
    # Additional metadata
    retrieved_chunk_ids: List[str]  # All chunk IDs retrieved
    num_chunks_retrieved: int


class TestCaseResult(BaseModel):
    """Complete result for a single test case."""
    
    test_question: TestQuestion
    chat_output: ChatOutput
    metrics: TestCaseResults
    
    # Optional: store raw evaluation details
    faithfulness_reasoning: str | None = None
    correctness_reasoning: str | None = None


class AggregateResults(BaseModel):
    """Aggregate metrics across multiple test cases."""
    
    # Retrieval metrics
    precision_at_k: float = Field(..., ge=0.0, le=1.0, description="Average precision: relevant items retrieved / total items retrieved")
    recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Average recall: relevant items retrieved / total relevant items")
    average_chunk_rank: float | None = Field(None, description="Average rank of source chunks (only for found chunks)")
    
    # Answer quality metrics
    average_faithfulness: float = Field(..., ge=0.0, le=1.0, description="Average faithfulness score across all test cases")
    average_correctness: float = Field(..., ge=0.0, le=1.0, description="Average answer correctness score across all test cases")
    
    # Summary statistics
    total_test_cases: int = Field(..., description="Total number of test cases evaluated")
    chunks_found_in_top_k: int = Field(..., description="Number of test cases where source chunk was in top-k")
    chunks_not_found: int = Field(..., description="Number of test cases where source chunk was not retrieved at all")

