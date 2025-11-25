"""Testing and evaluation module for the RAG chatbot."""

from ai_exercise.testing.models import (
    TestQuestion,
    TestCaseResults,
    TestCaseResult,
    AggregateResults,
    FaithfulnessEvaluation,
    CorrectnessEvaluation,
)
from ai_exercise.testing.evaluate import (
    run_chat_query,
    calculate_retrieval_metrics,
    evaluate_faithfulness,
    evaluate_answer_correctness,
    evaluate_test_case,
    aggregate_results,
)

__all__ = [
    "TestQuestion",
    "TestCaseResults",
    "TestCaseResult",
    "AggregateResults",
    "FaithfulnessEvaluation",
    "CorrectnessEvaluation",
    "run_chat_query",
    "calculate_retrieval_metrics",
    "evaluate_faithfulness",
    "evaluate_answer_correctness",
    "evaluate_test_case",
    "aggregate_results",
]

