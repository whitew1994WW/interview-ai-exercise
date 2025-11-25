"""Functions for evaluating chatbot responses against test questions."""

import requests
from typing import List

from ai_exercise.models import SearchChunk, ChatOutput
from ai_exercise.testing.models import (
    TestQuestion,
    TestCaseResults,
    TestCaseResult,
    AggregateResults,
    FaithfulnessEvaluation,
    CorrectnessEvaluation,
)


def run_chat_query(question: str, base_url: str = "http://localhost") -> ChatOutput:
    """Run a chat query against the API and return the response.
    
    Args:
        question: The question to ask
        base_url: Base URL of the API (default: http://localhost)
        
    Returns:
        ChatResponse with the answer and context chunks
        
    Raises:
        requests.exceptions.RequestException: If the API call fails
    """
    response = requests.post(
        f"{base_url}/chat",
        json={"query": question},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    
    # Convert context dictionaries to SearchChunk objects
    context_data = result.get("context", [])
    context_chunks = [SearchChunk(**chunk) for chunk in context_data]
    
    return ChatOutput(
        message=result["message"],
        context=context_chunks,
    )


def calculate_retrieval_metrics(
    source_chunk_id: str,
    retrieved_chunks: List[SearchChunk],
    k: int | None = None,
) -> tuple[bool, int | None, List[str]]:
    """Calculate retrieval metrics (precision and recall).
    
    Args:
        source_chunk_id: The expected chunk ID
        retrieved_chunks: List of SearchChunk objects returned by the system
        k: Number of chunks to consider (if None, uses all retrieved chunks)
        
    Returns:
        Tuple of (chunk_in_top_k, chunk_rank, chunk_ids)
        - chunk_in_top_k: True if source chunk in top-k, False otherwise
        - chunk_rank: Position of source chunk (1-indexed), None if not found
        - chunk_ids: List of all retrieved chunk IDs
    """
    # Extract chunk IDs from retrieved chunks
    chunk_ids = [chunk.id for chunk in retrieved_chunks]
    
    # Limit to top-k if specified
    if k is not None:
        chunk_ids_to_check = chunk_ids[:k]
    else:
        chunk_ids_to_check = chunk_ids
    
    # Check if source chunk is in top-k
    is_in_top_k = source_chunk_id in chunk_ids_to_check
    
    # Find rank of source chunk (1-indexed)
    chunk_rank = None
    if source_chunk_id in chunk_ids:
        chunk_rank = chunk_ids.index(source_chunk_id) + 1
    
    return is_in_top_k, chunk_rank, chunk_ids


def evaluate_faithfulness(
    answer: str,
    context_chunks: List[SearchChunk],
    openai_client,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    """Evaluate how well the answer is grounded in the provided context.
    
    Uses LLM-as-judge with structured output to assess whether the answer is faithful to the context.
    
    Args:
        answer: The generated answer
        context_chunks: List of SearchChunk objects used
        openai_client: OpenAI client instance
        model: Model to use for evaluation (must support structured outputs)
        
    Returns:
        Tuple of (score, reasoning)
        - score: 0.0 to 1.0 (0.0 = not faithful, 1.0 = fully faithful)
        - reasoning: Explanation of the score
    """
    # Combine context chunks into a single string
    context_str = "\n\n".join([
        f"Chunk {i+1}:\n{chunk.document}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    prompt = f"""Evaluate whether the following answer is faithful to the provided context.
An answer is faithful if all claims in the answer can be verified from the context.

Context:
{context_str}

Answer:
{answer}

Evaluate the faithfulness on a scale from 0.0 to 1.0 where:
- 1.0: All claims in the answer are directly supported by the context
- 0.7-0.9: Most claims are supported, minor unsupported details
- 0.4-0.6: Some claims are supported, some are not
- 0.1-0.3: Few claims are supported by context
- 0.0: Answer contradicts context or is entirely unsupported

Provide a score and brief reasoning for your evaluation."""

    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at evaluating whether answers are grounded in provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=FaithfulnessEvaluation,
        temperature=0.0,
    )
    
    evaluation = completion.choices[0].message.parsed
    
    return evaluation.score, evaluation.reasoning


def evaluate_answer_correctness(
    generated_answer: str,
    ground_truth_answer: str,
    openai_client,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    """Evaluate how correct the generated answer is compared to ground truth.
    
    Uses LLM-as-judge with structured output to assess semantic similarity and correctness.
    
    Args:
        generated_answer: The answer produced by the system
        ground_truth_answer: The expected correct answer
        openai_client: OpenAI client instance
        model: Model to use for evaluation (must support structured outputs)
        
    Returns:
        Tuple of (score, reasoning)
        - score: 0.0 to 1.0 (0.0 = incorrect, 1.0 = fully correct)
        - reasoning: Explanation of the score
    """
    prompt = f"""Compare the generated answer to the ground truth answer and evaluate correctness.
The answers don't need to be identical, but the generated answer should convey the same information.

Ground Truth Answer:
{ground_truth_answer}

Generated Answer:
{generated_answer}

Evaluate the correctness on a scale from 0.0 to 1.0 where:
- 1.0: Generated answer is semantically equivalent to ground truth
- 0.7-0.9: Most key information is correct, minor omissions
- 0.4-0.6: Some correct information, but missing key points or has errors
- 0.1-0.3: Mostly incorrect or irrelevant
- 0.0: Completely incorrect or contradicts ground truth

Provide a score and brief reasoning for your evaluation."""

    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at evaluating answer quality and correctness."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=CorrectnessEvaluation,
        temperature=0.0,
    )
    
    evaluation = completion.choices[0].message.parsed
    
    return evaluation.score, evaluation.reasoning


def evaluate_test_case(
    test_question: TestQuestion,
    openai_client,
    base_url: str = "http://localhost",
    k: int = 5,
    eval_model: str = "gpt-4o",
) -> TestCaseResult:
    """Evaluate a single test case end-to-end.
    
    Args:
        test_question: The test question to evaluate
        openai_client: OpenAI client for LLM-as-judge evaluation
        base_url: Base URL of the chat API
        k: Number of chunks to consider for top-k metrics
        eval_model: Model to use for evaluation
        
    Returns:
        Complete TestCaseResult with all metrics
    """
    # Step 1: Run the chat query
    chat_response = run_chat_query(test_question.question, base_url)
    
    # Step 2: Calculate retrieval metrics
    chunk_in_top_k, chunk_rank, chunk_ids = calculate_retrieval_metrics(
        source_chunk_id=test_question.source_chunk_id,
        retrieved_chunks=chat_response.context,
        k=k,
    )
    
    # Step 3: Evaluate faithfulness
    faithfulness_score, faithfulness_reasoning = evaluate_faithfulness(
        answer=chat_response.message,
        context_chunks=chat_response.context,
        openai_client=openai_client,
        model=eval_model,
    )
    
    # Step 4: Evaluate answer correctness
    correctness_score, correctness_reasoning = evaluate_answer_correctness(
        generated_answer=chat_response.message,
        ground_truth_answer=test_question.answer,
        openai_client=openai_client,
        model=eval_model,
    )
    
    # Step 5: Combine into metrics
    metrics = TestCaseResults(
        chunk_in_top_k=chunk_in_top_k,
        chunk_rank=chunk_rank,
        faithfulness_score=faithfulness_score,
        answer_correctness_score=correctness_score,
        retrieved_chunk_ids=chunk_ids,
        num_chunks_retrieved=len(chunk_ids),
    )
    
    # Step 6: Return complete result
    return TestCaseResult(
        test_question=test_question,
        chat_output=chat_response,
        metrics=metrics,
        faithfulness_reasoning=faithfulness_reasoning,
        correctness_reasoning=correctness_reasoning,
    )


def aggregate_results(test_case_results: List[TestCaseResult]) -> AggregateResults:
    """Aggregate metrics from multiple test case results.
    
    Args:
        test_case_results: List of TestCaseResult objects to aggregate
        
    Returns:
        AggregateResults with computed metrics across all test cases
        
    Raises:
        ValueError: If the list is empty
    """
    if not test_case_results:
        raise ValueError("Cannot aggregate empty list of test case results")
    
    total_cases = len(test_case_results)
    
    # Count chunks found in top-k
    chunks_in_top_k = sum(
        1 for result in test_case_results 
        if result.metrics.chunk_in_top_k
    )
    
    # Count chunks not found at all (rank is None)
    chunks_not_found = sum(
        1 for result in test_case_results 
        if result.metrics.chunk_rank is None
    )
    
    # Calculate precision at k
    # Precision = (relevant items retrieved in top-k) / (total items retrieved in top-k)
    # For each query, we retrieve k items. If the relevant item is in top-k, precision = 1/k, else 0
    # Average precision across all queries
    total_retrieved_per_query = test_case_results[0].metrics.num_chunks_retrieved if test_case_results else 0
    if total_retrieved_per_query > 0:
        precision_at_k = chunks_in_top_k / total_cases  # Simplified: proportion of queries with relevant doc in top-k
    else:
        precision_at_k = 0.0
    
    # Calculate recall at k
    # Recall = (relevant items retrieved in top-k) / (total relevant items)
    # For each query, there is exactly 1 relevant item. If it's in top-k, recall = 1, else 0
    # Average recall across all queries
    recall_at_k = chunks_in_top_k / total_cases
    
    # Calculate average chunk rank (only for chunks that were found)
    chunk_ranks = [
        result.metrics.chunk_rank 
        for result in test_case_results 
        if result.metrics.chunk_rank is not None
    ]
    average_chunk_rank = sum(chunk_ranks) / len(chunk_ranks) if chunk_ranks else None
    
    # Calculate average faithfulness score
    faithfulness_scores = [
        result.metrics.faithfulness_score 
        for result in test_case_results
    ]
    average_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    
    # Calculate average correctness score
    correctness_scores = [
        result.metrics.answer_correctness_score 
        for result in test_case_results
    ]
    average_correctness = sum(correctness_scores) / len(correctness_scores)
    
    return AggregateResults(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        average_chunk_rank=average_chunk_rank,
        average_faithfulness=average_faithfulness,
        average_correctness=average_correctness,
        total_test_cases=total_cases,
        chunks_found_in_top_k=chunks_in_top_k,
        chunks_not_found=chunks_not_found,
    )

