"""Evaluation page for generating and viewing synthetic test datasets."""

import json
from pathlib import Path
from typing import List

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel

from demo.ping import display_message_if_ping_fails

from ai_exercise.constants import SETTINGS
from ai_exercise.testing.evaluate import evaluate_test_case, aggregate_results
from ai_exercise.testing.models import TestQuestion


# Pydantic models for structured output
class QuestionAnswer(BaseModel):
    """A single question-answer pair."""
    question: str
    answer: str


class QuestionSet(BaseModel):
    """A set of questions generated from a chunk."""
    questions: List[QuestionAnswer]

st.set_page_config(
    page_title="Evaluation - RAG Example",
    page_icon="📊",
)

# Path to store evaluation datasets
EVAL_DATA_PATH = Path("eval_data")
EVAL_DATA_PATH.mkdir(exist_ok=True)
EVAL_FILE = EVAL_DATA_PATH / "test_questions.json"

with st.sidebar:
    display_message_if_ping_fails()
    
    st.divider()
    st.subheader("Vector Database")
    
    if st.button("🔄 Rebuild Vector DB", type="primary", use_container_width=True):
        with st.spinner("Loading documents into vector database..."):
            try:
                response = requests.get("http://localhost/load", timeout=300)
                response.raise_for_status()
                result = response.json()
                if result.get("status") == "ok":
                    st.success("✅ Vector database rebuilt successfully!")
                else:
                    st.warning(f"⚠️ Response: {result}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to rebuild vector database: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

st.title("📊 Evaluation")
st.markdown("Generate synthetic test datasets and evaluate the RAG chatbot.")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Generate Test Set", "View Test Questions", "Run Evaluation"])

with tab1:
    st.subheader("🔨 Generate Synthetic Test Set")
    st.markdown(
        """
        Generate question-answer pairs based on the chunks in the vector database.
        Each chunk will be used to generate 2 questions, with both the chunk context
        and the full schema provided to GPT-4o.
        """
    )
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        questions_per_chunk = st.number_input(
            "Questions per chunk",
            min_value=1,
            max_value=10,
            value=2,
            help="Number of questions to generate for each chunk"
        )
    
    with col2:
        max_chunks = st.number_input(
            "Max chunks to process",
            min_value=1,
            max_value=1000,
            value=50,
            help="Limit the number of chunks to process (useful for testing)"
        )
    
    generate_button = st.button(
        "🚀 Generate Test Set",
        type="primary",
        use_container_width=True
    )
    
    if generate_button:
        # Check for OpenAI API key
        openai_client = OpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())
        
        progress_bar = st.progress(0, text="Starting generation...")
        status_text = st.empty()
        
        try:
            # First, get all chunks from the vector database
            status_text.text("Fetching chunks from vector database...")
            response = requests.post(
                "http://localhost/search",
                json={"query": "API", "k": max_chunks},
                timeout=60,
            )
            response.raise_for_status()
            search_result = response.json()
            chunks = search_result.get("chunks", [])
            
            if not chunks:
                st.error("❌ No chunks found in the vector database. Please load documents first.")
                st.stop()
            
            st.info(f"Found {len(chunks)} chunks in the database")
            
            # Load API schemas on-demand for each chunk
            status_text.text("Preparing to generate questions...")
            schemas_cache = {}  # Cache loaded schemas to avoid redundant requests
            
            # Generate questions for each chunk
            all_questions = []
            total_chunks = len(chunks)
            
            for idx, chunk in enumerate(chunks):
                progress = (idx + 1) / total_chunks
                progress_bar.progress(progress, text=f"Processing chunk {idx + 1}/{total_chunks}...")
                status_text.text(f"Generating questions for chunk {idx + 1}/{total_chunks}")
                
                # Get chunk metadata
                chunk_metadata = chunk.get("metadata", {})
                chunk_url = chunk_metadata.get("url", "")
                chunk_attribute = chunk_metadata.get("attribute", "unknown")
                
                # Load the full API schema for this chunk
                full_schema = None
                if chunk_url and chunk_url not in schemas_cache:
                    try:
                        status_text.text(f"Loading schema from {chunk_url}...")
                        schema_response = requests.get(
                            chunk_url,
                            headers={"Accept": "application/json"},
                            timeout=30
                        )
                        schema_response.raise_for_status()
                        schemas_cache[chunk_url] = schema_response.json()
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load schema from {chunk_url}: {e}")
                        schemas_cache[chunk_url] = None
                
                full_schema = schemas_cache.get(chunk_url)
                
                # Create context with full schema
                schema_context = ""
                if full_schema:
                    # Include relevant parts of the schema for context
                    schema_info = {
                        "title": full_schema.get("info", {}).get("title", ""),
                        "version": full_schema.get("info", {}).get("version", ""),
                        "description": full_schema.get("info", {}).get("description", ""),
                        "servers": full_schema.get("servers", []),
                    }
                    schema_context = f"\n\nFull API Schema Context:\n{json.dumps(schema_info, indent=2)}"
                
                # Create prompt for GPT-4o with structured output
                prompt = f"""You are an expert API documentation assistant. Given the following API documentation chunk and its full schema context, generate {questions_per_chunk} diverse and specific questions that a developer might ask about this API, along with accurate answers based solely on the documentation provided.

API Documentation Chunk:
{chunk['document']}

Metadata:
- URL: {chunk_url}
- Attribute: {chunk_attribute}
- Chunk ID: {chunk['id']}
{schema_context}

Generate {questions_per_chunk} realistic questions covering different aspects like:
- Endpoints and their purposes
- Request/response parameters
- Authentication methods
- Data formats and schemas
- Error handling
- Best practices"""

                # Call OpenAI API with structured output
                try:
                    completion = openai_client.beta.chat.completions.parse(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert at creating test questions from API documentation."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        response_format=QuestionSet,
                        temperature=0.7,
                    )
                    
                    question_set = completion.choices[0].message.parsed
                    
                    # Add chunk metadata to each question
                    for q in question_set.questions:
                        all_questions.append({
                            "question": q.question,
                            "answer": q.answer,
                            "difficulty": q.difficulty,
                            "source_chunk_id": chunk["id"],
                            "source_metadata": chunk.get("metadata", {}),
                        })
                
                except Exception as e:
                    st.warning(f"⚠️ Error processing chunk {idx + 1}: {e}")
                    continue
            
            # Save questions to file
            status_text.text("Saving questions to file...")
            with open(EVAL_FILE, "w") as f:
                json.dump({
                    "questions": all_questions,
                    "total_questions": len(all_questions),
                    "chunks_processed": total_chunks,
                    "questions_per_chunk": questions_per_chunk,
                }, f, indent=2)
            
            progress_bar.progress(1.0, text="Complete!")
            status_text.text("")
            st.success(f"✅ Generated {len(all_questions)} questions from {total_chunks} chunks!")
            st.balloons()
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            import traceback
            st.code(traceback.format_exc())

with tab2:
    st.subheader("📋 View Test Questions")
    
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r") as f:
            eval_data = json.load(f)
        
        questions = eval_data.get("questions", [])
        total = eval_data.get("total_questions", 0)
        chunks_processed = eval_data.get("chunks_processed", 0)
        
        st.metric("Total Questions", total)
        st.metric("Chunks Processed", chunks_processed)
        
        if questions:
            st.divider()
            

            
            # Display questions
            for idx, q in enumerate(questions, 1):
                with st.expander(
                    f"Q{idx}: {q['question'][:80]}..." if len(q['question']) > 80 else f"Q{idx}: {q['question']}",
                    expanded=False
                ):
                    st.markdown(f"**Question:**  \n{q['question']}")
                    st.markdown(f"**Answer:**  \n{q['answer']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Chunk ID: {q.get('source_chunk_id', 'N/A')}")
                    with col2:
                        st.caption(f"Source: {q.get('source_metadata', {}).get('source', 'N/A')}")
            
            # Download button
            st.divider()
            st.download_button(
                label="📥 Download Test Set (JSON)",
                data=json.dumps(eval_data, indent=2),
                file_name="test_questions.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("No test questions generated yet. Go to the 'Generate Test Set' tab to create one.")

with tab3:
    st.subheader("🧪 Run Evaluation")
    st.markdown(
        """
        Evaluate the RAG chatbot using the generated test questions. 
        This will measure retrieval metrics (precision, recall, chunk rank) and 
        answer quality metrics (faithfulness, correctness).
        """
    )
    
    # Check if test questions exist
    if not EVAL_FILE.exists():
        st.warning("⚠️ No test questions found. Please generate test questions first in the 'Generate Test Set' tab.")
        st.stop()
    
    # Load test questions
    with open(EVAL_FILE, "r") as f:
        eval_data = json.load(f)
    
    questions = eval_data.get("questions", [])
    if not questions:
        st.warning("⚠️ No questions found in the test set.")
        st.stop()
    
    st.info(f"📊 Found {len(questions)} test questions ready for evaluation")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        max_questions = st.number_input(
            "Max questions to evaluate",
            min_value=1,
            max_value=len(questions),
            value=min(10, len(questions)),
            help="Limit the number of questions to evaluate (useful for testing)"
        )
    
    with col2:
        k_neighbors = st.number_input(
            "K (top-k)",
            min_value=1,
            max_value=20,
            value=SETTINGS.k_neighbors,
            help="Number of chunks to retrieve for evaluation"
        )
    
    # Path to store evaluation results
    EVAL_RESULTS_FILE = EVAL_DATA_PATH / "evaluation_results.json"
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())
        
        progress_bar = st.progress(0, text="Starting evaluation...")
        status_text = st.empty()
        
        try:
            results = []
            questions_to_eval = questions[:max_questions]
            
            for idx, q_data in enumerate(questions_to_eval):
                progress = (idx + 1) / len(questions_to_eval)
                progress_bar.progress(progress, text=f"Evaluating question {idx + 1}/{len(questions_to_eval)}...")
                status_text.text(f"Processing: {q_data['question'][:60]}...")
                
                # Convert to TestQuestion model
                test_question = TestQuestion(
                    question=q_data["question"],
                    answer=q_data["answer"],
                    source_chunk_id=q_data["source_chunk_id"],
                    source_metadata=q_data.get("source_metadata"),
                )
                
                # Evaluate the test case
                result = evaluate_test_case(
                    test_question=test_question,
                    base_url="http://localhost",
                    openai_client=openai_client,
                    k=k_neighbors,
                )
                results.append(result)

            
            if not results:
                st.error("❌ No results were generated. Please check the API and try again.")
                st.stop()
            
            # Calculate aggregate metrics
            status_text.text("Calculating aggregate metrics...")
            aggregate = aggregate_results(results)
            
            # Save results
            status_text.text("Saving results...")
            results_data = {
                "aggregate_metrics": aggregate.model_dump(),
                "individual_results": [
                    {
                        "question": r.test_question.question,
                        "answer": r.test_question.answer,
                        "generated_answer": r.chat_output.message,
                        "source_chunk_id": r.test_question.source_chunk_id,
                        "chunk_in_top_k": r.metrics.chunk_in_top_k,
                        "chunk_rank": r.metrics.chunk_rank,
                        "faithfulness_score": r.metrics.faithfulness_score,
                        "faithfulness_reasoning": r.faithfulness_reasoning,
                        "answer_correctness_score": r.metrics.answer_correctness_score,
                        "correctness_reasoning": r.correctness_reasoning,
                        "retrieved_chunk_ids": r.metrics.retrieved_chunk_ids,
                        "num_chunks_retrieved": r.metrics.num_chunks_retrieved,
                    }
                    for r in results
                ],
                "evaluation_config": {
                    "k_neighbors": k_neighbors,
                    "total_questions_evaluated": len(results),
                    "max_questions": max_questions,
                },
            }
            
            with open(EVAL_RESULTS_FILE, "w") as f:
                json.dump(results_data, f, indent=2)
            
            progress_bar.progress(1.0, text="Complete!")
            status_text.text("")
            st.success(f"✅ Evaluation complete! Evaluated {len(results)} questions.")
            st.balloons()
            
            # Store results in session state for display
            st.session_state.eval_results = results_data
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Unexpected error during evaluation: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results if they exist
    st.divider()
    
    if EVAL_RESULTS_FILE.exists() or "eval_results" in st.session_state:
        # Load results from file if not in session state
        if "eval_results" not in st.session_state:
            with open(EVAL_RESULTS_FILE, "r") as f:
                st.session_state.eval_results = json.load(f)
        
        results_data = st.session_state.eval_results
        aggregate = results_data["aggregate_metrics"]
        individual = results_data["individual_results"]
        config = results_data["evaluation_config"]
        
        st.subheader("📊 Evaluation Results")
        
        # Display aggregate metrics
        st.markdown("### 🎯 Aggregate Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Precision@K",
                f"{aggregate['precision_at_k']:.2%}",
                help="Proportion of queries where relevant chunk was in top-k"
            )
        
        with col2:
            st.metric(
                "Recall@K",
                f"{aggregate['recall_at_k']:.2%}",
                help="Proportion of relevant chunks retrieved"
            )
        
        with col3:
            avg_rank = aggregate.get('average_chunk_rank')
            st.metric(
                "Avg Chunk Rank",
                f"{avg_rank:.2f}" if avg_rank is not None else "N/A",
                help="Average rank of source chunks when found"
            )
        
        with col4:
            st.metric(
                "Avg Faithfulness",
                f"{aggregate['average_faithfulness']:.2%}",
                help="How well answers are grounded in context"
            )
        
        with col5:
            st.metric(
                "Avg Correctness",
                f"{aggregate['average_correctness']:.2%}",
                help="Semantic similarity to ground truth"
            )
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Test Cases", aggregate["total_test_cases"])
        
        with col2:
            st.metric("Chunks Found in Top-K", aggregate["chunks_found_in_top_k"])
        
        with col3:
            st.metric("Chunks Not Found", aggregate["chunks_not_found"])
        
        st.caption(f"Evaluated with k={config['k_neighbors']} neighbors")
        
        # Individual results table
        st.markdown("### 📋 Individual Results")
        
        # Create DataFrame for display
        df_data = []
        for idx, result in enumerate(individual, 1):
            df_data.append({
                "#": idx,
                "Question": result["question"][:50] + "..." if len(result["question"]) > 50 else result["question"],
                "In Top-K": "✅" if result["chunk_in_top_k"] else "❌",
                "Rank": result["chunk_rank"] if result["chunk_rank"] is not None else "N/A",
                "Faithfulness": f"{result['faithfulness_score']:.2%}",
                "Correctness": f"{result['answer_correctness_score']:.2%}",
                "Retrieved": result["num_chunks_retrieved"],
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
        
        # Detailed results (expandable)
        st.markdown("### 🔍 Detailed Results")
        
        for idx, result in enumerate(individual, 1):
            with st.expander(f"Question {idx}: {result['question'][:60]}...", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Question:**")
                    st.write(result["question"])
                    
                    st.markdown("**✅ Expected Answer:**")
                    st.write(result["answer"])
                
                with col2:
                    st.markdown("**🤖 Generated Answer:**")
                    st.write(result["generated_answer"])
                
                st.divider()
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("In Top-K", "✅ Yes" if result["chunk_in_top_k"] else "❌ No")
                
                with col2:
                    rank = result["chunk_rank"]
                    st.metric("Chunk Rank", rank if rank is not None else "Not Found")
                
                with col3:
                    st.metric("Faithfulness", f"{result['faithfulness_score']:.2%}")
                
                with col4:
                    st.metric("Correctness", f"{result['answer_correctness_score']:.2%}")
                
                # Reasoning
                st.markdown("**🧠 Faithfulness Reasoning:**")
                st.caption(result["faithfulness_reasoning"])
                
                st.markdown("**🧠 Correctness Reasoning:**")
                st.caption(result["correctness_reasoning"])
                
                # Retrieved chunks
                st.markdown(f"**📚 Retrieved Chunks ({result['num_chunks_retrieved']}):**")
                st.caption(f"Source chunk ID: `{result['source_chunk_id']}`")
                st.caption(f"Retrieved chunk IDs: {', '.join([f'`{cid}`' for cid in result['retrieved_chunk_ids'][:5]])}{'...' if len(result['retrieved_chunk_ids']) > 5 else ''}")
        
        # Download button
        st.divider()
        st.download_button(
            label="📥 Download Evaluation Results (JSON)",
            data=json.dumps(results_data, indent=2),
            file_name="evaluation_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    else:
        st.info("💡 Click 'Run Evaluation' above to evaluate the test questions and see results here.")

