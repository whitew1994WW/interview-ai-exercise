# RAG Chatbot - Features & Improvements

My focus for my time during this excercise was on the RAG & chat testing approach. The reason for this is that without evaluation, it is difficult to know/understand where bottlenecks are and where time should be spent on improvement.

The eval system was built with the UI as the main point of interaction. The reason for this is that it is easier to inspect and understand results in a visual way, especially at this stage in a project where it is mostly PoC and there is high potential for change to the evaluation/chat system.

The UI code written is mostly throwaway code (little time spent tidying it up), as it will never be production code, and will mostly be used for debugging and development.

So far in this excercise I have:

- Loaded in the json schemas
- Updated metadata handling so that the url is stored in the chromadb
- Updated the demo UI so that you can test the vector DB in isolation
- Updated the chat UI so that you can see the retrieved chunks in the chat response
- Created methods and functions for evaluating the chatbots performance against key metrics (precision, recall, chunk rank, faithfulness, correctness) (here I have created these myself as it is not a lot of code and I would rather see exactly what is happening)
- Created a new UI for generating a synthetic test set and evaluating the chatbots performance against it
- Updated the prompt to instruct the chatbot to decline to answer with insufficient information.


Proposed improvements:
- The testing identified that many chunks are not being retrieved for questions. This indicates that the chunking strategy needs to be improved. On inspection of the strategy, it is not optimised. Json structure should be preserved in the chunking process not split across {} pairs, information on where in the json structure the information is located should be stored and used to guide the retrieval process (path within the json). Finally, keeping each API/webhook in a single chunk should be considered. (several strategies can be tried to see what improves the precision/recall and average chunk rank).
- Consider not using json at all and converting all the docs to structured markdown to make it easier for the LLM and humans to review (evaluate against the test set)
- A one line summary of the overall document could be included in the prompt to the chatbot to provide context of the retrieved chunks
- An agent (I vote for langgraph) instead of a prompt should be used, this wold allow the chatbot to be more fail proof (due to the ReAct loop). Aditionally, agentic RAG should be used to protect context quality incase irrelevant questions are asked by the user.
- Summarisation of longer conversations could be considered to reduce context length and improve performance in long conversations.
- Manually created test cases should be used to align the automated testing with user expectations (as not all chunks are useful to users, but all are used to generate questions in automated testing)
- Using several different embedding models to see which perform best against top line metrics
- Using a logging platform to track performance of the RAG chatbot for both online and offline metrics, to see progress and run experiments on new models/chunking/embedding strategies
- Improved testing of the API endpoints (general software testing)
- Implementing a persistant key-value store to keep the chat history between sessions
- Setting up the testing & build of the vector DB to execute automatically on a CI/CD pipeline each time changes are made (can even generate a test case name automatically based on the commit message e.g. chunking_improvements)
- If halucination is important to prevent then add a pre-response online evaluation to prevent messages with hallucinations from being sent to the user.
- Track p50/p99 latencies of the chatbot apis to identify poor user experiences

Current testing results (also in a json file in the eval_data folder):

<img width="1161" height="736" alt="Screenshot 2025-11-25 163106" src="https://github.com/user-attachments/assets/c1e6ba49-551f-46e5-aaf4-68ed2438fee0" />

