## RAG Chain Flow with LCEL

This project uses a RAG pipeline.

RAG means:

> Retrieve relevant information first, then ask the LLM to answer using that information.

### Main Chain

```python
retrieval_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
)


How it works
User Question
   ↓
Extract the question
   ↓
Retriever searches Pinecone
   ↓
Top 3 relevant chunks are returned
   ↓
Chunks are formatted into one context string
   ↓
Prompt template receives:
   - question
   - context
   ↓
LLM generates an answer
   ↓
Output parser returns clean text



Expected behavior:

1. The app connects to Pinecone.
2. It searches for the most relevant chunks.
3. It sends those chunks with the question to the LLM.
4. It prints a grounded answer based on the retrieved context.

Example output:

Answer:
Pinecone is a vector database used to store and search embeddings. 
In machine learning, it is commonly used for semantic search and RAG systems...