Website documentation
→ extract text
→ convert to LangChain Documents
→ split into chunks
→ create embeddings
→ store in Pinecone





Step 1 — Crawl docs
TavilyCrawl

collects documentation text.

Step 2 — Convert to Documents

LangChain standard format.

Step 3 — Split into chunks

Because:

huge pages are bad for embeddings
AI retrieval works better on smaller chunks
Step 4 — Generate embeddings

OpenAI converts text into vectors:

Note: this classic RAG 