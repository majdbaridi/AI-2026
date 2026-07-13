"""API: search the FAQ, then let the LLM answer using ONLY those results."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from demo_chatbot.llm.client import build_llm
from demo_chatbot.rag.ingest import build_retriever

app = FastAPI(title="railway operator Assistant")

# built once at startup (not per request)
llm = build_llm()
retriever = build_retriever()   # loads + indexes the FAQ


class AskRequest(BaseModel):
    message: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "provider": type(llm).__name__}


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    # 1. search the FAQ for chunks relevant to the question
    hits = retriever.search(req.message, k=3)

    # 2. glue the found chunks into an "evidence" block
    evidence = "\n".join(f"- {h.chunk.text}" for h in hits)

    # 3. tell the LLM to answer ONLY from that evidence
    system = (
        "You are a transport provider train assistant. "
        "Answer ONLY using the FACTS below. If they don't cover it, say you don't know. "
        "Be brief.\n\nFACTS:\n" + evidence
    )

    # 4. get the grounded answer
    answer = await llm.complete(system, req.message)

    # 5. return the answer AND which chunks were used (so you can see RAG working)
    return {
        "answer": answer,
        "used_chunks": [h.chunk.text for h in hits],
    }