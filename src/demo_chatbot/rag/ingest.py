"""Load the FAQ file, turn it into searchable chunks, and expose a simple search()."""
from __future__ import annotations

# Path = an easy way to find and read files
from pathlib import Path

# the two pieces you already built
from demo_chatbot.rag.embeddings import HashingEmbedder
from demo_chatbot.rag.store import Chunk, InMemoryVectorStore


class Retriever:
    """Bundles the embedder + the filled store, so callers just do .search(question)."""

    def __init__(self, embedder: HashingEmbedder, store: InMemoryVectorStore) -> None:
        # remember the embedder (turns text into vectors)
        self._embedder = embedder
        # remember the store (holds chunks and searches them)
        self._store = store

    def search(self, query: str, k: int = 3):
        # turn the user's question into a vector (encode expects a list, so [query], then take [0])
        query_vector = self._embedder.encode([query])[0]
        # ask the store for the k closest chunks to that vector
        return self._store.search(query_vector, k=k)


def build_retriever(kb_dir: str = "data/knowledge_base") -> Retriever:
    # create a fresh embedder
    embedder = HashingEmbedder()
    # create an empty store
    store = InMemoryVectorStore()

    # this list will collect every chunk from every file
    chunks: list[Chunk] = []

    # loop over every .md file in the knowledge base folder (sorted for stable order)
    for md_file in sorted(Path(kb_dir).glob("*.md")):
        # read the whole file into a string
        text = md_file.read_text(encoding="utf-8")
        # split the file into blocks separated by blank lines; number each block with i
        for i, block in enumerate(b.strip() for b in text.split("\n\n")):
            # skip empty blocks
            if block:
                # make a Chunk (id, the text, which file it came from) and keep it
                chunks.append(Chunk(id=f"{md_file.name}:{i}", text=block, source=md_file.name))

    # if the folder had no usable text, fail with a clear message instead of silently
    if not chunks:
        raise RuntimeError(f"No .md documents found in {kb_dir}")

    # embed all chunk texts at once -> a grid of vectors
    vectors = embedder.encode([c.text for c in chunks])
    # load the chunks + their vectors into the store
    store.add(chunks, vectors)

    # hand back a ready-to-use Retriever
    return Retriever(embedder, store)