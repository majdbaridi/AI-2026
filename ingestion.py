import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl

from logger import Colors, log_error, log_header, log_info, log_success, log_warning


# Step 1: Load environment variables
load_dotenv()

# Step 2: Configure SSL certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Step 3: Configure OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# Step 4: Connect to Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2025",
    embedding=embeddings,
)

# Step 5: Configure Tavily crawler
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Add documents to Pinecone in batches."""
    log_header("VECTOR STORAGE PHASE")

    batches = [
        documents[i : i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore Indexing: Split {len(documents)} chunks into {len(batches)} batches",
        Colors.DARKCYAN,
    )

    successful = 0

    for batch_num, batch in enumerate(batches, start=1):
        try:
            await vectorstore.aadd_documents(batch)
            successful += 1

            log_success(
                f"VectorStore Indexing: Added batch {batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )

        except Exception as e:
            log_error(
                f"VectorStore Indexing: Failed batch {batch_num}/{len(batches)} - {e}"
            )

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully "
            f"({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully",
            Colors.YELLOW,
        )


async def main():
    """Run the documentation ingestion pipeline."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    # Step 6: Crawl website and extract readable content
    log_info(
        "TavilyCrawl: Starting to crawl LangChain documentation",
        Colors.PURPLE,
    )

    res = await tavily_crawl.ainvoke(
        {
            "url": "https://python.langchain.com/docs/",
            "max_depth": 2,
            "extract_depth": "advanced",
            "instructions": "Extract readable documentation content.",
        }
    )

    if not isinstance(res, dict) or "results" not in res:
        log_error(f"TavilyCrawl: Invalid response returned: {res}")
        return

    # Step 7: Convert Tavily results to LangChain Documents
    all_docs = []

    for item in res["results"]:
        url = item.get("url")
        raw_content = item.get("raw_content")

        if not raw_content:
            continue

        log_info(f"TavilyCrawl: Successfully crawled {url}")

        all_docs.append(
            Document(
                page_content=raw_content,
                metadata={"source": url},
            )
        )

    if not all_docs:
        log_error("Document Conversion: No valid documents extracted.")
        return

    log_success(f"Document Conversion: Created {len(all_docs)} documents")

    # Step 8: Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Step 9: Store chunks in Pinecone
    await index_documents_async(splitted_docs, batch_size=50)

    # Step 10: Finish
    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished successfully!")
    log_info(f"Documents extracted: {len(all_docs)}")
    log_info(f"Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())