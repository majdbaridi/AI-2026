from operator import itemgetter
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_retrieval_chain():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    chain = create_retrieval_chain()

    query = "what is Pinecone in machine learning?"

    result = chain.invoke({"question": query})

    print("\nAnswer:")
    print(result)