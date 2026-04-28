from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyResearch

load_dotenv()


class Job(BaseModel):
    """One job result returned by the agent."""

    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location")
    link: str = Field(description="Direct URL to the job posting")
    why_it_fits: str = Field(description="Why this job matches the requested criteria")


class AgentResponse(BaseModel):
    """Structured response for job search results."""

    jobs: List[Job] = Field(description="List of relevant AI Engineer job results")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [TavilyResearch()]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Search for AI Engineer jobs in Germany focused on Python, "
                        "LLMs, RAG, LangChain, and agentic AI. Return top results with "
                        "title, company, location, link, and why they fit."
                    )
                )
            ]
        }
    )

    structured: AgentResponse = result["structured_response"]

    for job in structured.jobs:
        print(f"Title: {job.title}")
        print(f"Company: {job.company}")
        print(f"Location: {job.location}")
        print(f"Link: {job.link}")
        print(f"Why it fits: {job.why_it_fits}")
        print("-" * 50)


if __name__ == "__main__":
    main()