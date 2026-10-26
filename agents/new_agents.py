
from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph_checkpoint_aws import DynamoDBSaver
from tavily import TavilyClient

import json
import os
import uuid

load_dotenv()

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# Clients
langfuse_handler = CallbackHandler()
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

checkpoint_saver = DynamoDBSaver(
    table_name="research_agent_checkpoints",
    region_name=AWS_REGION,
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

system_prompt = """
You are a helpful research agent.
Use web search when current or factual information is needed.
Base your answer on relevant evidence returned by the search tool.
Summarize findings concisely and avoid repeating source content.
Mention uncertainty when the available evidence is insufficient.
"""

@tool
def web_search(query: str) -> str:
    """Search the web and return a small set of relevant results."""

    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=3,
    )

    # Return only useful fields and limit each snippet.
    results = []

    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")[:800],
        })

    return json.dumps(results, ensure_ascii=False)


def main():
    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=system_prompt,
        checkpointer=checkpoint_saver,
    )

    # A fresh thread for an independent research run.
    # Reuse the same ID when continuing an existing conversation.
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "callbacks": [langfuse_handler],
    }

    user_query = "defination of artificial intelligence"

    agent_response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        },
        config=config,
    )

    print("Thread ID:", thread_id)
    print("Agent Response 1:", agent_response["messages"][-1].content)

    agent_response2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what was the previous question?",
                }
            ]
        },
        config=config,
    )

    print("Agent Response 2:", agent_response2["messages"][-1].content)


if __name__ == "__main__":
    main()
