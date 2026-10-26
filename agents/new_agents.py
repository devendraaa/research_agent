from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph_checkpoint_aws import DynamoDBSaver
from tavily import TavilyClient
from langchain.agents.middleware import SummarizationMiddleware

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

system_prompt_research_agent = ""
research_agent = create_agent(model=llm, system_prompt=system_prompt_research_agent)

system_prompt_calculator_agent = ""
calculator_agent = create_agent(model=llm, system_prompt=system_prompt_calculator_agent)

system_prompt = """

Please always greet the user by their name available in the memory.

You are a helpful research agent.
<tools>
Here are the list of tools and descriptions and input schema utilize them as per need:
make sure you know what user is asking for and use the tool correctly.
<tool>
name : web_search 
description : this tool help in web search of an given query and return a small set of relevant results.
inputSchema : query as a string
</tool>

<tool>
name : research
description : Perform research on a given topic and return a summary of findings.
inputSchema : query as a string
</tool>

<tool>
name : calculator
description : Evaluate a mathematical expression and return the result.
inputSchema : query as a string
</tool>


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

@tool
def research(query: str) -> str:
    """Perform research on a given topic and return a summary of findings."""

    response = research_agent.invoke({
        "messages": [{
            "role": "user", 
            "content": query
            }]}, 
            )
    
    return response["messages"][-1].content

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    response = calculator_agent.invoke({
        "messages": [{
            "role": "user", 
            "content": expression
            }]}, 
       )
    return response["messages"][-1].content

def main():
    
    agent = create_agent(
        model=llm,
        system_prompt=system_prompt,
        checkpointer=checkpoint_saver,
        middleware=[
            SummarizationMiddleware(
                model=llm,      
                trigger={"tokens" : 4000},
                keep=("messages", 15)
            )],
            tools=[web_search, research, calculator],
    )

    config = {
        "configurable": {
            "thread_id": "dev_004"
        },
        "callbacks": [langfuse_handler],
    }

    user_query = "what is 2+2?"

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

    print("Thread ID:", config["configurable"]["thread_id"])
    print("Agent Response 1:", agent_response["messages"][-1].content)


if __name__ == "__main__":
    main()
