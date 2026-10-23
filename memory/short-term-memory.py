from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient
import os
from pydantic import BaseModel

class answer(BaseModel):
    answer: str
    confidence: float

load_dotenv()  # Load environment variables from .env file

langfuse_handler = CallbackHandler() 
tavily_client = TavilyClient()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("api key not found in .env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

config={
        "configurable": {"thread_id": "1"},
        "callbacks": [langfuse_handler]
        }

def tavily_web_search(query: str) -> str:
    """Search the web for relevant information."""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5
        )

        return str(response.get("results", []))

    except Exception as e:
        print(f"Tavily API error: {e}")
        return "Web search failed. Please try again."

system_prompt = """
you are a helpful research agent that can access the internet and perform research tools.

you have access to the following tools:

<tools>

this tool return the results of a web search using the tavily websearch tool, you can search the web for relevant information.

</tools>
"""

agents = create_agent(
    model=llm,
    tools=[tavily_web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

agent_response = agents.invoke({
    "messages": [{
        "role": "user",
        "content": "What is the gdp of india in 2026?"
    }]
},config=config)

final_message = agent_response["messages"][-1]
final_text = final_message.content

structured_llm = llm.with_structured_output(
        answer,
        method="json_schema"
    )

structured_response = structured_llm.invoke(
        f"""
        Convert the following research result into the required schema.
        Provide an answer and a confidence score between 0 and 1.

        Research result:
        {final_text}
        """
    )

print("Answer:", structured_response.answer)
print("Confidence:", structured_response.confidence)

