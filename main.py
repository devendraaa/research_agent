from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langchain.tools import tool
from tavily import TavilyClient
# from tavily.errors import TavilyError
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import os

load_dotenv()  # Load environment variables from .env file
langfuse_handler = CallbackHandler()
tavily_client = TavilyClient()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("api ket not found in .env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

system_prompt = "you are a research agent that can access the internet and perform research tools."

class answer(BaseModel):
    answer: str
    confidence: float

tavily_client = TavilyClient()

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

@tool
def weather(location: str) -> str:
    """this tool takes a location and returns the current weather information."""
    # Placeholder for weather functionality
    return f"Current weather in {location}: Sunny, 25°C"

def search(query: str) -> str:
    """this tool takes user query and returns the search results from the internet."""
    # Placeholder for search functionality
    return f"Search results for '{query}'"



def main():
    agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    tools=[tavily_web_search]
    )

    agent_response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "What is the gdp of india in 2026?"
        }] 
    },config={
        "callbacks": [langfuse_handler]
  
    })

    # Get the agent's final response
    final_message = agent_response["messages"][-1]
    final_text = final_message.content

    # Convert the final text into your Pydantic schema
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



    # print(f"Agent Response: {agent_response["structured_response"]}")

if __name__ == "__main__":
    main() 
