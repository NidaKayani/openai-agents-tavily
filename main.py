import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
from dotenv import load_dotenv
# from exa_py import Exa
from tavily import TavilyClient

load_dotenv()
tavily_api = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api)

# Client and model setup (unchanged)
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

set_tracing_disabled(True)
model = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.5-flash",
)

# Exa search tool (new)
# EXA_API_KEY = os.getenv("EXA_API_KEY")
# exa = Exa(api_key=EXA_API_KEY)

# result = exa.search_and_contents(
#   "An article about the state of AGI",
#   type="auto",
#   text=True,
# )
# print(result)
@function_tool
def tavily_search(query: str) -> str:
    """
    Perform a web search using Tavily and return a summarized result.
    """
    response = tavily_client.search(query,search_depth='advanced',max_results='5')
    results = response.get("results", [])
    return results or "No results found."
# Existing weather tool (unchanged)
@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is rainy."

# Agent setup (updated)
agent = Agent(
    name="Assistant Agent",
    instructions="You are a helpful assistant that can answer questions and help with tasks. For questions that require current information, use your tavily_search tool. For weather questions, use your weather tool.",
    model=model,
    tools=[tavily_search],  # Add both tools here
)

# Running the agent
response = Runner.run_sync(agent, "What were the latest developments in large language models?")
print(response.final_output)
