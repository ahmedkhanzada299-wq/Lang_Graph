from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os

os.environ['LANGCHAIN_PROJECT'] = 'AGENT APP'


load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> dict:
    """
    This function fetches the current weather data for a given city.
    """
    import requests

    url = f'https://api.weatherstack.com/current?access_key=b349654fc238371c1ff1fc21e137ff93&query={city}'
    
    response = requests.get(url)
    
    return response.json()

# Use a deterministic LLM for agents (temperature=0) and specify a modern model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the agent using the new create_agent API (ReAct-style built-in)
agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful assistant. Use the available tools to answer questions step by step. "
        "Think about what tool to use, call it if needed, and provide the final answer."
    )
)

# No separate AgentExecutor; invoke directly on the agent
# To mimic verbose output, we can stream and print steps
def run_agent_with_streaming(input_query: str):
    print(f"\nRunning agent for: {input_query}\n")
    inputs = {"messages": [HumanMessage(content=input_query)]}
    final_state = None
    for chunk in agent.stream(inputs, stream_mode="values"):
        # Print intermediate steps for verbosity
        latest_messages = chunk.get("messages", [])
        if latest_messages:
            last_msg = latest_messages[-1]
            if last_msg.content:
                print(f"Agent thought: {last_msg.content}")
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"Tool calls: {[tc['name'] for tc in last_msg.tool_calls]}")
        final_state = chunk
    # Extract final output from the last message
    if final_state and "messages" in final_state:
        output = final_state["messages"][-1].content
    else:
        output = "No output generated."
    return {"output": output}

# Example queries you can test (uncomment one at a time):
# response = run_agent_with_streaming("What is the release date of Dhadak 2?")
# response = run_agent_with_streaming("What is the current temp of gurgaon")
response = run_agent_with_streaming("Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.")

# Current invocation from your original code
# response = run_agent_with_streaming("What is the current temp of Multan")
# print(response)

print(response['output'])