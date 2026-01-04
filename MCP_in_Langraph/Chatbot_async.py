import os
import aiosqlite
import requests
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated

# =================================== ENV ===================================
os.environ['LANGCHAIN_PROJECT'] = 'Agents in LangGraph'
load_dotenv()

# =================================== LLM ===================================
llm = ChatOpenAI(model="gpt-4o-mini")

# =================================== TOOLS ===================================
@tool
def date_time() -> dict:
    """Return the current date and time."""
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}

@tool
def search(query: str) -> dict:
    """Perform a DuckDuckGo search and return the results."""
    search_tool = DuckDuckGoSearchRun(region="us-en")
    results = search_tool.run(query)
    return {"query": query, "results": results}

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a symbol using Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=VSHTNXI6SXNLAMAD"
    r = requests.get(url)
    return r.json()


tools = [date_time, search, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# =================================== STATE ===================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =================================== NODES ===================================
async def chat_node(state: ChatState):
    """Async LLM node that answers messages or requests a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# =================================== CHECKPOINTER ===================================
async def get_async_checkpointer():
    conn = await aiosqlite.connect('chatbot_async.db')
    return AsyncSqliteSaver(conn=conn)

# =================================== GRAPH ===================================
async def build_graph():
    checkpointer = await get_async_checkpointer()
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    workflow = graph.compile(checkpointer=checkpointer)
    return workflow, checkpointer

# =================================== HELPER ===================================
async def retrieve_all_threads(checkpointer: AsyncSqliteSaver):
    """
    Retrieve all thread IDs stored in the Async SQLite checkpointer.
    """
    all_thread_ids = set()
    async for checkpoints in checkpointer.alist(None):
        all_thread_ids.add(checkpoints.config['configurable']['thread_id'])
    return list(all_thread_ids)

# =================================== MAIN ===================================
async def main():
    workflow, checkpointer = await build_graph()
    thread_id = "1"

    while True:
        user_message = input("Type here: ")
        print("User:", user_message)
        if user_message.strip().lower() in ["exit", "quit", "bye"]:
            break
        config = {"configurable": {"thread_id": thread_id}}
        response = await workflow.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config
        )
        print("AI:", response["messages"][-1].content)

    # Example: retrieve all threads at the end
    threads = await retrieve_all_threads(checkpointer)
    print("All saved thread IDs:", threads)

if __name__ == "__main__":
    asyncio.run(main())
