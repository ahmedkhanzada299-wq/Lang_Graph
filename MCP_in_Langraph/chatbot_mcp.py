import os
import aiosqlite
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
from typing import TypedDict, Annotated
from langchain_mcp_adapters.client import MultiServerMCPClient

# =================================== ENV ===================================
os.environ['LANGCHAIN_PROJECT'] = 'Agents in LangGraph'
load_dotenv()

# =================================== LLM ===================================
llm = ChatOpenAI(model="gpt-4o-mini")

# =================================== MCP TOOLS ===================================
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": r"C:\Users\AhmedKhan\AppData\Local\Programs\Python\Python311\Scripts\uv.exe",
            "args": [
                "run",
                "fastmcp",
                "run",
                r"F:\MCP\MCP-MATH-SERVER\main.py"
            ]
        },
        "expense": {
        "transport": "streamable_http",
        "url": "https://ahmed-khanzada.fastmcp.app/mcp"
    },
    }
)

# =================================== STATE ===================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =================================== CHECKPOINTER ===================================
async def get_async_checkpointer():
    conn = await aiosqlite.connect('chatbot_async.db')
    return AsyncSqliteSaver(conn=conn)

# =================================== MAIN GRAPH BUILDER ===================================
async def build_graph():
    # Get MCP tools
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    print(f"Loaded {len(tools)} tools from MCP servers.")

    # Define async chat node now that llm_with_tools exists
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Tool node
    tool_node = ToolNode(tools)

    # Async checkpointer
    checkpointer = await get_async_checkpointer()

    # Build graph
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
