from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

os.environ['LANGCHAIN_PROJECT'] = 'Agents in LangGraph'

load_dotenv()
# =================================== LLM ===================================
llm = ChatOpenAI(model="gpt-4o-mini")
# =================================== TOOLS ===================================

@tool
def date_time() -> dict:
    """
    Get the current date and time.
    """
    from datetime import datetime
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}
@tool
def search(query: str) -> dict:
    """
    Perform a web search using DuckDuckGo and return the results and also any query related to time.
    """
    search_tool = DuckDuckGoSearchRun(region="us-en")
    results = search_tool.run(query)
    return {"query": query, "results": results}
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
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
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=VSHTNXI6SXNLAMAD"
    r = requests.get(url)
    return r.json()



tools = [date_time,search, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# =================================== State ================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =================================== Nodes ===================================

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# =================================== Checkpointer ===================================


conn = sqlite3.connect(database='chatbot.db' , check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

# =================================== Graph ===================================


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
# Edge conditions
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools", 'chat_node')
# graph.add_edge("chat_node", END)
workflow = graph.compile(checkpointer=checkpointer)

# =================================== HELPER ===================================


def retrieve_all_threads():
    all_thread_ids = set()
    for checkpoints in checkpointer.list(None):
        all_thread_ids.add(checkpoints.config['configurable']['thread_id'])
    return list(all_thread_ids)


if __name__ == "__main__":
    thread_id = "1"
    while True:
        user_message = input("Type here: ")
        print("User:", user_message)
        if user_message.strip().lower() in ["exit", "quit", "bye"]:
            break
        config = {"configurable": {"thread_id": thread_id}}
        response = workflow.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)
        print("AI:", response["messages"][-1].content)


# for message_chunk , metadata in workflow.stream(
#     {"messages": [HumanMessage(content="Tell me how to make a paper airplane.") ]},
#     config={"configurable": {"thread_id": "test_thread"}},
#     # chunk_size=10,
#     stream_mode= "messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content , end=' ', flush=True)

