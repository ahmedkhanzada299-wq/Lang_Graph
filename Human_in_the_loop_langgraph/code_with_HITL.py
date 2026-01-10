import os
import requests
import sqlite3  # Added for sync connection
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Specify model and temperature for consistency
llm_with_tools = None  # Will bind after tools

# -------------------
# 2. Tools
# -------------------
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> str:
    """
    Simulate purchasing a stock. Requires human approval via HITL interrupt.
    """
    if quantity <= 0:
        return "Quantity must be positive."

    payload = {
        "question": f"Do you approve purchasing {quantity} shares of {symbol.upper()}?",
        "details": f"Symbol: {symbol.upper()} | Quantity: {quantity}",
        "instructions": "Respond with 'yes' to approve or anything else to decline."
    }

    decision = interrupt(payload)

    # Flexible approval: starts with 'y' (yes, y, yeah, etc.)
    if isinstance(decision, str) and decision.lower().strip().startswith("y"):
        return f"Purchase approved and executed: {quantity} shares of {symbol.upper()}."
    else:
        return f"Purchase declined: {quantity} shares of {symbol.upper()}."


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------
def agent(state: ChatState) -> dict:
    """Agent node: LLM decides to answer or call tools."""
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error in agent: {str(e)}")]}


tool_node = ToolNode(tools=tools)


# -------------------
# 5. Graph
# -------------------
graph = StateGraph(state_schema=ChatState)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# Use SqliteSaver with sync connection for synchronous code
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)  # File-based for persistence across runs
memory = SqliteSaver(conn=conn)
chatbot = graph.compile(checkpointer=memory)

# -------------------
# 6. CLI with HITL handling and error tolerance
# -------------------
if __name__ == "__main__":
    thread_id = "stock-bot-thread-persistent"

    print("Stock Trading Bot (type 'exit' or 'quit' to end)\n")
    print("Note: Conversation state is persisted across runs with SQLite checkpointing.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            conn.close()  # Clean up connection on exit
            break
        if not user_input:
            continue

        config = {"configurable": {"thread_id": thread_id}}
        input_state = {"messages": [HumanMessage(content=user_input)]}
        result = {}  # Initialize result to avoid NameError

        # Loop to handle potential multiple interrupts (future-proof)
        while True:
            try:
                result = chatbot.invoke(input_state, config=config)
            except Exception as e:
                print(f"Graph execution error: {str(e)}")
                result = {}  # Reset on error
                break

            interrupts = result.get("__interrupt__", [])
            if not interrupts:
                break  # No interrupt → done

            # Handle interrupt (assume one, but iterable)
            for intr in interrupts:
                payload = intr.value if hasattr(intr, "value") else intr
                print("\n" + "="*50)
                print("HUMAN APPROVAL REQUIRED")
                print("="*50)
                if isinstance(payload, dict):
                    print(payload.get("question", "Approval needed."))
                    print(payload.get("details", ""))
                    print(payload.get("instructions", "yes/no"))
                else:
                    print(payload)
                print("="*50)
                decision = input("Your decision: ").strip()

                # Resume with decision
                input_state = Command(resume=decision)

        # Display final response safely
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"\nBot: {last_msg.content}\n")
            else:
                # Fallback to last non-empty content
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content:
                        print(f"\nBot: {msg.content}\n")
                        break
        else:
            print("\nBot: No response generated.\n")