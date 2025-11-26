from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
import os

os.environ['LANGCHAIN_PROJECT'] = 'Observability in LangGraph'

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db' , check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
workflow = graph.compile(checkpointer=checkpointer)

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

