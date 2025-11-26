import streamlit as st
from LangGraph_Backend import workflow
from langchain_core.messages import HumanMessage
import uuid  # to generate unique thread ids per user

st.title("MG Apparel Chatbot")

# Create a unique thread ID per Streamlit session
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

print(f"Thread ID for this session: {st.session_state['thread_id']}")
# Initialize message history
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

# Display previous chat history
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field
user_input = st.chat_input("Type here...")

if user_input:
    # Add user message
    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare configuration for LangGraph
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # Invoke the LangGraph workflow
    response = workflow.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    # Extract AI message
    ai_message = response["messages"][-1].content

    # Save AI response to session history
    st.session_state["messages_history"].append({"role": "assistant", "content": ai_message})

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_message)
