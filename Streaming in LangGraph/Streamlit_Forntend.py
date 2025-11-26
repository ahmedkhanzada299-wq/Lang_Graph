import streamlit as st
from LangGraph_Backend import workflow
from langchain_core.messages import HumanMessage
import uuid

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


    with st.chat_message("assistant"):
        ai_message =  st.write_stream(
            message_chunk.content for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
        )
        )

        st.session_state["messages_history"].append({"role": "assistant", "content": ai_message})
# Placeholder for streaming message
    # response_placeholder = st.write_stream("assistant")
    # placeholder = response_placeholder.empty()
    # streamed_text = ""

    # # Invoke the LangGraph workflow
    # for message_chunk, metadata in workflow.stream(
    #     {"messages": [HumanMessage(content=user_input)]},
    #     config=config,
    #     stream_mode="messages"
    # ):
    #     # Extract chunk content safely
    #     ai_chunk = getattr(message_chunk, "content", None)
    #     if not ai_chunk:
    #         continue

    #     streamed_text += ai_chunk  # accumulate stream

    #     # Update assistant message live
    #     placeholder.markdown(streamed_text)

    # # Save final assistant message to history
    # st.session_state["messages_history"].append({
    #     "role": "assistant",
    #     "content": streamed_text
    # })