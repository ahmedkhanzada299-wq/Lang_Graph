import streamlit as st
from LangGraph_Backend import workflow , retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

st.title("MG Apparel Chatbot")


# ======================== Helpers =========================

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    """Create a completely fresh chat session."""
    new_thread = generate_thread_id()
    st.session_state["thread_id"] = new_thread
    add_thread(new_thread)

    # Initial message from bot
    st.session_state["messages_history"] = [
        {"role": "assistant", "content": "🆕 New conversation started!"}
    ]

    st.rerun()   # Force refresh


def add_thread(thread_id):
    """Save thread ID in sidebar list."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    """Safely load previous conversation from LangGraph."""
    state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    values = state.values

    if "messages" in values:
        return values["messages"]
    if "chat_history" in values:
        return values["chat_history"]
    if "history" in values:
        return values["history"]
    if "conversation" in values:
        return values["conversation"]

    return []


def get_chat_title(messages):
    """Generate chat title from first user message."""
    if not messages:
        return "New Chat"

    first_user_msg = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            first_user_msg = msg.content
            break

    if not first_user_msg:
        return "New Chat"

    words = first_user_msg.split()
    short_title = " ".join(words[:7])
    return short_title + ("..." if len(words) > 7 else "")


# ======================== Initialize state =========================

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


# ======================= Sidebar ========================

st.sidebar.title("MG Apparel History Chatbot")

# NEW CHAT BUTTON
if st.sidebar.button("New Chat", key="new_chat_btn"):
    reset_chat()

st.sidebar.header("Previous Chats")

# List all previous chats
for thread_id in st.session_state["chat_threads"][::-1]:

    # Load title
    messages = load_conversation(thread_id)
    chat_title = get_chat_title(messages)

    # Button MUST have unique key
    if st.sidebar.button(chat_title, key=f"chat_{thread_id}"):

        st.session_state["thread_id"] = thread_id

        # Load message history
        loaded_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            loaded_messages.append({"role": role, "content": msg.content})

        st.session_state["messages_history"] = loaded_messages
        st.rerun()  # Update UI immediately


# ====================== Display chat ======================

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ====================== Chat input ======================

user_input = st.chat_input("Type here...")

if user_input:
    # Add user message
    st.session_state["messages_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }
    # Stream assistant message
    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

        st.session_state["messages_history"].append(
            {"role": "assistant", "content": ai_message}
        )
 