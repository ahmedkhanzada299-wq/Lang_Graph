import streamlit as st
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.types import interrupt, Command

# Import the graph from backend
from code_with_HITL import chatbot as workflow  # Assuming backend file is code_with_HITL.py

st.title("MG Apparel Stock Chatbot")

# ======================== Helpers =========================

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    new_thread = generate_thread_id()
    st.session_state["thread_id"] = new_thread
    add_thread(new_thread)

    st.session_state["messages_history"] = [
        {"role": "assistant", "content": "🆕 New conversation started!"}
    ]

    # Reset HITL states
    st.session_state["hitl_pending"] = False
    st.session_state["interrupt_payload"] = None
    st.rerun()

def load_conversation(thread_id):
    state = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    values = state.values

    for key in ("messages",):
        if key in values:
            return values[key]
    return []

def get_chat_title(messages):
    for msg in messages:
        if isinstance(msg, HumanMessage):
            words = msg.content.split()
            title = " ".join(words[:7])
            return title + ("..." if len(words) > 7 else "")
    return "New Chat"

# ======================== Session Init =========================

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "hitl_pending" not in st.session_state:
    st.session_state["hitl_pending"] = False

if "interrupt_payload" not in st.session_state:
    st.session_state["interrupt_payload"] = None

add_thread(st.session_state["thread_id"])

# ======================= Sidebar ========================

st.sidebar.title("MG Apparel Stock Bot")
st.sidebar.markdown(f"**Thread ID:** `{st.session_state['thread_id']}`")

if st.sidebar.button("New Chat", key="new_chat_btn"):
    reset_chat()

# -------- Previous Chats --------
st.sidebar.header("Previous Chats")

for tid in st.session_state["chat_threads"][::-1]:
    msgs = load_conversation(tid)
    title = get_chat_title(msgs)

    if st.sidebar.button(title, key=f"chat_{tid}"):
        st.session_state["thread_id"] = tid

        temp = []
        for msg in msgs:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            temp.append({"role": role, "content": content})

        st.session_state["messages_history"] = temp
        st.session_state["hitl_pending"] = False
        st.session_state["interrupt_payload"] = None
        st.rerun()

# ====================== Chat Display ======================

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ====================== HITL UI ======================

if st.session_state["hitl_pending"]:
    payload = st.session_state["interrupt_payload"]
    with st.container():
        st.warning("Human Approval Required!")
        st.markdown(f"**Question:** {payload.get('question', 'Approval needed.')}")
        st.markdown(f"**Details:** {payload.get('details', '')}")
        st.markdown(f"**Instructions:** {payload.get('instructions', 'yes/no')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve (Yes)", key="approve_btn"):
                decision = "yes"
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                result = workflow.invoke(Command(resume=decision), config=config)
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    ai_content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    st.session_state["messages_history"].append({"role": "assistant", "content": ai_content})
                st.session_state["hitl_pending"] = False
                st.session_state["interrupt_payload"] = None
                st.rerun()
        with col2:
            if st.button("Decline (No)", key="decline_btn"):
                decision = "no"
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                result = workflow.invoke(Command(resume=decision), config=config)
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    ai_content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    st.session_state["messages_history"].append({"role": "assistant", "content": ai_content})
                st.session_state["hitl_pending"] = False
                st.session_state["interrupt_payload"] = None
                st.rerun()

# ====================== Chat Input ======================

else:
    user_input = st.chat_input("Type here...")

    if user_input:
        st.session_state["messages_history"].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        input_state = {"messages": [HumanMessage(content=user_input)]}

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                result = workflow.invoke(input_state, config=config)

            interrupts = result.get("__interrupt__", [])
            if interrupts:
                payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                st.session_state["interrupt_payload"] = payload if isinstance(payload, dict) else {"question": str(payload)}
                st.session_state["hitl_pending"] = True
                st.rerun()
            else:
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    ai_content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    st.markdown(ai_content)
                    st.session_state["messages_history"].append(
                        {"role": "assistant", "content": ai_content}
                    )
                else:
                    st.markdown("No response generated.")