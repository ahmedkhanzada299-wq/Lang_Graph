import streamlit as st
import uuid

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from LangGraph_Backend import (
    workflow,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.title("MG Apparel Chatbot")

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

    # create empty doc store for this thread
    st.session_state["ingested_docs"].setdefault(new_thread, {})
    st.rerun()


def load_conversation(thread_id):
    state = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    values = state.values

    for key in ("messages", "chat_history", "history", "conversation"):
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
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])
st.session_state["ingested_docs"].setdefault(st.session_state["thread_id"], {})

# ======================= Sidebar ========================

st.sidebar.title("MG Apparel History Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{st.session_state['thread_id']}`")

if st.sidebar.button("New Chat", key="new_chat_btn"):
    reset_chat()

# -------- PDF Upload Section --------
thread_key = st.session_state["thread_id"]
thread_docs = st.session_state["ingested_docs"][thread_key]

if thread_docs:
    latest = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest['filename']}` "
        f"({latest['chunks']} chunks, {latest['documents']} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF for this chat", type=["pdf"]
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already indexed.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )

# -------- Previous Chats --------
st.sidebar.header("Previous Chats")

for tid in st.session_state["chat_threads"][::-1]:
    msgs = load_conversation(tid)
    title = get_chat_title(msgs)

    if st.sidebar.button(title, key=f"chat_{tid}"):
        st.session_state["thread_id"] = tid
        st.session_state["ingested_docs"].setdefault(tid, {})

        temp = []
        for msg in msgs:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp.append({"role": role, "content": msg.content})

        st.session_state["messages_history"] = temp
        st.rerun()

# ====================== Chat Display ======================

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ====================== Chat Input ======================

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state["messages_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for chunk, _ in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    tool = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"]:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    st.session_state["messages_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # -------- Document metadata footer --------
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Document: {doc_meta['filename']} "
            f"(chunks: {doc_meta['chunks']}, pages: {doc_meta['documents']})"
        )
