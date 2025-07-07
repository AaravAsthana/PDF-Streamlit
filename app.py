import os
import uuid
import tempfile

import streamlit as st
from ingestion import parse_pdf
from vectorstore import add_documents, query_documents
from llm import rewrite_query, ask_gemini_with_history

# ── Page Config & Title ─────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Bot")

# ── Dark‑Mode WhatsApp‑Style Bubble CSS ──────────────────────────────
st.markdown(
    """
    <style>
    .chat-container { padding: 1rem; }
    .bot {
      background-color: #2f3136;
      color: #e1e1e1;
      padding: 0.6rem;
      border-radius: 1rem 1rem 1rem 0;
      margin: 0.5rem 0;
      max-width: 70%;
    }
    .user {
      background-color: #3e4045;
      color: #e1e1e1;
      padding: 0.6rem;
      border-radius: 1rem 1rem 0 1rem;
      margin: 0.5rem 0;
      margin-left: auto;
      max-width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Session State Defaults ───────────────────────────────────────────
# Ensure every key we’ll use exists.
defaults = {
    "session_id": str(uuid.uuid4()),
    "history": [],
    "indexing": False,
    "indexed": False,
    "pdf_path": None,
    "input_counter": 0,          # uploader + input-reset counter
    "run_query": False,          # flag to trigger LLM fetch
    "current_question": "",      # holds the question to send
    "user_input": "",            # text_input binding
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar: Clear Session ───────────────────────────────────────────
if st.sidebar.button("🔄 Clear Session"):
    from vectorstore import collection

    # Delete the indexed vectors for this session
    collection.delete(where={"session_id": st.session_state.session_id})

    # Remove the temp PDF if on disk
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        try:
            os.remove(st.session_state.pdf_path)
        except OSError:
            pass

    # Reset all relevant session state
    st.session_state.history           = []
    st.session_state.pdf_path          = None
    st.session_state.indexed           = False
    st.session_state.run_query         = False
    st.session_state.current_question  = ""
    st.session_state.user_input        = ""
    # Bump the counter so the uploader is re-created empty
    st.session_state.input_counter += 1

    st.sidebar.success("Session cleared!")

# ── File Uploader & Indexing ────────────────────────────────────────
uploaded = st.file_uploader(
    "📥 Drag & drop a PDF here",
    type="pdf",
    key=f"uploader_{st.session_state.input_counter}"
)
if uploaded and not st.session_state.indexed:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tf.write(uploaded.read())
    tf.flush()
    st.session_state.pdf_path = tf.name
    st.session_state.indexing = True

if st.session_state.indexing and not st.session_state.indexed:
    with st.spinner("Indexing PDF…"):
        try:
            chunks = parse_pdf(st.session_state.pdf_path)
            add_documents(st.session_state.session_id, chunks)
            st.session_state.history.append({
                "role": "assistant",
                "content": "✅ PDF indexed! Ask me anything."
            })
            st.session_state.indexed = True
        except Exception:
            st.error("❌ Failed to parse/index PDF. Please try again.")
        finally:
            st.session_state.indexing = False

# ── Render Chat History ─────────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for turn in st.session_state.history:
    cls = "user" if turn["role"] == "user" else "bot"
    # user bubble vs bot bubble
    st.markdown(
        f'<div class="{cls}">{turn["content"]}</div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ── The Send Pipeline Callback ──────────────────────────────────────
def on_send():
    q = st.session_state.user_input.strip()
    if not q:
        return

    # 1️⃣ Move the user bubble
    st.session_state.history.append({"role": "user", "content": q})

    # 2️⃣ Clear the input box
    st.session_state.user_input = ""

    # 3️⃣ Show spinner + run the full pipeline
    placeholder = st.empty()
    with st.spinner("Fetching answer…"):
        placeholder.markdown("💬 Thinking…")

        # a) Rewrite the query
        terms = rewrite_query(q)

        # b) Retrieve & lightly filter
        docs = query_documents(st.session_state.session_id, top_k=15)
        paras = [d for d in docs if any(t.lower() in d.lower() for t in terms)]
        if not paras:
            paras = docs[:5]
        context = "\n\n".join(paras)

        # c) Ask Gemini
        try:
            answer = ask_gemini_with_history(
                st.session_state.history.copy(),
                context,
                q
            )
        except Exception:
            answer = "⚠️ Service busy. Please try again shortly."

    # 4️⃣ Append the assistant bubble & clear spinner
    st.session_state.history.append({"role": "assistant", "content": answer})
    placeholder.empty()

# ── Query Input & Send Button ────────────────────────────────────────
disable_input = st.session_state.indexing or not st.session_state.indexed

st.text_input(
    "Your question:",
    key="user_input",
    value=st.session_state.user_input,
    disabled=disable_input,
    placeholder="Type your question here…"
)
st.button("Send", disabled=disable_input, on_click=on_send)
