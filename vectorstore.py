import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# In‑memory client; for persistence, pass persist_directory=
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_function
)

def add_documents(session_id: str, chunks: list):
    collection.delete(where={"session_id": session_id})
    texts, metas, ids = [], [], []
    for i, c in enumerate(chunks):
        texts.append(c["page_content"])
        meta = {k: str(v) for k, v in c["metadata"].items() if v is not None}
        meta["session_id"] = session_id
        metas.append(meta)
        ids.append(f"{session_id}_{i}")
    collection.add(documents=texts, metadatas=metas, ids=ids)

def query_documents(session_id: str, top_k: int = 10):
    res = collection.query(
        query_texts=[""],  # overwritten by llm flow
        n_results=top_k,
        where={"session_id": session_id},
        include=["documents"]
    )
    return res["documents"][0]

def summarize_session(session_id: str, model_fn) -> str:
    res = collection.get(where={"session_id": session_id})
    docs = sum(res.get("documents", []), [])
    combined = "\n\n".join(docs)
    return model_fn(combined)
