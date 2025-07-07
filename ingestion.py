from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configuration import LLAMA_CLOUD_API_KEY

def parse_pdf(path: str):
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        num_workers=4,
        verbose=False,
        language="en"
    )
    result = parser.parse(path)
    docs = result.get_text_documents(split_by_page=True)

    # Paragraph split + merge small ones
    chunks = []
    for doc in docs:
        paras = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        merged, buf = [], ""
        for p in paras:
            if buf and len(buf) < 200:
                buf += "\n\n" + p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        for part in merged:
            chunks.append({
                "page_content": part,
                "metadata": {"page": doc.metadata.get("page")}
            })
    return chunks
