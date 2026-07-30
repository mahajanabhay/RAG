import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

def ingest_file(file_path, doc_id):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return len(chunks)