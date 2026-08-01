from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ingest import ingest_file, collection, embed
import shutil, os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = ingest_file(path, file.filename)
    return {"chunks_added": chunks}

@app.post("/chat")
async def chat(query: str):
    q_embedding = embed([query])
    results = collection.query(query_embeddings=q_embedding, n_results=3)
    context = "\n".join(results["documents"][0])
    prompt = f"Answer using only this context:\n{context}\n\nQuestion: {query}"
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response.choices[0].message.content}