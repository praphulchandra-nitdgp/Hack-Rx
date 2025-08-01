from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, fitz, requests, faiss, re, uuid, traceback
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Setup Directories ---
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Embedding Model ---
print("🔍 Loading sentence-transformer embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded.")

# --- Get API key from environment variables ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --- Initialize Groq/OpenAI Client ---
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
MODEL_NAME = "llama3-8b-8192"  # ✅ Supported Groq model

# --- In-memory stores ---
doc_store = {}
embedding_store = {}

# --- Fallback sentence splitter ---
def fallback_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# --- Request schema ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Query Groq LLM ---
def query_groq_llm(context: str, question: str) -> str:
    try:
        prompt = (
            "You are a policy expert. Given the context from an insurance document, "
            "answer the user's question in one or two concise sentences. "
            "Avoid phrases like 'According to the text' or 'The document states'. "
            "Avoid bullet points, markdown symbols, and excess wording. "
            "Answer should be directly usable in a chatbot or UI.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error getting response from LLM: {e}"

# --- Main endpoint ---
@app.post("/hackrx/run")
async def hackrx_run(request: QueryRequest):
    try:
        # Step 1: Download PDF
        pdf_url = request.documents
        pdf_bytes = requests.get(pdf_url).content
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        # Step 2: Extract and chunk PDF
        all_sentences = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    sentences = fallback_sent_tokenize(text)
                    all_sentences.extend(sentences)

        if not all_sentences:
            return {"error": "No extractable text found in PDF."}

        # Step 3: Embed and index
        doc_store[doc_id] = all_sentences
        embeddings = embedder.encode(all_sentences)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        embedding_store[doc_id] = {"index": index, "texts": all_sentences}

        # Step 4: Answer questions
        answers = []
        for question in request.questions:
            question_embedding = embedder.encode([question])[0]
            D, I = index.search(np.array([question_embedding]), k=6)
            selected_texts = [all_sentences[i] for i in I[0] if i < len(all_sentences)]

            context = "\n".join(selected_texts)
            while len(context.split()) > 1600:
                selected_texts = selected_texts[:-1]
                context = "\n".join(selected_texts)

            answer = query_groq_llm(context, question)
            answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        return {
            "error": "Failed to process request.",
            "details": str(e),
            "trace": traceback.format_exc()
        }