import os
import re
import requests
import fitz  # PyMuPDF
import cohere
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BEARER_KEY = os.getenv("BEARER_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set in environment variables.")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables.")

co = cohere.Client(COHERE_API_KEY)
client_llm = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL_NAME = "llama3-8b-8192"

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

def extract_sentences_from_pdf_bytes(pdf_bytes):
    sentences = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            sents = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences.extend([s.strip() for s in sents if s.strip()])
    return sentences

def get_embeddings_batch(texts: List[str], input_type="search_document"):
    # Single API call to Cohere embed
    response = co.embed(
        texts=texts,
        model="embed-english-v2.0",
        input_type=input_type
    )
    return response.embeddings

def batch(iterable, n=16):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def query_llm(context: str, question: str) -> str:
    prompt = (
        "You are a policy expert. Given the context from an insurance document, "
        "answer the user's question in one or two concise sentences. "
        "Avoid phrases like 'According to the text' or 'The document states'. "
        "Avoid bullet points, markdown symbols, and excess wording. "
        "Answer should be directly usable in a chatbot or UI.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = client_llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


@app.post("/hackrx/run")
async def hackrx_run(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None)
):
    if BEARER_KEY:
        if not authorization or authorization != f"Bearer {BEARER_KEY}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Step 1: Download PDF
    try:
        resp = requests.get(request.documents, timeout=30)
        resp.raise_for_status()
        pdf_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch PDF: {str(e)}")

    # Step 2: Extract sentences
    try:
        sentences = extract_sentences_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")
    if not sentences:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    # Step 3: Concurrent embeddings of sentences
    sentence_embeddings = []
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = []
        for batch_sents in batch(sentences, 16):
            futures.append(executor.submit(get_embeddings_batch, batch_sents, "search_document"))
        for future in as_completed(futures):
            try:
                batch_embeds = future.result()
                sentence_embeddings.extend(batch_embeds)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Embedding API error (sentences): {str(e)}")

    answers = []
    # Step 4: For each question, embed & find top contexts + query LLM
    for question in request.questions:
        try:
            question_embedding = get_embeddings_batch([question], input_type="search_query")[0]
        except Exception as e:
            answers.append(f"Error embedding question: {str(e)}")
            continue

        # Compute similarities and get top 6 relevant sentences
        try:
            sims = [cosine_similarity(question_embedding, sent_emb) for sent_emb in sentence_embeddings]
            top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:6]
            context_chunks = [sentences[i] for i in top_indices]
            context = " ".join(context_chunks)
        except Exception as e:
            answers.append(f"Error during context similarity search: {str(e)}")
            continue

        # Query LLM
        try:
            answer = query_llm(context, question)
        except Exception as e:
            answer = f"Error generating answer from LLM: {str(e)}"
        answers.append(answer)

    return {"answers": answers}
