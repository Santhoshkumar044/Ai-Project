import os
import re
import uuid
import time
import requests
from io import BytesIO
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


import tiktoken
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ============ Configuration ============ #
PINECONE_API_KEY = "pcsk_5eZ6Mn_9qEdhgRUVUGwaT2SyYntXWz7ZENoSsyfmRmuVNuo5bgsYAGuGv3qPsQUbbXumtQ"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "policy-check"
EMBED_DIM = 384
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# ============ Pinecone Setup ============ #
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print("⏳ Waiting for index to be ready...")
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(2)

index = pc.Index(INDEX_NAME)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============ Utilities ============ #
def clean_text(text: str) -> str:
    text = text.replace('\\', '').replace('*', '')
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, max_tokens: int = 200, overlap: int = 50, model_name: str = "gpt-3.5-turbo") -> List[str]:
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(enc.decode(chunk))
        start += max_tokens - overlap
    return chunks

def read_pdf_from_url_or_path(input_path: str) -> str:
    try:
        if input_path.startswith("http://") or input_path.startswith("https://"):
            response = requests.get(input_path)
            response.raise_for_status()
            pdf_data = BytesIO(response.content)
        else:
            pdf_data = open(input_path, 'rb')

        reader = PyPDF2.PdfReader(pdf_data)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {str(e)}")

def embed_text(text: str) -> List[float]:
    return model.encode(text).tolist()

def store_chunks(chunks: List[str]):
    index.delete(delete_all=True)  # Clear index before storing new data
    for chunk in chunks:
        store_chunk(chunk)

def store_chunk(chunk_text: str):
    chunk_id = str(uuid.uuid4())
    vector = embed_text(chunk_text)
    index.upsert([{
        "id": chunk_id,
        "values": vector,
        "metadata": {"text": chunk_text}
    }],namespace="policy")
    print(f"✅ Stored chunk '{chunk_id}'")

def search_similar_chunks(query_text: str, top_k: int = 1) -> str:
    vector = embed_text(query_text)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True,namespace="policy")
    matches = results.get("matches", [])
    if not matches:
        print("❌ No matches found.")
        return "No relevant answer found."
    top_match = matches[0]
    print(f"🔍 Best match (score={top_match['score']:.3f}):\n{top_match['metadata']['text']}")
    return top_match['metadata']['text']

# ============ Main Pipeline ============ #
def process_document_for_pinecone(input_path: str, questions: List[str]) -> List[str]:
    raw_text = read_pdf_from_url_or_path(input_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    return chunks

if __name__ =="__main__":

    # ✅ Initialize Pinecone and index
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        print("⏳ Waiting for index to be ready...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)

    # ✅ Get index client and model
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
