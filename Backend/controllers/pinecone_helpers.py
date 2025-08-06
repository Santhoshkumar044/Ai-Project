import os
import re
import uuid
import time
import requests
from io import BytesIO
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("PINECONE_API_KEY")

# ============ Configuration ============ #
PINECONE_API_KEY = HF_TOKEN
PINECONE_ENV = "us-east-1"
INDEX_NAME = "policy-check"
EMBED_DIM = 384
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Global placeholders
index = None
model = None

def setup():
    global index, model
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)

    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ============ Utilities ============ #
def clean_text(text: str) -> str:
    text = text.replace('\\', '').replace('*', '')
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


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

def store_chunks(chunks: List[str], batch_size: int = 100):
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {"text": chunk}
        })

        # Upsert in batches
        if len(vectors) == batch_size or i == len(chunks) - 1:
            index.upsert(vectors, namespace="policy")
            vectors = []
def clear_namespace():
    index.delete(delete_all=True, namespace="policy")

def store_chunk(chunk_text: str):
    chunk_id = str(uuid.uuid4())
    vector = embed_text(chunk_text)
    index.upsert([{
        "id": chunk_id,
        "values": vector,
        "metadata": {"text": chunk_text}
    }],namespace="policy")

def search_similar_chunks(query_text: str, top_k: int = 1) -> str:
    vector = embed_text(query_text)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True,namespace="policy")
    matches = results.get("matches", [])
    if not matches:
        return "No relevant answer found."
    top_match = matches[0]
    return top_match['metadata']['text']

# ============ Main Pipeline ============ #
def process_document_for_pinecone(input_path: str, questions: List[str]) -> List[str]:
        setup()
        raw_text = read_pdf_from_url_or_path(input_path)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        clear_namespace()
        store_chunks(chunks=chunks)
        answers = []
        for question in questions:
            answer = search_similar_chunks(question)
            answers.append(answer)

        return answers

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
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)

    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
