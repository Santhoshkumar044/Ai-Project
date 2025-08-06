import os
import re
import textwrap
import json
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken  # ✅ For token-safe trimming

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# HuggingFace-compatible OpenAI client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)
#To convert the document into ch
def chunk_text(text, max_tokens=300, overlap=50, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(enc.decode(chunk))
        start += max_tokens - overlap
    return chunks


# ✅ Clean messy model output
def clean_text(text):
    text = text.replace('\\', '')
    text = text.replace('*', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ✅ Token-safe text trimming for models with 8k limit
def trim_to_token_limit(text: str, max_tokens: int = 7000) -> str:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # works well with most models
    tokens = enc.encode(text)
    trimmed_tokens = tokens[:max_tokens]
    return enc.decode(trimmed_tokens)

# ✅ Main response function
def generate_response(uploaded_text: str, message: str = "", prompt_type="qa", full_prompt: str = None):
    trimmed_text = trim_to_token_limit(uploaded_text, max_tokens=7000) if uploaded_text else ""

    # ✅ Prioritize full custom prompt
    if full_prompt:
        prompt = full_prompt
    elif not trimmed_text:
        prompt = f"Answer the following question:\n\n{message}"
    elif prompt_type == "summary" or message.strip() == "":
        prompt = f"Summarize the following document:\n\n{trimmed_text}"
    else:
        prompt = (
            f"You are an insurance assistant. Based only on the document below, answer the question:\n\n"
            f"Question: {message}\n\n"
            f"Document:\n{trimmed_text}"
        )

    # 🔁 Model call
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        return [f"LLM processing failed. Error: {str(e)}"]

    response_text = completion.choices[0].message.content.strip()

    # ✅ Try JSON response parsing
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict) and "summary" in parsed:
            summary = parsed["summary"]
            if isinstance(summary, list):
                return [clean_text(item) for item in summary]
            elif isinstance(summary, str):
                return textwrap.wrap(clean_text(summary), width=1000)
    except Exception:
        pass

    # ✅ Fallback to plain text
    return textwrap.wrap(clean_text(response_text), width=1000)
