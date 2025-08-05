import os
import re
import textwrap
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize OpenAI-compatible client for HuggingFace
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

def clean_text(text):
    """Remove unwanted characters like *, \, excessive whitespace."""
    text = text.replace('\\', '')  # Remove backslashes
    text = text.replace('*', '')   # Remove asterisks
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def generate_response(uploaded_text: str, message: str = "", prompt_type="qa"):
    trimmed_text = uploaded_text[:12000] if uploaded_text else ""

    # Construct prompt
    if not trimmed_text:
        prompt = f"Answer the following question:\n\n{message}"
    elif prompt_type == "summary" or message.strip() == "":
        prompt = f"Summarize the following document:\n\n{trimmed_text}"
    else:
        prompt = f"Based on the following document, answer this:\n{message}\n\nDocument:\n{trimmed_text}"

    # Call the model
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
        messages=[{"role": "user", "content": prompt}]
    )

    # Get the raw response
    response_text = completion.choices[0].message.content.strip()

    # Try parsing as JSON if it's a dict (like {"summary": [...], "pages": ...})
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict) and "summary" in parsed:
            summary = parsed["summary"]
            if isinstance(summary, list):
                return [clean_text(item) for item in summary]
            elif isinstance(summary, str):
                summary = clean_text(summary)
                return textwrap.wrap(summary, width=1000)
    except Exception:
        pass  # Not JSON, continue with normal flow

    # Clean and chunk plain response
    clean = clean_text(response_text)
    return textwrap.wrap(clean, width=1000)
