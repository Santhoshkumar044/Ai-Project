import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load .env file

HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

def generate_response(uploaded_text: str, message: str = "", prompt_type="qa"):
    trimmed_text = uploaded_text[:12000] if uploaded_text else ""

    if not trimmed_text:  # No document provided
        prompt = f"Answer the following question:\n\n{message}"
    elif prompt_type == "summary" or message.strip() == "":
        prompt = f"Summarize the following document:\n\n{trimmed_text}"
    else:
        prompt = f"Based on the following document, answer this:\n{message}\n\nDocument:\n{trimmed_text}"

    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content
