from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from controllers.file_parser import extract_text_from_file
from controllers.llama_chat import generate_response, trim_to_token_limit  # ✅ import the helper
import httpx
from controllers.pinecone_helpers import read_pdf_from_url_or_path

router = APIRouter()
uploaded_text = ""

class DocQueryRequest(BaseModel):
    documents: str
    questions: List[str] | None = None


@router.post("/hackrx")
async def process_document_from_url(request: DocQueryRequest):
    global uploaded_text
    url = request.documents
    questions = request.questions

    answers = []

    # Download the file
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            return {"error": "Failed to download document from URL."}
        file_bytes = response.content

    # Extract file name
    filename = url.split("?")[0].split("/")[-1] or "document.pdf"
    # Extract and clean text
    uploaded_text = extract_text_from_file(filename, file_bytes).strip()
    if not uploaded_text:
        return {"error": "Failed to extract text from document."}

    # ✅ If user asks a question, trim the document before adding to full prompt
    if questions:
        for question in questions:
            safe_text = trim_to_token_limit(uploaded_text, max_tokens=7000)  # Token-safe
            prompt = (
                "You are an insurance assistant. Read the document below and "
                "answer the user's question using only the document's content.\n\n"
                f"Document:\n{safe_text}\n\n"
                f"Question: {question}\nAnswer:"
            )
            try:
                result = generate_response(
                    uploaded_text=uploaded_text,
                    prompt_type="custom",
                    full_prompt=prompt
                )
                answers.append(result[0])
            except Exception as e:
                return {
                    "error": "LLM processing failed.",
                    "details": str(e)
                }


        return {
            "answers":answers
        }

    # ✅ Otherwise, just summarize the full (trimmed) doc
    result = generate_response(uploaded_text, prompt_type="summary")
    return {
        "summary": result,
        "pages": len(uploaded_text.split("\n")) // 40,
        "source_url": url
    }
