from fastapi import APIRouter, UploadFile, File
from model.chat_model import ChatRequest
from controllers.file_parser import extract_text_from_file
from controllers.llama_chat import generate_response
import io

router = APIRouter()
uploaded_text = ""  # simple in-memory storage


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_text
    file_bytes = await file.read()
    uploaded_text = extract_text_from_file(file.filename, file_bytes).strip()

    if not uploaded_text:
        return {"error": "Failed to extract text from document."}

    response = generate_response(uploaded_text, prompt_type="summary")

    return {
        "summary": response,
        "pages": len(uploaded_text.split("\n")) // 40
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    global uploaded_text

    if not uploaded_text:
        return {"error": "No document uploaded yet."}

    response = generate_response(uploaded_text, message=req.message)
    return {"response": response}
