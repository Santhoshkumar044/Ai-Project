from fastapi import APIRouter
from pydantic import BaseModel
from controllers.llama_chat import generate_response

router = APIRouter()

class QueryRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_only(request: QueryRequest):
    response = generate_response(uploaded_text="", message=request.message)
    return {"response": response}
