


from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

from controllers.pinecone_helpers import process_document_for_pinecone, read_pdf_from_url_or_path


class DocQueryRequest(BaseModel):
    url: str
    questions: List[str]


router = APIRouter()

@router.post("/hackrx/run")
async def process_doc_from_url(request:DocQueryRequest):
    url = request.url
    questions = request.questions
    print(url)
    chunks = process_document_for_pinecone(input_path=url,questions=questions)
    return chunks