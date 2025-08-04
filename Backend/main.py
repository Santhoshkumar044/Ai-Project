# from fastapi import FastAPI, UploadFile, File,Form
# from pydantic import BaseModel
# from openai import OpenAI
# import pandas as pd
# import PyPDF2
# import docx
# import io


# HF_TOKEN = "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"

# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=HF_TOKEN,
# )

# app = FastAPI()

# # Store uploaded text globally (temporary memory, not for prod)
# uploaded_text = ""

# class ChatRequest(BaseModel):
#     message: str = ""

# def extract_pdf_text(file_bytes):
#     reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
#     text = ""
#     for page in reader.pages:
#         try:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#         except Exception:
#             continue
#     return text

# def extract_docx_text(file_bytes):
#     doc = docx.Document(io.BytesIO(file_bytes))
#     return "\n".join([para.text for para in doc.paragraphs])

# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(...),
#     message: str = Form(default="")
# ):
#     global uploaded_text
#     file_content = await file.read()

#     if file.filename.endswith(".pdf"):
#         text = extract_pdf_text(file_content)
#     elif file.filename.endswith(".docx"):
#         text = extract_docx_text(file_content)
#     elif file.filename.endswith(".csv"):
#         df = pd.read_csv(io.BytesIO(file_content))
#         text = df.to_string()
#     else:
#         return {"error": "Unsupported file format. Use PDF, DOCX, or CSV."}

#     uploaded_text = text.strip()
#     trimmed_text = uploaded_text[:12000]

#     if not message.strip():
#         prompt = f"Summarize the following document:\n\n{trimmed_text}"
#     else:
#         prompt = f"Based on the following document, answer this:\n{message}\n\nDocument:\n{trimmed_text}"

#     completion = client.chat.completions.create(
#         model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return {
#         "result": completion.choices[0].message.content,
#         "pages": len(uploaded_text.split("\n")) // 40
#     }

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     global uploaded_text

#     if not uploaded_text:
#         return {"error": "No document uploaded yet."}

#     if not req.message.strip():
#         # If no message, return summary
#         trimmed_text = uploaded_text[:12000]
#         prompt = f"Summarize the following document:\n\n{trimmed_text}"
#     else:
#         trimmed_text = uploaded_text[:12000]
#         prompt = f"Based on the following document, answer this:\n{req.message}\n\nDocument:\n{trimmed_text}"

#     completion = client.chat.completions.create(
#         model="meta-llama/Meta-Llama-3-8B-Instruct:novita",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return {"response": completion.choices[0].message.content}


from fastapi import FastAPI
from routes import document_routes

app = FastAPI()

app.include_router(document_routes.router)
