import PyPDF2
import docx
import pandas as pd
import io

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception:
            continue
    return text

def extract_docx_text(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_csv_text(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string()

def extract_text_from_file(filename, file_bytes):
    if filename.endswith(".pdf"):
        return extract_pdf_text(file_bytes)
    elif filename.endswith(".docx"):
        return extract_docx_text(file_bytes)
    elif filename.endswith(".csv"):
        return extract_csv_text(file_bytes)
    else:
        return "Unsupported file format. Use PDF, DOCX, or CSV."
