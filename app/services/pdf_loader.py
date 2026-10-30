import fitz
from pathlib import Path

def load_pdf(pdf_path:Path)->list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"pdf not found:{pdf_path}")
    pages=[]
    with fitz.open(pdf_path)as pdf:
        for page_number,page in enumerate(pdf,start=1):
            text=page.get_text()
            if text:
                pages.append({"page":page_number,
                             "text":text})
        if not pages:
            raise ValueError("No text extracted")
        return pages       
                