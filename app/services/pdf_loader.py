from pathlib import Path
from app.services.document_loader import load_document, _load_pdf

def load_pdf(pdf_path: Path) -> list[dict]:
    return _load_pdf(pdf_path)
