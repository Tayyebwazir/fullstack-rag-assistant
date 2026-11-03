import re
from pathlib import Path
import fitz
import docx


def load_document(file_path: Path | str) -> list[dict]:
    """Loads and extracts text from PDF, DOCX, and DOC documents.
    Returns a list of dicts: [{"page": int, "text": str}]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".docx":
        return _load_docx(path)
    elif ext == ".doc":
        return _load_doc(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, DOC.")


def _load_pdf(pdf_path: Path) -> list[dict]:
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text and text.strip():
                pages.append({"page": page_number, "text": text.strip()})
    if not pages:
        raise ValueError("No readable text found in PDF document.")
    return pages


def _load_docx(docx_path: Path) -> list[dict]:
    doc = docx.Document(docx_path)
    paragraphs = []

    # Extract text from paragraphs
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            paragraphs.append(t)

    # Extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                paragraphs.append(" | ".join(row_text))

    if not paragraphs:
        raise ValueError("No readable text found in DOCX document.")

    # Group paragraphs into page-like chunks (~500 words or 3000 chars per page)
    pages = []
    current_page = []
    current_length = 0
    page_num = 1

    for para in paragraphs:
        current_page.append(para)
        current_length += len(para)
        if current_length >= 2500:
            pages.append({"page": page_num, "text": "\n\n".join(current_page)})
            page_num += 1
            current_page = []
            current_length = 0

    if current_page:
        pages.append({"page": page_num, "text": "\n\n".join(current_page)})

    return pages


def _load_doc(doc_path: Path) -> list[dict]:
    """Fallback text extractor for legacy .doc binary files."""
    try:
        # Try opening with PyMuPDF first (some versions support XPS/DOC)
        doc = fitz.open(doc_path)
        pages = []
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                pages.append({"page": page_number, "text": text.strip()})
        if pages:
            return pages
    except Exception:
        pass

    # Binary string extraction fallback for legacy Word 97-2003
    with open(doc_path, "rb") as f:
        content = f.read()

    # Extract printable ASCII / UTF-8 text strings of length >= 4
    raw_strings = re.findall(rb"[\x20-\x7E\t\r\n]{4,}", content)
    decoded = []
    for s in raw_strings:
        try:
            line = s.decode("utf-8", errors="ignore").strip()
            if len(line) > 10 and not line.startswith("<?xml") and not line.startswith("http"):
                decoded.append(line)
        except Exception:
            continue

    if not decoded:
        raise ValueError("Could not extract readable text from .doc file. Please convert to DOCX or PDF.")

    full_text = "\n\n".join(decoded)
    return [{"page": 1, "text": full_text}]
