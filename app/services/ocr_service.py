"""
PaddleOCR Service for Student AI Assistant
Handles optical character recognition for:
1. Handwritten notes and whiteboard photos (.png, .jpg, .jpeg, .webp, .bmp)
2. Scanned and image-heavy PDF documents (page-by-page rendering + OCR)

Architecture:
- Native in-process PaddleOCR when installed in current Python environment.
- High-performance Python 3.12 subprocess bridge fallback if running on a Python 
  version without native PaddlePaddle wheels (e.g., Python 3.14+).
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Union, List, Dict, Any

# Supported image file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}

_ocr_engine = None


def _get_python312_executable() -> str:
    """Finds a compatible Python 3.12/3.11 executable on Windows for PaddleOCR."""
    possible_paths = [
        r"C:\Program Files\Python312\python.exe",
        r"C:\Program Files\Python311\python.exe",
        r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return "py -3.12"


def _get_native_ocr():
    """Initializes and caches the native PaddleOCR engine."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from paddleocr import PaddleOCR
        # use_angle_cls=True enables 180-degree flipped text orientation detection
        # lang='en' is fast and accurate for English handwritten/printed notes
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _ocr_engine
    except Exception as e:
        return None


# ==========================================
# 1. Image OCR Functions
# ==========================================

def ocr_image(image_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Runs PaddleOCR on a single image file.
    
    Returns:
        List of detected text lines:
        [
            {"text": "Extracted line text", "confidence": 0.98, "box": [[x1, y1], ...]},
            ...
        ]
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Check if native PaddleOCR is available in current process
    engine = _get_native_ocr()
    if engine is not None:
        try:
            results = engine.ocr(str(image_path), cls=True)
            lines = []
            if results and results[0]:
                for line in results[0]:
                    box = line[0]
                    text, conf = line[1]
                    lines.append({
                        "text": text.strip(),
                        "confidence": round(float(conf), 4),
                        "box": box,
                    })
            return lines
        except Exception as e:
            print(f"[PaddleOCR Native] Error: {e}")

    # Subprocess fallback for Python 3.12 compatibility
    return _ocr_image_via_subprocess(image_path)


def _ocr_image_via_subprocess(image_path: Path) -> List[Dict[str, Any]]:
    """Runs PaddleOCR via Python 3.12 subprocess bridge."""
    py_exe = _get_python312_executable()
    script = """
import sys, json
from paddleocr import PaddleOCR

img_path = sys.argv[1]
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
results = ocr.ocr(img_path, cls=True)
output = []
if results and results[0]:
    for line in results[0]:
        box = line[0]
        text, conf = line[1]
        output.append({
            "text": text.strip(),
            "confidence": round(float(conf), 4),
            "box": box
        })
print("__OCR_JSON_START__")
print(json.dumps(output))
print("__OCR_JSON_END__")
"""
    try:
        cmd = [py_exe, "-c", script, str(image_path)] if not py_exe.startswith("py") else ["py", "-3.12", "-c", script, str(image_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        stdout = res.stdout
        if "__OCR_JSON_START__" in stdout and "__OCR_JSON_END__" in stdout:
            json_part = stdout.split("__OCR_JSON_START__")[1].split("__OCR_JSON_END__")[0].strip()
            return json.loads(json_part)
        return []
    except Exception as e:
        print(f"[PaddleOCR Subprocess] Error running OCR on {image_path.name}: {e}")
        return []


# ==========================================
# 2. PDF Scanned & Handwritten OCR
# ==========================================

def ocr_pdf(pdf_path: Union[str, Path], dpi: int = 200) -> List[Dict[str, Any]]:
    """Renders each page of a PDF as a high-res image and runs PaddleOCR on it.
    Preserves page boundaries and line citations.
    
    Returns:
        List of page dicts:
        [
            {
                "page": 1,
                "text": "Full concatenated text of page 1...",
                "lines": [{"text": "...", "confidence": 0.95}, ...]
            },
            ...
        ]
    """
    import tempfile
    import pymupdf  # fitz

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    pages_data = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Try fast native text extraction first
            native_text = page.get_text().strip()

            # If the page already has rich selectable text (e.g. standard PDF), use it
            # Otherwise (scanned / handwritten note / image-only page), render to image and run PaddleOCR
            if len(native_text) > 100:
                pages_data.append({
                    "page": page_num,
                    "text": native_text,
                    "lines": [{"text": line, "confidence": 1.0} for line in native_text.splitlines() if line.strip()],
                    "source_type": "native_digital_text",
                })
                continue

            # 2. Render page to high-res image for OCR
            zoom = dpi / 72.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_file = tmp_path / f"page_{page_num}.png"
            pix.save(str(img_file))

            # Run OCR on the rendered page
            detected_lines = ocr_image(img_file)
            page_text = "\n".join(l["text"] for l in detected_lines)

            pages_data.append({
                "page": page_num,
                "text": page_text,
                "lines": detected_lines,
                "source_type": "paddle_ocr",
            })

    doc.close()
    return pages_data


# ==========================================
# 3. Unified Dispatcher Function
# ==========================================

def extract_text_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Universal loader that auto-routes PDFs or Images to the appropriate OCR pipeline.
    
    Returns:
        {
            "filename": "...",
            "file_type": "pdf" | "image",
            "full_text": "...",
            "pages": [...],
            "total_pages": int
        }
    """
    path = Path(file_path).resolve()
    ext = path.suffix.lower()

    if ext in PDF_EXTENSIONS:
        pages = ocr_pdf(path)
        full_text = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages if p["text"].strip())
        return {
            "filename": path.name,
            "file_type": "pdf",
            "full_text": full_text,
            "pages": pages,
            "total_pages": len(pages),
        }

    elif ext in IMAGE_EXTENSIONS:
        lines = ocr_image(path)
        full_text = "\n".join(l["text"] for l in lines)
        return {
            "filename": path.name,
            "file_type": "image",
            "full_text": full_text,
            "pages": [
                {
                    "page": 1,
                    "text": full_text,
                    "lines": lines,
                    "source_type": "paddle_ocr",
                }
            ],
            "total_pages": 1,
        }

    else:
        raise ValueError(f"Unsupported file format for OCR: {ext}. Supported: {IMAGE_EXTENSIONS | PDF_EXTENSIONS}")


if __name__ == "__main__":
    print("PaddleOCR Service Loaded Successfully.")
    print("Supported Image Extensions:", sorted(list(IMAGE_EXTENSIONS)))
    print("Supported Document Extensions:", sorted(list(PDF_EXTENSIONS)))
