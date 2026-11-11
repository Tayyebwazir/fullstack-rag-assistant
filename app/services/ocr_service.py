
import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
OCR_PYTHON = os.getenv("OCR_PYTHON", r"E:\My_works\RAG_Weaviate\.venv-ocr\Scripts\python.exe")
_ocr = None

def _worker(paths):
    script = """
import sys,json
from paddleocr import PaddleOCR
ocr=PaddleOCR(ocr_version="PP-OCRv5",lang="en",use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=True)
output=[]
for path in json.loads(sys.argv[1]):
    lines=[]
    for result in ocr.predict(path):
        data=result.json
        if callable(data): data=data()
        data=data.get("res",data)
        texts=data.get("rec_texts",[])
        scores=data.get("rec_scores",[])
        boxes=data.get("rec_polys") or data.get("dt_polys") or []
        for i,text in enumerate(texts):
            box=boxes[i] if i<len(boxes) else []
            if hasattr(box,"tolist"): box=box.tolist()
            lines.append({"text":str(text),"confidence":float(scores[i]) if i<len(scores) else None,"box":box})
    output.append(lines)
print("OCR_RESULT="+json.dumps(output))
"""
    result = subprocess.run(
        [OCR_PYTHON, "-c", script, json.dumps([str(p) for p in paths])],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(450, len(paths)*60)
    )
    if result.returncode:
        raise RuntimeError(f"PaddleOCR failed: {result.stderr[-2000:]}")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("OCR_RESULT="):
            return json.loads(line[len("OCR_RESULT="):])
    raise RuntimeError("PaddleOCR returned no valid result")

def ocr_image(image_path):
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image format")
    from PIL import Image, ImageOps
    with tempfile.TemporaryDirectory() as tmp:
        normalized = Path(tmp) / "image.png"
        with Image.open(path) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(normalized)
        return _worker([normalized])[0]

def ocr_pdf(pdf_path, dpi=200):
    import pymupdf
    pages = []
    with tempfile.TemporaryDirectory() as tmp:
        with pymupdf.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if len(text) >= 100:
                    pages.append({
                        "page": i+1, "text": text,
                        "lines": [{"text": t, "confidence": None, "box": []} for t in text.splitlines() if t.strip()],
                        "source_type": "native_digital_text"
                    })
                else:
                    image_path = Path(tmp) / f"page_{i+1}.png"
                    page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72), alpha=False).save(str(image_path))
                    pages.append({"page": i+1, "image_path": image_path})
        pending = [p for p in pages if "image_path" in p]
        for start in range(0, len(pending), 8):
            batch = pending[start:start+8]
            results = _worker([p["image_path"] for p in batch])
            for p, lines in zip(batch, results):
                p["text"] = "\n".join(line["text"] for line in lines)
                p["lines"] = lines
                p["source_type"] = "paddle_ocr_v5"
                del p["image_path"]
    return pages



def extract_text_from_file(file_path):
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pdf":
        pages = ocr_pdf(path)
        file_type = "pdf"
    elif path.suffix.lower() in IMAGE_EXTENSIONS:
        lines = ocr_image(path)
        text = "\n".join(line["text"] for line in lines)
        pages = [{"page": 1, "text": text, "lines": lines, "source_type": "paddle_ocr_v5"}]
        file_type = "image"
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return {
        "filename": path.name,
        "file_type": file_type,
        "full_text": "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages if p["text"].strip()),
        "pages": pages,
        "total_pages": len(pages)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = extract_text_from_file(sys.argv[1])
        print(result["full_text"] or "No text detected")
    else:
        print("Usage: python ocr_service.py <path_to_image_or_pdf>")
