import sys
from pathlib import Path

# Lazy-import heavy deps so the script can still run without OCR installed
try:
    import pdfplumber
except Exception as e:  # pragma: no cover
    pdfplumber = None

def _extract_with_pdfplumber(pdf_path: str) -> str:
    """
    Try to extract selectable text from a PDF using pdfplumber.
    Returns a single string (may be empty if the PDF is image-scanned).
    """
    if pdfplumber is None:
        return ""
    try:
        pages_text: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text).strip()
    except Exception:
        return ""

def _maybe_extract_with_ocr(pdf_path: str) -> str:
    """
    Best-effort OCR fallback using pytesseract + pdf2image if installed locally.
    If OCR deps are missing, returns empty string.
    Note: OCR is slower and intended as a last resort for scanned PDFs.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""

    try:
        images = convert_from_path(pdf_path)
        ocr_text_parts: list[str] = []
        for img in images:
            ocr_text_parts.append(pytesseract.image_to_string(img))
        return "\n".join(ocr_text_parts).strip()
    except Exception:
        return ""

def pdf_to_txt(pdf_path: str, out_dir: str = "data") -> Path:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir_path / (Path(pdf_path).stem + ".txt")

    print(f"➡️  Processing: {pdf_path}")

    # 1) Try normal text extraction
    text = _extract_with_pdfplumber(pdf_path)

    # 2) If empty, attempt OCR fallback (optional)
    used_ocr = False
    if not text:
        print("   • No selectable text found. Attempting OCR fallback (if available)...")
        text = _maybe_extract_with_ocr(pdf_path)
        used_ocr = bool(text)

    # 3) Persist result (even if empty, so the user sees a file got created)
    if not text:
        text = ""  # create an empty file as a signal; we still write it
    out_txt_path.write_text(text)

    if used_ocr:
        print(f"✅ Wrote (OCR): {out_txt_path}")
    else:
        # If empty, inform the user about next steps
        if text:
            print(f"✅ Wrote: {out_txt_path}")
        else:
            print(f"⚠️  Created empty file (no text extracted): {out_txt_path}")
            print("   Try one of these: \n"
                  "   - Open the PDF and copy/paste a few paragraphs into a .txt under data/\n"
                  "   - Install OCR deps: pip install pdf2image pytesseract poppler-utils\n"
                  "   - Re-run this script to regenerate the .txt")

    return out_txt_path

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/pdf_to_txt.py <PDF_PATH> [<PDF_PATH> ...]")
        return 1
    for p in argv[1:]:
        pdf_to_txt(p)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))