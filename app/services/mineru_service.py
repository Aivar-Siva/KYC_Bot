"""Text extraction: pytesseract for images, PyMuPDF for PDFs."""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_text(img: np.ndarray | None = None, pdf_path: Path | None = None) -> tuple[str, list[str]]:
    """
    Extract text from image array or PDF path.
    Returns (text, warnings).
    """
    warnings: list[str] = []

    if img is not None:
        return _from_image(img, warnings)
    if pdf_path is not None:
        return _from_pdf(pdf_path, warnings)
    return "", ["mineru_low_confidence_text"]


def _from_image(img: np.ndarray, warnings: list[str]) -> tuple[str, list[str]]:
    try:
        import pytesseract
        # Tesseract with Hindi + English for Aadhaar/Voter ID support
        text = pytesseract.image_to_string(img, lang="eng+hin", config="--psm 3")
        if not text.strip():
            warnings.append("mineru_low_confidence_text")
        return text.strip(), warnings
    except Exception as exc:
        logger.warning("Tesseract extraction failed: %s", exc)
        warnings.append("mineru_low_confidence_text")
        return "", warnings


def _from_pdf(pdf_path: Path, warnings: list[str]) -> tuple[str, list[str]]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        # Process first page only (KYC = single identity page)
        page = doc[0]
        text = page.get_text("text")

        # If text layer is empty (scanned PDF), render and OCR
        if not text.strip():
            pix = page.get_pixmap(dpi=200)
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return _from_image(img, warnings)

        doc.close()
        if not text.strip():
            warnings.append("mineru_low_confidence_text")
        return text.strip(), warnings
    except Exception as exc:
        logger.warning("PDF text extraction failed: %s", exc)
        warnings.append("mineru_low_confidence_text")
        return "", warnings
