"""Image preprocessing: contrast enhancement + compress for Groq API limit."""
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _compress_for_groq(img: np.ndarray) -> bytes:
    """Resize to max 2000px longest side, JPEG q85, target ≤3.5MB."""
    h, w = img.shape[:2]
    max_px = settings.groq_image_max_px
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    quality = settings.groq_image_quality
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)

    while buf.tell() > settings.groq_image_max_bytes and quality > 50:
        quality -= 10
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)

    return buf.getvalue()


def preprocess(image_path: Path) -> tuple[np.ndarray, bytes, list[str]]:
    """
    Returns:
        full_img: contrast-enhanced ndarray for MinerU
        groq_bytes: compressed JPEG bytes (≤3.5MB) for Groq API
        warnings: list of warning strings
    """
    warnings: list[str] = []
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Cannot read image")
        img = _enhance_contrast(img)
        groq_bytes = _compress_for_groq(img)
        return img, groq_bytes, warnings
    except Exception as exc:
        logger.warning("Preprocessing failed: %s", exc)
        warnings.append("preprocessing_failed")
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError("Cannot read image file") from exc
        return img, _compress_for_groq(img), warnings
