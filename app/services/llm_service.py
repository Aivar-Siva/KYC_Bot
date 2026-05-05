"""LLM extraction: Groq primary (vision + JSON mode) with Bedrock text-only fallback."""
import base64
import json
import logging
import re
import time

import httpx
from groq import Groq

from app.config import settings, REQUIRED_FIELDS, SUPPORTED_TYPES

logger = logging.getLogger(__name__)

_groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

_SYSTEM_PROMPT = f"""You are a KYC document extraction system for Indian identity documents.

Supported document types: {", ".join(SUPPORTED_TYPES)}

Field schemas:
- aadhaar: name, dob, gender, address, aadhaar_number
- pan: name, fathers_name, dob, pan_number
- voter_id: name, fathers_or_husband_name, dob, voter_id_number, address, constituency
- driving_licence: name, dob, dl_number, validity_start_date, validity_end_date, vehicle_classes (list), address
- passport: name, dob, passport_number, nationality, expiry_date, mrz_line (both MRZ lines concatenated)

Rules:
1. Return ONLY valid JSON, no preamble or explanation.
2. Set document_type to "unsupported" if the document is not one of the 5 types above.
3. Use null for any field that is missing, unreadable, or uncertain — never invent values.
4. Normalize all dates to YYYY-MM-DD format.
5. Set confidence (0.0–1.0) for the overall classification.
6. Set field_confidence (0.0–1.0) for each extracted field.
7. Add warning names to extraction_warnings for low-confidence or occluded fields.
8. Use the image to recover fields where the extracted text is incomplete, including Hindi and regional language text.
9. For passport documents, extract both MRZ lines into mrz_line (concatenated, no separator).

Valid warning names: low_scan_quality_on_address_field, dob_partially_occluded, missing_required_field, classification_low_confidence

Output schema:
{{
  "document_type": "<type or unsupported>",
  "confidence": <float>,
  "fields": {{}},
  "field_confidence": {{}},
  "extraction_warnings": [],
  "mrz_line": "<passport MRZ only, null otherwise>"
}}"""


def _build_user_message(mineru_text: str, groq_image_bytes: bytes) -> list[dict]:
    b64 = base64.b64encode(groq_image_bytes).decode()
    return [
        {
            "type": "text",
            "text": f"Extracted text from document (may be incomplete):\n\n{mineru_text or '(no text extracted)'}\n\nExtract all fields from the document image.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]


def _call_groq(mineru_text: str, groq_image_bytes: bytes) -> dict:
    if _groq_client is None:
        raise RuntimeError("GROQ_API_KEY not set")

    for attempt in range(2):
        try:
            resp = _groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(mineru_text, groq_image_bytes)},
                ],
                response_format={"type": "json_object"},
                max_tokens=settings.groq_max_tokens,
                timeout=settings.groq_timeout,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            logger.warning("Groq attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)
            else:
                raise


def _call_bedrock(mineru_text: str) -> dict:
    """Text-only fallback via Bedrock proxy."""
    user_msg = (
        f"You are a KYC document extraction system.\n\n"
        f"{_SYSTEM_PROMPT}\n\n"
        f"Extracted text from document:\n\n{mineru_text or '(no text extracted)'}\n\n"
        f"Extract all fields and return JSON only."
    )
    payload = {
        "model_id": settings.bedrock_fallback_model,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": settings.groq_max_tokens,
    }
    resp = httpx.post(settings.bedrock_proxy_url, json=payload, timeout=60.0)
    resp.raise_for_status()
    raw = resp.json()

    # Extract text from response
    content = ""
    if isinstance(raw, dict):
        content = (
            raw.get("content", "")
            or raw.get("text", "")
            or raw.get("generation", "")
            or str(raw)
        )
    elif isinstance(raw, str):
        content = raw

    # Try direct JSON parse, then regex extraction
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError("Bedrock response contained no valid JSON")


def extract(mineru_text: str, groq_image_bytes: bytes) -> tuple[dict, list[str]]:
    """
    Call Groq primary; fall back to Bedrock on failure.
    Returns (llm_output_dict, extra_warnings).
    """
    extra_warnings: list[str] = []

    try:
        result = _call_groq(mineru_text, groq_image_bytes)
        return result, extra_warnings
    except Exception as groq_exc:
        logger.warning("Groq failed, switching to Bedrock fallback: %s", groq_exc)
        extra_warnings.append("vision_unavailable")

    try:
        result = _call_bedrock(mineru_text)
        return result, extra_warnings
    except Exception as bedrock_exc:
        logger.error("Bedrock fallback also failed: %s", bedrock_exc)
        raise RuntimeError("processing_error") from bedrock_exc
