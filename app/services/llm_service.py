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


_DOC_SIGNATURES = {
    "aadhaar": (
        "Aadhaar card. Key markers: 'Unique Identification Authority of India' or 'UIDAI' or 'आधार', "
        "a 12-digit Aadhaar number (printed as XXXX XXXX XXXX), name, DOB, gender, address. "
        "May include Hindi text. The 12-digit number is the strongest signal."
    ),
    "pan": (
        "PAN card. Key markers: 'Income Tax Department' or 'Permanent Account Number', "
        "a 10-character PAN number in format AAAAA9999A (5 letters, 4 digits, 1 letter), "
        "name, father's name, DOB. Issued by Government of India."
    ),
    "passport": (
        "Indian Passport. Key markers: 'Republic of India', 'Passport' or 'पासपोर्ट', "
        "a passport number in format A9999999 (1 letter + 7 digits), "
        "MRZ lines at the bottom (two lines of 44 characters with '<' separators, starting with P<IND), "
        "nationality 'INDIAN', place of birth, date of issue/expiry. "
        "The MRZ lines and passport number format are the strongest signals — look for them even if the image is rotated."
    ),
    "voter_id": (
        "Voter ID / EPIC card. Key markers: 'Election Commission of India', 'EPIC No' or 'Voter ID', "
        "an EPIC number (alphanumeric, typically 3 letters + 7 digits), "
        "name, father's/husband's name, constituency, assembly segment. "
        "Layout and language vary by state."
    ),
    "driving_licence": (
        "Driving Licence. Key markers: 'Driving Licence' or 'DL No', state RTO name, "
        "a DL number in format SS-RR-YYYY-NNNNNNN (state code + RTO + year + number), "
        "vehicle classes (LMV, MCWG, etc.), validity dates, blood group. "
        "Issued by state Transport Department."
    ),
}


def gate_document_type(expected_type: str, groq_image_bytes: bytes) -> tuple[bool, str]:
    """
    Layer 1: Content-based document type gate.
    Uses textual markers and content signatures, not visual layout, so rotation doesn't matter.
    Returns (is_correct_type, reason).
    """
    if _groq_client is None:
        return True, "Groq unavailable, gate skipped"
    try:
        signature = _DOC_SIGNATURES.get(expected_type, expected_type)
        resp = _groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a KYC document classifier for Indian identity documents. "
                        "Identify documents by their TEXT CONTENT and printed labels — not by orientation or layout. "
                        "A rotated or upside-down document still contains the same text. "
                        "Return only valid JSON: {\"is_correct_type\": true/false, \"reason\": \"one sentence citing specific text you found or did not find\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"I need to verify this is a {expected_type}.\n\n"
                                f"A genuine {expected_type} contains these specific markers:\n{signature}\n\n"
                                f"Look at the image carefully — read all visible text regardless of orientation. "
                                f"Do you find these markers in this document? "
                                f"Cite the specific text you found or could not find in your reason."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(groq_image_bytes).decode()}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            timeout=20.0,
        )
        result = json.loads(resp.choices[0].message.content)
        return bool(result.get("is_correct_type", False)), str(result.get("reason", ""))
    except Exception as exc:
        logger.warning("Layer 1 gate failed (non-blocking): %s", exc)
        return True, "gate check unavailable"


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


_VERIFY_PROMPT = """You are a strict KYC fraud detection officer reviewing Indian identity documents.

You will receive:
1. A document image
2. The REQUIRED document type (what the user is supposed to upload)
3. Fields extracted from the image

Your task: determine if this submission is valid. Apply these checks IN ORDER and stop at the first failure.

CHECK 1 — CORRECT DOCUMENT TYPE (HARD FAIL):
Is the document in the image actually a {expected_type}?
- Look at the physical layout, header text, logos, and format of the document
- A PAN card is NOT an Aadhaar card. A Voter ID is NOT a Passport. Be precise.
- If the document in the image is NOT a {expected_type}, set overall_pass=false immediately.
- Do not be fooled by similar-looking documents. Each Indian ID has distinct visual markers.

CHECK 2 — FIELD ACCURACY (HARD FAIL):
Do the extracted field values exactly match what is printed/visible on the document?
- Compare each field value against what you can read in the image
- If a field value is not visible on the document or contradicts what is shown, fail this check

CHECK 3 — DATA INTEGRITY:
Does the data make sense as a real person's document?
- Impossible DOB (future date, age > 120): fail
- ID number format doesn't match what's visible: fail

Return ONLY this JSON with no explanation outside it:
{{
  "authentic": true or false,
  "correct_document_type": true or false,
  "fields_match": true or false,
  "data_integrity": true or false,
  "overall_pass": true or false,
  "reasoning": "one sentence — be specific about what failed",
  "failed_checks": ["specific issue 1", "specific issue 2"]
}}

overall_pass = true ONLY if ALL checks pass. Be strict. A PAN card submitted for Aadhaar verification MUST fail."""


def verify_document(fields: dict, doc_type: str, groq_image_bytes: bytes, expected_type: str | None = None) -> tuple[bool, str, list[str]]:
    """
    Fraud reasoning pass. expected_type is what the user is supposed to be submitting.
    Returns (passed, reasoning, failed_checks).
    """
    if _groq_client is None:
        return True, "", []
    try:
        required_type = expected_type or doc_type
        prompt = _VERIFY_PROMPT.replace("{expected_type}", required_type)
        fields_text = "\n".join(f"  {k}: {v}" for k, v in fields.items())
        user_content = [
            {
                "type": "text",
                "text": f"Required document type: {required_type}\nExtracted fields:\n{fields_text}\n\nAnalyse the document image and return your fraud assessment JSON."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(groq_image_bytes).decode()}"}
            },
        ]
        resp = _groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
            timeout=25.0,
        )
        result = json.loads(resp.choices[0].message.content)
        passed = bool(result.get("overall_pass", True))
        reasoning = str(result.get("reasoning", ""))
        failed_checks = list(result.get("failed_checks", []))
        return passed, reasoning, failed_checks
    except Exception as exc:
        logger.warning("Document verification failed (non-blocking): %s", exc)
        return True, "", []
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
