"""Validation, masking, and confidence calibration."""
import re
import logging
from datetime import date, datetime
from typing import Any

from dateutil import parser as dateutil_parser

from app.config import REQUIRED_FIELDS, SUPPORTED_TYPES, settings

logger = logging.getLogger(__name__)

_AADHAAR_RE = re.compile(r"[\s-]?".join([r"(\d{4})"] * 3))
_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


# ---------------------------------------------------------------------------
# Aadhaar masking
# ---------------------------------------------------------------------------

def mask_aadhaar(number: str | None) -> str | None:
    """Return XXXX XXXX NNNN form. Input may be spaced or unspaced."""
    if not number:
        return number
    digits = re.sub(r"[\s-]", "", number)
    if len(digits) != 12 or not digits.isdigit():
        return number
    return f"XXXX XXXX {digits[-4:]}"


# ---------------------------------------------------------------------------
# PAN validation
# ---------------------------------------------------------------------------

def validate_pan(pan: str | None) -> bool:
    if not pan:
        return False
    return bool(_PAN_RE.match(pan.strip()))


# ---------------------------------------------------------------------------
# DL expiry validation
# ---------------------------------------------------------------------------

def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return dateutil_parser.parse(str(value), dayfirst=True).date()
    except Exception:
        return None


def validate_dl_expiry(validity_end_date: str | None) -> bool:
    """Returns True if expiry is in the future."""
    d = _parse_date(validity_end_date)
    if d is None:
        return False
    return d >= date.today()


# ---------------------------------------------------------------------------
# Passport MRZ TD3 validation
# ---------------------------------------------------------------------------

_MRZ_WEIGHTS = [7, 3, 1]
_MRZ_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"


def _mrz_char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c.isalpha():
        return ord(c.upper()) - ord("A") + 10
    return 0  # '<'


def _mrz_checksum(s: str) -> int:
    total = sum(_mrz_char_value(c) * _MRZ_WEIGHTS[i % 3] for i, c in enumerate(s))
    return total % 10


def validate_mrz(mrz_line: str | None, fields: dict) -> tuple[bool, list[str]]:
    """
    Validate TD3 MRZ checksums and compare against biographical fields.
    Returns (mrz_valid, warnings).
    mrz_line should be 88 chars (two 44-char lines concatenated).
    """
    warnings: list[str] = []
    if not mrz_line or len(mrz_line) < 88:
        # Try passporteye as fallback parser
        return False, ["mrz_mismatch_with_biographical_data"]

    line1 = mrz_line[:44]
    line2 = mrz_line[44:88]

    try:
        passport_num = line2[0:9]
        passport_check = int(line2[9])
        dob = line2[13:19]
        dob_check = int(line2[19])
        expiry = line2[21:27]
        expiry_check = int(line2[27])
        composite = line2[0:43]  # TD3: composite checksum covers entire line2 except final check digit
        composite_check = int(line2[43])
        checks = [
            _mrz_checksum(passport_num) == passport_check,
            _mrz_checksum(dob) == dob_check,
            _mrz_checksum(expiry) == expiry_check,
            _mrz_checksum(composite) == composite_check,
        ]

        if not all(checks):
            warnings.append("mrz_mismatch_with_biographical_data")
            return False, warnings

        # Cross-check against biographical fields
        bio_passport = re.sub(r"[\s-]", "", str(fields.get("passport_number", "") or "")).upper()
        mrz_passport = passport_num.rstrip("<")
        if bio_passport and mrz_passport and bio_passport != mrz_passport:
            warnings.append("mrz_mismatch_with_biographical_data")
            return False, warnings

        return True, warnings

    except (ValueError, IndexError) as exc:
        logger.warning("MRZ parse error: %s", exc)
        warnings.append("mrz_mismatch_with_biographical_data")
        return False, warnings


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def calibrate_confidence(
    raw_confidence: float,
    doc_type: str,
    fields: dict,
    field_confidence: dict,
    warnings: list[str],
    validation_passed: dict[str, bool],
) -> float:
    score = raw_confidence

    if doc_type not in SUPPORTED_TYPES:
        return 0.0

    required = REQUIRED_FIELDS.get(doc_type, [])

    # Positive: all required fields present
    null_count = sum(1 for f in required if not fields.get(f))
    if null_count == 0:
        score += 0.05
    else:
        score -= 0.05 * null_count

    # Positive: validation checks pass
    for passed in validation_passed.values():
        score += 0.03 if passed else -0.05

    # Negative: warnings
    score -= 0.03 * len(warnings)

    # Positive: high field confidence
    fc_values = [v for v in field_confidence.values() if isinstance(v, (int, float))]
    if fc_values and all(v >= 0.85 for v in fc_values):
        score += 0.03

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Main validate function
# ---------------------------------------------------------------------------

def validate(llm_output: dict, extra_warnings: list[str]) -> dict:
    """
    Apply all validation rules to LLM output.
    Returns the final validated response dict (ready for API output).
    Raises ValueError('unsupported_document_type') for unsupported docs.
    """
    doc_type = str(llm_output.get("document_type", "unsupported")).lower().strip()
    raw_confidence = float(llm_output.get("confidence", 0.0))
    fields: dict = dict(llm_output.get("fields") or {})
    field_confidence: dict = dict(llm_output.get("field_confidence") or {})
    warnings: list[str] = list(llm_output.get("extraction_warnings") or []) + extra_warnings
    mrz_line: str | None = llm_output.get("mrz_line")

    validation_passed: dict[str, bool] = {}
    mrz_valid: bool | None = None
    masked = False

    # --- Unsupported rejection (pre-calibration) ---
    if doc_type not in SUPPORTED_TYPES or raw_confidence < settings.classification_threshold:
        raise ValueError("unsupported_document_type")

    # --- Aadhaar masking (must happen before output construction) ---
    if doc_type == "aadhaar":
        raw_num = fields.get("aadhaar_number")
        fields["aadhaar_number"] = mask_aadhaar(raw_num)
        masked = True

    # --- PAN validation ---
    if doc_type == "pan":
        pan_ok = validate_pan(fields.get("pan_number"))
        validation_passed["pan"] = pan_ok
        if not pan_ok:
            warnings.append("invalid_pan_format")

    # --- DL expiry ---
    if doc_type == "driving_licence":
        dl_ok = validate_dl_expiry(fields.get("validity_end_date"))
        validation_passed["dl_expiry"] = dl_ok
        if not dl_ok:
            warnings.append("driving_licence_expired")

    # --- Passport MRZ ---
    if doc_type == "passport":
        mrz_valid, mrz_warnings = validate_mrz(mrz_line, fields)
        warnings.extend(mrz_warnings)
        validation_passed["mrz"] = mrz_valid

    # --- Missing required fields ---
    for f in REQUIRED_FIELDS.get(doc_type, []):
        if not fields.get(f):
            if "missing_required_field" not in warnings:
                warnings.append("missing_required_field")

    # --- Confidence calibration ---
    calibrated = calibrate_confidence(
        raw_confidence, doc_type, fields, field_confidence, warnings, validation_passed
    )

    # --- Post-calibration unsupported check ---
    if calibrated < settings.classification_threshold:
        raise ValueError("unsupported_document_type")

    # --- Build response ---
    response: dict[str, Any] = {
        "document_type": doc_type,
        "confidence": round(calibrated, 4),
        "fields": fields,
        "field_confidence": field_confidence,
        "masked": masked,
        "extraction_warnings": list(dict.fromkeys(warnings)),  # deduplicate, preserve order
    }
    if doc_type == "passport":
        response["mrz_valid"] = mrz_valid

    return response
