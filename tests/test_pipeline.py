"""Unit tests for validators, masking, and confidence calibration."""
import pytest
from app.services.validation_service import (
    mask_aadhaar,
    validate_pan,
    validate_dl_expiry,
    validate_mrz,
    calibrate_confidence,
    validate,
)


# ---------------------------------------------------------------------------
# Aadhaar masking
# ---------------------------------------------------------------------------

def test_mask_aadhaar_unspaced():
    assert mask_aadhaar("123456789012") == "XXXX XXXX 9012"

def test_mask_aadhaar_spaced():
    assert mask_aadhaar("1234 5678 9012") == "XXXX XXXX 9012"

def test_mask_aadhaar_already_masked():
    # Already masked input should pass through unchanged (not 12 digits)
    result = mask_aadhaar("XXXX XXXX 9012")
    assert result == "XXXX XXXX 9012"

def test_mask_aadhaar_none():
    assert mask_aadhaar(None) is None

def test_mask_aadhaar_never_exposes_first_8():
    result = mask_aadhaar("123456789012")
    assert "1234" not in result
    assert "5678" not in result
    assert "9012" in result


# ---------------------------------------------------------------------------
# PAN validation
# ---------------------------------------------------------------------------

def test_pan_valid():
    assert validate_pan("ABCDE1234F") is True

def test_pan_invalid_lowercase():
    assert validate_pan("abcde1234f") is False

def test_pan_invalid_short():
    assert validate_pan("ABCD1234F") is False

def test_pan_invalid_format():
    assert validate_pan("12345ABCDE") is False

def test_pan_none():
    assert validate_pan(None) is False


# ---------------------------------------------------------------------------
# DL expiry
# ---------------------------------------------------------------------------

def test_dl_expiry_future():
    assert validate_dl_expiry("2099-12-31") is True

def test_dl_expiry_past():
    assert validate_dl_expiry("2000-01-01") is False

def test_dl_expiry_none():
    assert validate_dl_expiry(None) is False

def test_dl_expiry_indian_format():
    assert validate_dl_expiry("31/12/2099") is True


# ---------------------------------------------------------------------------
# MRZ validation
# ---------------------------------------------------------------------------

def _make_mrz_checksum(s: str) -> int:
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(s):
        if c.isdigit():
            v = int(c)
        elif c.isalpha():
            v = ord(c.upper()) - ord('A') + 10
        else:
            v = 0
        total += v * weights[i % 3]
    return total % 10

def test_mrz_none_returns_invalid():
    valid, warnings = validate_mrz(None, {})
    assert valid is False
    assert "mrz_mismatch_with_biographical_data" in warnings

def test_mrz_too_short_returns_invalid():
    valid, warnings = validate_mrz("SHORT", {})
    assert valid is False

def test_mrz_valid_checksums():
    # TD3 line2 layout (44 chars):
    # [0:9]  passport number (9)
    # [9]    passport check
    # [10:13] nationality (3)
    # [13:19] DOB YYMMDD (6)
    # [19]   DOB check
    # [20]   sex
    # [21:27] expiry YYMMDD (6)
    # [27]   expiry check
    # [28:42] personal number (14)
    # [42]   personal check
    # [43]   composite check

    def ck(s):
        return str(_make_mrz_checksum(s))

    pn = "A1234567<"          # 9 chars (passport number padded)
    nat = "IND"
    dob = "900412"
    sex = "F"
    exp = "300411"
    personal = "<<<<<<<<<<<<<<" # 14 chars
    personal_check = ck(personal)

    composite = pn + ck(pn) + nat + dob + ck(dob) + sex + exp + ck(exp) + personal + personal_check
    assert len(composite) == 43, f"composite len={len(composite)}"
    line2 = composite + ck(composite)
    assert len(line2) == 44

    line1 = "P<INDSHARMAA<<PRIYA".ljust(44, "<")
    assert len(line1) == 44
    mrz = line1 + line2

    fields = {"passport_number": "A1234567"}
    valid, warnings = validate_mrz(mrz, fields)
    assert valid is True
    assert warnings == []


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def test_calibration_all_fields_present():
    score = calibrate_confidence(
        raw_confidence=0.8,
        doc_type="pan",
        fields={"name": "Test", "fathers_name": "Father", "dob": "1990-01-01", "pan_number": "ABCDE1234F"},
        field_confidence={"name": 0.9, "fathers_name": 0.9, "dob": 0.9, "pan_number": 0.9},
        warnings=[],
        validation_passed={"pan": True},
    )
    assert score > 0.8  # positive signals should push it up

def test_calibration_missing_fields_penalised():
    score = calibrate_confidence(
        raw_confidence=0.8,
        doc_type="pan",
        fields={"name": None, "fathers_name": None, "dob": None, "pan_number": None},
        field_confidence={},
        warnings=["missing_required_field"],
        validation_passed={},
    )
    assert score < 0.8

def test_calibration_clamped_to_zero():
    score = calibrate_confidence(
        raw_confidence=0.0,
        doc_type="aadhaar",
        fields={},
        field_confidence={},
        warnings=["w"] * 20,
        validation_passed={},
    )
    assert score == 0.0

def test_calibration_clamped_to_one():
    score = calibrate_confidence(
        raw_confidence=1.0,
        doc_type="aadhaar",
        fields={f: "v" for f in ["name", "dob", "gender", "address", "aadhaar_number"]},
        field_confidence={f: 0.99 for f in ["name", "dob", "gender", "address", "aadhaar_number"]},
        warnings=[],
        validation_passed={},
    )
    assert score == 1.0


# ---------------------------------------------------------------------------
# validate() integration
# ---------------------------------------------------------------------------

def test_validate_unsupported_raises():
    with pytest.raises(ValueError, match="unsupported_document_type"):
        validate({"document_type": "unsupported", "confidence": 0.9, "fields": {}, "field_confidence": {}, "extraction_warnings": []}, [])

def test_validate_low_confidence_raises():
    with pytest.raises(ValueError, match="unsupported_document_type"):
        validate({"document_type": "pan", "confidence": 0.3, "fields": {}, "field_confidence": {}, "extraction_warnings": []}, [])

def test_validate_aadhaar_masked_in_output():
    result = validate({
        "document_type": "aadhaar",
        "confidence": 0.9,
        "fields": {"name": "Test", "dob": "1990-01-01", "gender": "male", "address": "Addr", "aadhaar_number": "123456789012"},
        "field_confidence": {},
        "extraction_warnings": [],
    }, [])
    assert result["masked"] is True
    assert "1234" not in result["fields"]["aadhaar_number"]
    assert result["fields"]["aadhaar_number"] == "XXXX XXXX 9012"

def test_validate_pan_invalid_adds_warning():
    result = validate({
        "document_type": "pan",
        "confidence": 0.85,
        "fields": {"name": "Test", "fathers_name": "F", "dob": "1990-01-01", "pan_number": "INVALID"},
        "field_confidence": {},
        "extraction_warnings": [],
    }, [])
    assert "invalid_pan_format" in result["extraction_warnings"]

def test_validate_dl_expired_adds_warning():
    result = validate({
        "document_type": "driving_licence",
        "confidence": 0.85,
        "fields": {"name": "T", "dob": "1990-01-01", "dl_number": "DL123", "validity_start_date": "2010-01-01", "validity_end_date": "2000-01-01", "vehicle_classes": ["LMV"], "address": "Addr"},
        "field_confidence": {},
        "extraction_warnings": [],
    }, [])
    assert "driving_licence_expired" in result["extraction_warnings"]

def test_validate_passport_has_mrz_valid_key():
    result = validate({
        "document_type": "passport",
        "confidence": 0.85,
        "fields": {"name": "T", "dob": "1990-01-01", "passport_number": "A1234567", "nationality": "IND", "expiry_date": "2030-01-01"},
        "field_confidence": {},
        "extraction_warnings": [],
        "mrz_line": None,
    }, [])
    assert "mrz_valid" in result
