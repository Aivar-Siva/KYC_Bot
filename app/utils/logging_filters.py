import logging
import re

# Aadhaar: 12 digits, spaced or unspaced
_AADHAAR_RE = re.compile(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b")
# PAN: AAAAA9999A
_PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
# MRZ TD3: two lines of exactly 44 chars containing '<'
_MRZ_RE = re.compile(r"[A-Z0-9<]{44}")


def _redact(text: str) -> str:
    text = _AADHAAR_RE.sub("[AADHAAR-REDACTED]", text)
    text = _PAN_RE.sub("[PAN-REDACTED]", text)
    text = _MRZ_RE.sub("[MRZ-REDACTED]", text)
    return text


class PIIRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact(str(record.msg))
        record.args = tuple(_redact(str(a)) for a in record.args) if record.args else record.args
        return True
