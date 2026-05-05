"""Layer 2: persist extraction results and check cross-document person consistency via LLM."""
import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("output")


def _uploader_dir(uploader_name: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", uploader_name.strip())
    return OUTPUT_ROOT / safe


def store(uploader_name: str, doc_type: str, result: dict) -> Path:
    """Save extraction result to output/{uploader_name}/{doc_type}.json."""
    folder = _uploader_dir(uploader_name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc_type}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Stored %s for uploader=%s", doc_type, uploader_name)
    return path


def load_all(uploader_name: str) -> dict[str, dict]:
    """Load all previously stored documents for this uploader."""
    folder = _uploader_dir(uploader_name)
    if not folder.exists():
        return {}
    docs = {}
    for f in folder.glob("*.json"):
        try:
            docs[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return docs


def _llm_consistency_check(new_doc_type: str, new_fields: dict, existing: dict[str, dict]) -> dict:
    """
    Ask the LLM to reason across all stored documents + the new one
    and determine if they all belong to the same person.
    """
    from app.config import settings

    # Build a readable summary of all documents
    doc_summaries = []
    for doc_type, stored in existing.items():
        fields = stored.get("fields", {})
        relevant = {k: v for k, v in fields.items() if k in ("name", "dob", "fathers_name", "fathers_or_husband_name") and v}
        doc_summaries.append(f"[{doc_type}]: {json.dumps(relevant)}")

    new_relevant = {k: v for k, v in new_fields.items() if k in ("name", "dob", "fathers_name", "fathers_or_husband_name") and v}
    doc_summaries.append(f"[{new_doc_type} — NEW]: {json.dumps(new_relevant)}")

    prompt = (
        "You are a KYC identity consistency analyst.\n\n"
        "Below are identity fields extracted from multiple documents submitted by the same applicant.\n"
        "Determine if all documents belong to the same person.\n\n"
        "Rules:\n"
        "- Name variations like 'PRIYA SHARMA' vs 'Priya Sharma' are acceptable (case difference only)\n"
        "- Abbreviated names like 'P. Sharma' vs 'Priya Sharma' are acceptable if surname matches\n"
        "- Different names entirely (e.g. 'Priya Sharma' vs 'Rahul Gupta') are NOT acceptable\n"
        "- DOB must match exactly across all documents that have it\n"
        "- Father's name variations follow the same rules as name\n\n"
        "Documents:\n" + "\n".join(doc_summaries) + "\n\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "consistent": true or false,\n'
        '  "reliability": 0.0 to 1.0 (fraction of documents that agree with the majority identity),\n'
        '  "mismatches": [{"field": "...", "doc1": "...", "value1": "...", "doc2": "...", "value2": "..."}],\n'
        '  "reasoning": "one sentence summary"\n'
        "}"
    )

    try:
        import httpx
        payload = {
            "model_id": settings.bedrock_fallback_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
        resp = httpx.post(settings.bedrock_proxy_url, json=payload, timeout=30.0)
        resp.raise_for_status()
        raw = resp.json()
        content = raw.get("content") or raw.get("text") or raw.get("generation") or str(raw)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            result = json.loads(m.group()) if m else {}

        return {
            "consistent": bool(result.get("consistent", True)),
            "checked_against": list(existing.keys()),
            "mismatches": result.get("mismatches", []),
            "reliability": float(result.get("reliability", 1.0)),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as exc:
        logger.warning("LLM consistency check failed, falling back to basic check: %s", exc)
        return _basic_consistency_check(new_doc_type, new_fields, existing)


def _basic_consistency_check(new_doc_type: str, new_fields: dict, existing: dict[str, dict]) -> dict:
    """Simple string-based fallback when LLM is unavailable."""
    mismatches = []
    agree = 0
    for doc_type, stored in existing.items():
        sf = stored.get("fields", {})
        new_name = re.sub(r"\s+", " ", str(new_fields.get("name") or "")).strip().lower()
        stored_name = re.sub(r"\s+", " ", str(sf.get("name") or "")).strip().lower()
        new_dob = str(new_fields.get("dob") or "").strip()
        stored_dob = str(sf.get("dob") or "").strip()
        doc_ok = True
        if new_name and stored_name and new_name != stored_name:
            mismatches.append({"field": "name", "doc1": new_doc_type, "value1": new_fields.get("name"), "doc2": doc_type, "value2": sf.get("name")})
            doc_ok = False
        if new_dob and stored_dob and new_dob != stored_dob:
            mismatches.append({"field": "dob", "doc1": new_doc_type, "value1": new_dob, "doc2": doc_type, "value2": stored_dob})
            doc_ok = False
        if doc_ok:
            agree += 1
    return {
        "consistent": len(mismatches) == 0,
        "checked_against": list(existing.keys()),
        "mismatches": mismatches,
        "reliability": round(agree / len(existing), 2),
        "reasoning": "Basic string match (LLM unavailable)",
    }


def check_person_consistency(uploader_name: str, new_doc_type: str, new_fields: dict) -> dict:
    """
    Load ALL previously stored documents for this uploader and pass them to the LLM
    to reason whether they all belong to the same person as the new document.
    """
    existing = load_all(uploader_name)
    existing.pop(new_doc_type, None)  # exclude the doc just stored

    if not existing:
        return {"consistent": True, "checked_against": [], "mismatches": [], "reliability": 1.0, "reasoning": "First document — no prior documents to compare."}

    return _llm_consistency_check(new_doc_type, new_fields, existing)
