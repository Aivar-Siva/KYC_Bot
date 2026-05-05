"""API routes: POST /kyc/process, GET /health, GET /version."""
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import ALLOWED_EXTENSIONS, settings, SUPPORTED_TYPES
from app.models.response_models import ErrorResponse, HealthResponse, KYCResponse, VersionResponse
from app.services import llm_service, mineru_service, preprocessing_service, validation_service
from app.services import storage_service
from app.utils.temp_files import temp_file

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


def _validate_upload(file: UploadFile, data: bytes) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds 5MB limit")


@router.post("/kyc/process", response_model=KYCResponse)
@limiter.limit(settings.rate_limit)
async def process_kyc(
    request: Request,
    file: UploadFile = File(...),
    expected_type: str | None = Form(default=None),
    uploader_name: str = Form(default="anonymous"),
):
    start = time.monotonic()
    data = await file.read()
    _validate_upload(file, data)

    if expected_type and expected_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid expected_type: {expected_type}")

    ext = Path(file.filename or "").suffix.lower()
    is_pdf = ext == ".pdf"

    with temp_file(suffix=ext) as tmp_path:
        tmp_path.write_bytes(data)
        logger.info("Processing file type=%s size=%d uploader=%s", ext, len(data), uploader_name)

        try:
            # ── Preprocess ────────────────────────────────────────────────
            if is_pdf:
                import fitz, cv2, numpy as np
                doc = fitz.open(str(tmp_path))
                pix = doc[0].get_pixmap(dpi=150)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                doc.close()
                with temp_file(suffix=".png") as img_tmp:
                    cv2.imwrite(str(img_tmp), img_array)
                    _, groq_bytes, pre_warnings = preprocessing_service.preprocess(img_tmp)
                mineru_text, mineru_warnings = mineru_service.extract_text(pdf_path=tmp_path)
            else:
                full_img, groq_bytes, pre_warnings = preprocessing_service.preprocess(tmp_path)
                mineru_text, mineru_warnings = mineru_service.extract_text(img=full_img)

            # ── LAYER 1: Is this the correct document type? ───────────────
            if expected_type:
                is_correct, gate_reason = llm_service.gate_document_type(expected_type, groq_bytes)
                if not is_correct:
                    return JSONResponse(
                        status_code=422,
                        content=ErrorResponse(
                            error="wrong_document_type",
                            message=f"This does not appear to be a {expected_type}. {gate_reason}",
                        ).model_dump(),
                    )

            # ── LAYER 2: Extract + validate + store + consistency ─────────
            llm_output, llm_warnings = llm_service.extract(mineru_text, groq_bytes)

            all_extra = pre_warnings + mineru_warnings + llm_warnings
            result = validation_service.validate(llm_output, all_extra)

            # Enforce type match post-extraction
            if expected_type and result["document_type"] != expected_type:
                return JSONResponse(
                    status_code=422,
                    content=ErrorResponse(
                        error="wrong_document_type",
                        message=f"Expected {expected_type} but extracted {result['document_type']}.",
                    ).model_dump(),
                )

            # Fraud verification
            passed, reasoning, failed_checks = llm_service.verify_document(
                result["fields"], result["document_type"], groq_bytes, expected_type=expected_type
            )
            if not passed:
                return JSONResponse(
                    status_code=422,
                    content=ErrorResponse(
                        error="document_verification_failed",
                        message=reasoning or "Document failed fraud verification.",
                        supported=failed_checks if failed_checks else None,
                    ).model_dump(),
                )

            # Store result for this uploader
            storage_service.store(uploader_name, result["document_type"], result)

            # Cross-document person consistency
            consistency = storage_service.check_person_consistency(
                uploader_name, result["document_type"], result["fields"]
            )
            result["consistency_report"] = consistency

            # ── LAYER 3: Return result ────────────────────────────────────
        except ValueError as exc:
            if "unsupported_document_type" in str(exc):
                return JSONResponse(
                    status_code=422,
                    content=ErrorResponse(
                        error="unsupported_document_type",
                        message="Document is not one of the 5 supported types",
                        supported=SUPPORTED_TYPES,
                    ).model_dump(),
                )
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            if "processing_error" in str(exc):
                return JSONResponse(
                    status_code=503,
                    content=ErrorResponse(
                        error="processing_error",
                        message="Extraction service unavailable, please retry",
                    ).model_dump(),
                )
            raise HTTPException(status_code=500, detail="Internal error")

    elapsed = time.monotonic() - start
    logger.info("Completed doc_type=%s confidence=%.3f duration=%.2fs uploader=%s",
                result.get("document_type"), result.get("confidence"), elapsed, uploader_name)
    return KYCResponse(**result)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse(version=settings.app_version)
