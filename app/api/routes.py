"""API routes: POST /kyc/process, GET /health, GET /version."""
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import ALLOWED_EXTENSIONS, settings, SUPPORTED_TYPES
from app.models.response_models import ErrorResponse, HealthResponse, KYCResponse, VersionResponse
from app.services import llm_service, mineru_service, preprocessing_service, validation_service
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
async def process_kyc(request: Request, file: UploadFile = File(...)):
    start = time.monotonic()
    data = await file.read()
    _validate_upload(file, data)

    ext = Path(file.filename or "").suffix.lower()
    is_pdf = ext == ".pdf"

    with temp_file(suffix=ext) as tmp_path:
        tmp_path.write_bytes(data)
        logger.info("Processing file type=%s size=%d", ext, len(data))

        try:
            if is_pdf:
                # PyMuPDF handles text extraction and rasterization
                mineru_text, mineru_warnings = mineru_service.extract_text(pdf_path=tmp_path)
                # Render first page for Groq vision
                import fitz
                import cv2
                import numpy as np
                doc = fitz.open(str(tmp_path))
                pix = doc[0].get_pixmap(dpi=150)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                doc.close()
                with temp_file(suffix=".png") as img_tmp:
                    cv2.imwrite(str(img_tmp), img_array)
                    _, groq_bytes, pre_warnings = preprocessing_service.preprocess(img_tmp)
            else:
                full_img, groq_bytes, pre_warnings = preprocessing_service.preprocess(tmp_path)
                mineru_text, mineru_warnings = mineru_service.extract_text(img=full_img)

            # LLM extraction
            llm_output, llm_warnings = llm_service.extract(mineru_text, groq_bytes)

            # Validation + calibration
            all_extra = pre_warnings + mineru_warnings + llm_warnings
            result = validation_service.validate(llm_output, all_extra)

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
    logger.info("Completed doc_type=%s confidence=%.3f duration=%.2fs",
                result.get("document_type"), result.get("confidence"), elapsed)
    return KYCResponse(**result)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/version", response_model=VersionResponse)
async def version():
    return VersionResponse(version=settings.app_version)
