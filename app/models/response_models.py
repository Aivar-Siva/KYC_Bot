from typing import Any
from pydantic import BaseModel


class KYCResponse(BaseModel):
    document_type: str
    confidence: float
    fields: dict[str, Any]
    field_confidence: dict[str, float]
    masked: bool
    extraction_warnings: list[str]
    mrz_valid: bool | None = None  # passport only


class ErrorResponse(BaseModel):
    error: str
    message: str
    supported: list[str] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


class VersionResponse(BaseModel):
    version: str
