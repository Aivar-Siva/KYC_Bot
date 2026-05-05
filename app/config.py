from pydantic_settings import BaseSettings
from typing import Final


class Settings(BaseSettings):
    groq_api_key: str = ""
    bedrock_proxy_url: str = "https://ug36pewdpyfaepokw55klfit7y0ltgbn.lambda-url.us-west-2.on.aws/"

    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    bedrock_fallback_model: str = "us.meta.llama4-maverick-17b-instruct-v1:0"

    groq_timeout: float = 30.0
    groq_max_tokens: int = 2048

    max_upload_bytes: int = 5 * 1024 * 1024   # 5 MB
    groq_image_max_bytes: int = 3_500_000      # 3.5 MB
    groq_image_max_px: int = 2000              # longest side
    groq_image_quality: int = 85

    classification_threshold: float = 0.6
    rate_limit: str = "10/minute"

    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Accepted MIME types
ALLOWED_MIME: Final = {"image/jpeg", "image/png", "application/pdf"}
ALLOWED_EXTENSIONS: Final = {".jpg", ".jpeg", ".png", ".pdf"}

# Required fields per document type
REQUIRED_FIELDS: Final[dict[str, list[str]]] = {
    "aadhaar": ["name", "dob", "gender", "address", "aadhaar_number"],
    "pan": ["name", "fathers_name", "dob", "pan_number"],
    "voter_id": ["name", "fathers_or_husband_name", "dob", "voter_id_number", "address", "constituency"],
    "driving_licence": ["name", "dob", "dl_number", "validity_start_date", "validity_end_date", "vehicle_classes", "address"],
    "passport": ["name", "dob", "passport_number", "nationality", "expiry_date"],
}

SUPPORTED_TYPES: Final = list(REQUIRED_FIELDS.keys())
