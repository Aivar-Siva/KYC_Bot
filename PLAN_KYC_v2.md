# PLAN_KYC_v2.md

## Objective

Build an end-to-end KYC document processing pipeline for Indian identity documents that accepts image or PDF input, identifies the document type, extracts structured fields, validates document-specific constraints, enforces Aadhaar masking, and returns a production-ready JSON response through an API.

The system will support exactly five document types:
- Aadhaar Card
- PAN Card
- Voter ID
- Driving Licence
- Passport

The system will reject unsupported documents gracefully and must work on real-world scans, including low-quality webcam captures, skewed images, partial occlusion, and multilingual content including Hindi and state-language variants.

---

## Final Architecture

```text
User Input (scanned image / PDF upload, max 5MB)
        ↓
FastAPI upload endpoint
  - File type validation
  - File size enforcement
  - Rate limiting
        ↓
Input normalizer
  - Validate and sanitize MIME type
  - Convert PDF: select identity page, rasterize to image
  - Store in-memory temp working copy
        ↓
Image preprocessing layer  ← NEW
  - Deskew (correct rotation and perspective)
  - Denoise
  - Upscale if resolution too low
  - Enhance contrast for faded or low-light scans
        ↓
MinerU
  - Parse preprocessed image or PDF
  - Extract text blocks in reading order
  - Extract layout regions
  - Provide structured JSON representation
        ↓
LLM reasoning layer (multimodal — always receives image + MinerU text)
  - Classify document type
  - Extract structured fields
  - Handle Hindi and regional language fields natively via vision
  - Produce field-level confidence
  - Produce structured warnings
        ↓
Confidence calibration layer  ← NEW
  - Adjust LLM-reported confidence using rule-based signals
  - Penalize null fields, low field confidence, active warnings
  - Reward passing validation checks
        ↓
Validation & compliance layer
  - PAN regex validation
  - Aadhaar masking (enforced before any output construction)
  - DL expiry validation
  - Passport MRZ checksum and field consistency check  ← STRENGTHENED
  - Unsupported-document rejection
        ↓
Temp file cleanup (always runs, even on error)  ← NEW
        ↓
Response formatter
  - Standard JSON schema
  - Warning normalization
  - Confidence normalization
        ↓
API response (PII-safe, masked, log-filtered)
```

This design intentionally avoids custom model training. MinerU performs document parsing, the multimodal LLM handles semantic understanding and classification, and deterministic validators enforce all compliance rules.

---

## Why This Design

### 1. No custom model training
The task can be completed without training a classifier. MinerU extracts structured document content, and a multimodal LLM can infer the document type from extracted text, layout, and the original image. This eliminates dataset collection, training, tuning, and evaluation overhead.

### 2. Image preprocessing before MinerU
MinerU is optimized for PDFs and well-structured documents. Scanned images can be skewed, low-resolution, or poorly lit. Preprocessing with OpenCV normalizes image quality before MinerU processes it. This prevents garbage text extraction from degraded inputs and is essential for real-world KYC scan quality.

### 3. MinerU as the document parser
MinerU recovers layout structure, separates content regions, and exposes text in reading order. This is more useful than raw OCR dumps, especially for Indian identity documents where field positions vary across document variants and issuers.

### 4. Multimodal LLM — always, not optional
The LLM must receive both the MinerU text output and the original preprocessed image on every request. Text-only input fails when MinerU extraction is poor. The image acts as a safety net: the LLM can read fields directly from the visual when text extraction is incomplete. This is especially important for Hindi text on Aadhaar cards and regional language Voter IDs, which MinerU may not handle accurately.

**Primary model:** `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API. Supports vision (image + text), JSON mode, multilingual including Hindi/Devanagari, 128K context window.

**Fallback model:** `us.meta.llama4-maverick-17b-instruct-v1:0` via Bedrock proxy (`https://ug36pewdpyfaepokw55klfit7y0ltgbn.lambda-url.us-west-2.on.aws/`). Text-only — used when Groq is unavailable. Adds warning `vision_unavailable` to response when fallback is active.

**Timeout and retry:** 30-second timeout on Groq call. On timeout or HTTP error: one retry after 2 seconds. If second attempt fails: switch to Bedrock fallback. If Bedrock also fails: return `processing_error`.

### 5. Deterministic validators for compliance
Compliance rules must never rely on LLM output:
- Aadhaar masking
- PAN format validation
- Driving licence expiry check
- Passport MRZ checksum and field consistency
These are enforced with explicit logic after extraction.

### 6. Rule-based confidence calibration
LLM-reported confidence is not probabilistic — it reflects the model's self-assessment, which is unreliable for scoring. A separate calibration step applies rule-based adjustments based on field completeness, validation outcomes, and warning count to produce a more meaningful confidence score.

### 7. Guaranteed temp file cleanup
All uploaded files and intermediate images must be deleted after each request, even if the pipeline raises an exception. Cleanup must be enforced at the infrastructure level, not left to happy-path logic.

---

## Supported Inputs

### Accepted file formats
- `.jpg`
- `.jpeg`
- `.png`
- `.pdf`

### Input size constraint
- Maximum file size: **5 MB** per request
- Requests exceeding this limit must be rejected with a clear error before processing begins

### Input channels
- File upload from local machine (scanned image or PDF)

### Recommended capture mode
Scanned document upload. A flatbed or mobile scan at 200–300 DPI produces the best extraction results. Webcam capture is not supported.

### Input assumptions
- One document per request
- Front side only unless passport data page is provided
- PDF may contain one or more pages; the pipeline selects the most relevant identity page

### PDF page selection
For multi-page PDFs, the pipeline should:
- Process the first page by default
- If MinerU returns low-confidence text from page 1, attempt page 2
- Never process more than two pages per request

---

## Image Preprocessing Layer

This layer runs on every image input before MinerU. It is the primary defense against degraded scan quality.

### Operations in order
1. Deskew — detect and correct document rotation and perspective distortion
2. Denoise — reduce noise from low-quality scans
3. Resolution check — if image is below a usable DPI threshold, apply upscaling
4. Contrast enhancement — improve readability of faded or poorly lit documents
5. Border detection — optionally crop to document boundaries if background is present
6. Compress for Groq — produce a separate JPEG copy at quality 85, resized to max 2000px on longest side, targeting ≤3.5MB. This copy is sent to Groq. MinerU receives the full-quality preprocessed image from step 5.

### Output
Two images are produced:
- **Full-quality copy** — passed to MinerU for text extraction
- **Compressed copy (≤3.5MB JPEG)** — passed to Groq API as base64. Kept separate to avoid degrading MinerU input quality.

### Failure behavior
If preprocessing fails, pass the original image through unchanged and add warning `preprocessing_failed`. Do not block the pipeline.

---

## Document Types and Expected Fields

### Aadhaar Card
Required fields:
- name
- dob
- gender
- address
- aadhaar_number

Special rules:
- First 8 digits of Aadhaar number must be masked in all outputs
- Unmasked Aadhaar number must never appear in any API response or log
- Aadhaar includes Hindi text; the LLM must use image input to read fields when MinerU text is incomplete

Output example:
```json
{
  "document_type": "aadhaar",
  "confidence": 0.94,
  "fields": {
    "name": "Priya Sharma",
    "dob": "1990-04-12",
    "gender": "female",
    "address": "12 MG Road, Bengaluru, Karnataka",
    "aadhaar_number": "XXXX XXXX 3456"
  },
  "field_confidence": {
    "name": 0.99,
    "dob": 0.95,
    "gender": 0.99,
    "address": 0.88,
    "aadhaar_number": 0.97
  },
  "masked": true,
  "extraction_warnings": []
}
```

### PAN Card
Required fields:
- name
- fathers_name
- dob
- pan_number

Validation rule:
- PAN format must match the pattern: 5 uppercase letters, 4 digits, 1 uppercase letter
- Example valid format: ABCDE1234F

### Voter ID
Required fields:
- name
- fathers_or_husband_name
- dob
- voter_id_number
- address
- constituency

Notes:
- Voter IDs vary significantly by state in layout and language
- The LLM must use image input to handle state-language text not captured by MinerU

### Driving Licence
Required fields:
- name
- dob
- dl_number
- validity_start_date
- validity_end_date
- vehicle_classes
- address

Validation rule:
- Expiry date must be in the future relative to the date of processing
- If expired, add warning `driving_licence_expired` and continue extraction

### Passport
Required fields:
- name
- dob
- passport_number
- nationality
- expiry_date

MRZ handling:
- The LLM extracts the MRZ line from the image for internal validation only
- TD3 checksums are validated against extracted biographical fields
- `mrz_valid` (boolean) is returned in the response — the raw MRZ line is never returned
- Any checksum failure or field mismatch triggers warning `mrz_mismatch_with_biographical_data`

---

## End-to-End Flow

### Step 1: Upload
The frontend accepts a scanned image or PDF file (max 5MB). The file is sent to `POST /kyc/process`.

### Step 2: Input normalization
The backend enforces file type, file size limit, and rate limit before any processing begins. For PDFs, the relevant identity page is rasterized to an image. The file is held in memory or a secure temp location.

### Step 3: Image preprocessing
The image passes through the preprocessing layer: deskew, denoise, upscale if needed, contrast enhancement. The normalized image is the input to both MinerU and the LLM.

### Step 4: MinerU parsing
MinerU processes the normalized image or PDF and returns ordered text blocks, layout elements, and region labels. The output is a structured text representation of the document content in reading order.

### Step 5: LLM classification and extraction
The LLM receives both the MinerU text output and the preprocessed image. It classifies the document type, extracts all required fields, assigns per-field confidence scores, and emits structured warnings. The LLM uses image input to resolve fields that MinerU text could not capture, including Hindi and regional language content.

### Step 6: Confidence calibration
The pipeline applies rule-based adjustments to the LLM's reported confidence:
- All required fields present and non-null: positive adjustment
- Any null required field: negative adjustment per missing field
- Validation checks pass: positive adjustment
- Active warnings: negative adjustment per warning
- LLM-reported field confidence uniformly high: positive adjustment

Final confidence is clamped between 0.0 and 1.0.

### Step 7: Validation and compliance
Deterministic validators run in order:
- Reject unsupported documents
- Mask Aadhaar number before any output object is constructed
- Validate PAN format
- Validate DL expiry date
- Validate passport MRZ checksums and field consistency
- Attach warnings for any validation failures

### Step 8: Temp file cleanup
All uploaded files and intermediate images are deleted. This step must execute even if earlier steps raised exceptions.

### Step 9: Response formatting
The pipeline assembles the final JSON response from validated, masked, and calibrated output and returns it.

---

## MinerU Role in This System

MinerU is responsible for document parsing and text extraction only.

### MinerU should do
- Accept preprocessed image or PDF
- Extract text from scanned documents in reading order
- Recover document layout and region structure
- Expose text blocks as structured JSON

### MinerU should not be trusted alone for
- Final document type classification
- Aadhaar masking
- PAN regex enforcement
- Expiry logic
- MRZ validation
- Rejection of unsupported documents
- Hindi or regional language text (use LLM image input instead)

The principle: MinerU extracts, the LLM interprets, Python enforces.

---

## LLM Design

### Input to the LLM (always both)
- MinerU text output with layout structure
- Preprocessed image of the document

Sending only text is not acceptable. The image is required on every request to handle degraded extractions and multilingual content.

### Responsibilities of the LLM
1. Classify the document type from text and visual content
2. Map extracted content to the target schema for that document type
3. Estimate field-level confidence for each extracted field
4. Emit structured warnings when fields are uncertain, partially occluded, or unreadable

### Prompt design principles
- Provide the strict list of supported document types
- Provide exact field names expected for each type
- Instruct the LLM to return only valid JSON with no preamble
- Instruct the LLM to set `document_type` to `unsupported` when the document cannot be classified
- Instruct the LLM not to hallucinate or invent field values
- Instruct the LLM to use `null` for missing or unreadable fields
- Instruct the LLM to normalize dates to `YYYY-MM-DD` format
- Instruct the LLM to add warning names rather than guessing when scan quality is low
- Instruct the LLM to use the image to recover fields where text extraction is incomplete

### LLM output schema
```json
{
  "document_type": "aadhaar | pan | voter_id | driving_licence | passport | unsupported",
  "confidence": 0.0,
  "fields": {},
  "field_confidence": {},
  "extraction_warnings": [],
  "mrz_line": "raw MRZ string — passport only, used internally for validation, never forwarded to API response"
}
```

### JSON mode
Groq API JSON mode is used on every request (`response_format: {"type": "json_object"}`). This guarantees valid JSON output from the primary model — no retry logic needed for malformed JSON on the Groq path.

When the Bedrock fallback is active (text-only), JSON mode is unavailable. In that case: attempt to extract a JSON object from the response using a `{...}` regex before declaring failure. If extraction fails, return `processing_error`.

---

## Confidence Calibration

LLM-reported confidence scores are self-assessments and should not be used as-is. The calibration layer applies the following signal-based adjustments on top of the LLM score:

### Positive signals
- All required fields for the identified document type are non-null
- PAN format validation passes
- DL expiry is valid and in the future
- Passport MRZ checksums and field matches pass
- All field-level confidence values reported by the LLM are above 0.85

### Negative signals
- Each null required field
- Each active extraction warning
- MRZ checksum failure
- LLM classification confidence below **0.6** (also triggers unsupported rejection)

The adjusted score is clamped to the range 0.0 to 1.0 and replaces the raw LLM confidence in the response.

---

## Validation Rules

### Aadhaar masking
- The raw Aadhaar number may exist only in temporary in-memory processing during extraction
- Masking must occur before the output object is constructed — not after
- The API response must contain only the masked form: `XXXX XXXX 1234`
- Log filters must redact Aadhaar numbers (spaced and unspaced), PAN numbers, and MRZ lines from all log output
- The pattern to redact covers both `123456789012` and `1234 5678 9012` formats
- PAN pattern `[A-Z]{5}[0-9]{4}[A-Z]` is also redacted
- MRZ lines (44-char TD3 pattern) are also redacted
- No unmasked Aadhaar number may appear in any log, response body, or serialized intermediate

### PAN validation
- Format: 5 uppercase letters, 4 digits, 1 uppercase letter
- If invalid: add warning `invalid_pan_format` and reduce calibrated confidence
- Keep the extracted value internally for reference but never return an unvalidated PAN as confirmed

### Driving licence validation
- Parse expiry date safely, handling common Indian date formats
- Compare expiry date against the current date at time of processing
- If expired: add warning `driving_licence_expired`
- Extraction continues even for expired licences; expiry is a warning, not a rejection

### Passport MRZ validation
Full TD3 MRZ validation is required:
- Validate passport number checksum digit
- Validate date of birth checksum digit
- Validate expiry date checksum digit
- Validate composite checksum
- Compare MRZ-derived passport number, DOB, and expiry date against the extracted biographical fields
- Any checksum failure or biographical mismatch: add warning `mrz_mismatch_with_biographical_data`
- Set `mrz_valid: true` if all checks pass, `mrz_valid: false` otherwise
- The raw MRZ line must never appear in the API response or logs

Use a library that implements TD3 MRZ checksum logic rather than reimplementing from scratch.

### Unsupported document handling
If the LLM returns `unsupported` or calibrated confidence is below **0.6**:
```json
{
  "error": "unsupported_document_type",
  "message": "Document is not one of the 5 supported types",
  "supported": ["aadhaar", "pan", "voter_id", "driving_licence", "passport"]
}
```

---

## Response Schema

### Success response
```json
{
  "document_type": "aadhaar",
  "confidence": 0.94,
  "fields": {
    "name": "Priya Sharma",
    "dob": "1990-04-12",
    "gender": "female",
    "address": "12 MG Road, Bengaluru, Karnataka",
    "aadhaar_number": "XXXX XXXX 3456"
  },
  "field_confidence": {
    "name": 0.99,
    "dob": 0.95,
    "gender": 0.99,
    "address": 0.88,
    "aadhaar_number": 0.97
  },
  "masked": true,
  "extraction_warnings": []
}
```

For passport responses, `mrz_valid` is included instead of the raw MRZ line:
```json
{
  "document_type": "passport",
  "confidence": 0.96,
  "fields": {
    "name": "Priya Sharma",
    "dob": "1990-04-12",
    "passport_number": "A1234567",
    "nationality": "IND",
    "expiry_date": "2030-04-11"
  },
  "field_confidence": {...},
  "mrz_valid": true,
  "masked": false,
  "extraction_warnings": []
}
```

### Error response
```json
{
  "error": "unsupported_document_type",
  "message": "Document is not one of the 5 supported types",
  "supported": ["aadhaar", "pan", "voter_id", "driving_licence", "passport"]
}
```

### Warning catalog
- `low_scan_quality_on_address_field`
- `dob_partially_occluded`
- `invalid_pan_format`
- `driving_licence_expired`
- `mrz_mismatch_with_biographical_data`
- `classification_low_confidence`
- `preprocessing_failed`
- `mineru_low_confidence_text`
- `missing_required_field`
- `vision_unavailable` — Groq unavailable, Bedrock text-only fallback was used

---

## API Plan

### Primary endpoint
`POST /kyc/process`

Request:
- Content type: `multipart/form-data`
- Field: `file`
- Maximum file size: 10 MB

Response:
- JSON in standard response schema

### Secondary endpoints
- `GET /health` — liveness check
- `GET /version` — deployment metadata

### Rate limiting
The `/kyc/process` endpoint must have rate limiting applied to prevent abuse. A library such as `slowapi` provides this for FastAPI. Requests exceeding the rate limit should return HTTP 429.

---

## Frontend / Demo Plan

### Input mode
File upload only — scanned image or PDF, max 5MB.

### Demo flow
1. User selects a scanned document file
2. File is posted to `POST /kyc/process`
3. JSON result is rendered in the UI with field display and warning list

A single HTML page with a file input and a JSON viewer is sufficient.

---

## Security and Compliance Considerations

### Sensitive data handling
- Never store raw Aadhaar in logs or persistent storage
- Avoid persisting any uploaded document after request completion
- Delete temp files in a cleanup step that always runs
- Keep API responses minimal — return only required fields

### Logging policy
Allowed in logs:
- Request ID
- File type
- Document type (after classification)
- Processing duration
- Warning codes

Never allowed in logs:
- Unmasked Aadhaar number in any format
- PAN numbers
- Raw MRZ lines (in any mode)

### Log redaction
A log filter must intercept all log output and redact the following before any log line is written:
- **Aadhaar numbers** — any 12-digit sequence in spaced (`1234 5678 9012`) or unspaced (`123456789012`) format
- **PAN numbers** — any 10-character sequence matching `[A-Z]{5}[0-9]{4}[A-Z]`
- **MRZ lines** — any line matching the TD3 pattern (44-character lines containing `<` characters)

### Temp file cleanup
All uploaded files must be deleted after the request completes. Intermediate images generated by preprocessing or MinerU must also be cleaned up. This must run in a `finally` block or equivalent cleanup guarantee so that an exception in the pipeline cannot leave PII on disk.

---

## Failure Modes and Fallbacks

### Failure mode: LLM timeout or API error
- Groq primary: 30-second timeout, 1 retry after 2 seconds
- If both Groq attempts fail: switch to Bedrock fallback (`us.meta.llama4-maverick-17b-instruct-v1:0`), add warning `vision_unavailable`
- If Bedrock also fails: return `{"error": "processing_error", "message": "Extraction service unavailable, please retry"}`

### Failure mode: preprocessing fails
Fallback: pass original image unchanged, add warning `preprocessing_failed`, continue

### Failure mode: MinerU returns poor extraction
Fallback: the LLM still receives the original image and can extract fields visually; add warning `mineru_low_confidence_text`

### Failure mode: LLM returns malformed JSON
Groq primary uses JSON mode — malformed JSON cannot occur on that path. On Bedrock fallback: attempt regex extraction of `{...}` from response. If that fails, return `processing_error`.

### Failure mode: document type unclear
Fallback: return `unsupported` with warning `classification_low_confidence`

### Failure mode: missing critical field
Fallback: set field to `null`, add warning `missing_required_field`

### Failure mode: MRZ checksum fails
Fallback: add warning `mrz_mismatch_with_biographical_data`, return extraction with reduced confidence, do not reject

### Failure mode: exception during processing
Fallback: temp file cleanup still runs; return a safe generic error response with no PII

---

## Prompting Strategy

The LLM prompt must contain:
- System instruction establishing the KYC extraction role
- Strict list of five supported document types
- Exact field names for each document type
- Instruction to classify as `unsupported` when document is not one of the five types
- Instruction to return only valid JSON with no preamble or explanation
- Instruction to use `null` for missing or unreadable fields, not invented values
- Instruction to normalize all dates to `YYYY-MM-DD`
- Instruction to produce per-field confidence between 0.0 and 1.0
- Instruction to use the image to recover fields where text extraction is incomplete
- Instruction to add warning names for low-confidence or occluded fields rather than guessing

---

## Suggested Project Structure

```text
kyc-pipeline/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── preprocessing_service.py       ← NEW
│   │   ├── mineru_service.py
│   │   ├── llm_service.py
│   │   ├── confidence_service.py          ← NEW
│   │   ├── validation_service.py
│   │   └── masking_service.py
│   ├── models/
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── utils/
│   │   ├── temp_files.py                  ← cleanup guarantee
│   │   ├── logging_filters.py             ← Aadhaar redaction
│   │   └── date_utils.py
│   └── config.py
├── tests/
│   ├── samples/
│   ├── degraded/
│   └── test_pipeline.py
├── frontend/
│   └── demo.html
├── output/
└── requirements.txt
```

---

## Detailed Implementation Phases

### Phase 1 — Core backend scaffold
Deliverables:
- FastAPI app skeleton with upload endpoint
- File type and size validation
- Rate limiting
- Temp file management with guaranteed cleanup
- Basic error handling

Success criteria:
- Endpoint accepts image and PDF
- Oversized files and unsupported types are rejected cleanly
- Temp files are deleted even when exceptions occur

---

### Phase 2 — Image preprocessing layer
Deliverables:
- Preprocessing service: deskew, denoise, upscale, contrast enhancement
- Integration into pipeline before MinerU

Success criteria:
- A skewed webcam photo of a document is visibly corrected before MinerU processing
- Preprocessing failure does not block the pipeline

---

### Phase 3 — MinerU integration
Deliverables:
- MinerU service wrapper
- Normalized parser output with reading-order text blocks

Success criteria:
- Sample Aadhaar, PAN, and Passport documents produce usable text blocks from MinerU

---

### Phase 4 — LLM extraction layer
Deliverables:
- Multimodal LLM prompt (text + image always)
- JSON-only response parser with retry
- Schema mapping for all five document types
- Date and field normalization

Success criteria:
- At least 3 document types extracted successfully
- Hindi fields on Aadhaar extracted correctly via image input
- Unsupported documents return `unsupported`
- Malformed LLM response triggers retry and fallback

---

### Phase 5 — Validation, masking, and confidence calibration
Deliverables:
- Aadhaar masking utility with log redaction
- PAN format validator
- DL expiry validator
- Passport MRZ TD3 checksum validator
- Rule-based confidence calibration layer

Success criteria:
- Aadhaar is always masked in API output and logs
- Invalid PAN is flagged with warning
- Expired DL is flagged with warning
- MRZ checksum failure is flagged with warning
- Calibrated confidence reflects actual extraction quality

---

### Phase 6 — Demo UI
Deliverables:
- HTML page with file upload (image/PDF) and JSON response viewer with field display and warning list

Success criteria:
- User can upload a scanned document and see extracted structured output

---

### Phase 7 — Testing and evaluation
Deliverables:
- Test document set covering at least 3 document types
- At least one degraded scan
- At least one unsupported document
- Captured output JSON for each test case
- Notes on failure modes observed

Success criteria:
- Submission requirement satisfied
- Pipeline demonstrates graceful degradation and all special validation rules

---

## Testing Plan

### Minimum required test coverage
At least three document types including one degraded scan.

### Recommended test matrix

| Test Case | Input Type | Expected Result |
|---|---|---|
| Aadhaar clean | image/PDF | correct extraction + masked Aadhaar |
| Aadhaar with Hindi fields | scanned image | Hindi fields extracted via image input |
| Aadhaar degraded | scanned image | masked output + low quality warning |
| PAN clean | image/PDF | correct extraction + PAN regex pass |
| Passport clean | image/PDF | correct extraction + MRZ checksum pass |
| Driving licence expired | image/PDF | extraction + expiry warning |
| Voter ID low quality | scanned image | extraction with warnings |
| Unsupported document | image/PDF | graceful rejection |
| Skewed scan | image | preprocessing corrects before extraction |

### Evaluation dimensions
- Document classification accuracy
- Field extraction completeness
- Field accuracy
- Warning quality and relevance
- Masking correctness (Aadhaar must never appear unmasked in any output)
- MRZ checksum validation correctness
- Calibrated confidence correlation with actual extraction quality
- Unsupported document rejection quality
- Latency per document

---

## What to Show in the Submission

### Code
- Working API endpoint with rate limiting and file size enforcement
- Image preprocessing service
- MinerU integration
- Multimodal LLM extraction layer
- Confidence calibration layer
- Validators and masking rules
- Guaranteed temp file cleanup
- Log redaction filter

### Write-up
Explain:
- Why image preprocessing is required before MinerU for webcam inputs
- Why MinerU alone is insufficient and the LLM always receives the image
- How Hindi and regional language content is handled via multimodal LLM
- Why deterministic validators enforce compliance rather than the LLM
- How Aadhaar masking is guaranteed at the output construction layer
- How passport MRZ TD3 checksum validation works
- How confidence calibration produces a more reliable score than raw LLM output

### Demo cases
Show at least:
- One Aadhaar sample with Hindi text
- One PAN or Passport sample with MRZ validation result
- One degraded scan demonstrating preprocessing and graceful degradation
- One unsupported document rejection

---

## Final Recommendation

The strongest implementation for this task is:
- OpenCV-based image preprocessing for all image inputs before parsing
- MinerU for document text extraction and layout recovery
- A multimodal LLM receiving both MinerU text and the original image on every request, handling Hindi and regional language fields via vision
- Rule-based confidence calibration on top of LLM self-reported scores
- Python validators for all compliance rules: Aadhaar masking, PAN regex, DL expiry, Passport MRZ TD3 checksums
- Guaranteed temp file cleanup enforced at the infrastructure level
- Log redaction covering all Aadhaar number formats
- FastAPI with rate limiting and file size enforcement
- Lightweight webcam-enabled frontend for demo capture

This approach handles real-world scan quality, multilingual content, and all compliance requirements without custom model training, while keeping the architecture clean and each layer's responsibility well-defined.
