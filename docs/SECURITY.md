# MediSage Security, Privacy & Input Validation Checklist

## 1. Credential & Secrets Protection
- [x] API keys (`GEMINI_API_KEY`) are read exclusively from `.env` or environment variables via `python-dotenv` and `config/settings.py`.
- [x] `.env`, `.env.*` (except `.env.example`), `data/uploads/*`, `data/chroma_db/*`, `*.db`, and `logs/` are excluded from Git in `.gitignore`.
- [x] `.env.example` contains placeholder values only (`your_gemini_api_key_here`).
- [x] `core/logging_utils.py` applies `SensitiveDataFilter` to redact `AIza...` keys, `Bearer` tokens, and `GEMINI_API_KEY=...` assignments from all stream and file logs.

## 2. Upload Validation & Path Traversal Defense
- [x] `utils/validators.py` restricts extensions to `.pdf`, `.docx`, `.txt`, and `.md`.
- [x] Empty files (`0 bytes`) and files exceeding `MAX_UPLOAD_SIZE_MB` (default `20 MB`) are rejected prior to disk storage.
- [x] Container magic-byte inspection rejects spoofed or corrupt files (`%PDF-` header check for PDF; `PK\x03\x04` + internal `[Content_Types].xml` / `word/document.xml` check for DOCX; null-byte & UTF-8 check for TXT/MD).
- [x] Original user-supplied filenames are sanitized to basename only and stored exclusively as SQLite/Chroma metadata; on-disk storage uses generated UUID-prefixed filenames (`doc_<uuid>_<hash8>.<ext>`).
- [x] `build_safe_storage_path()` and `resolve_safe_stored_file()` verify `Path.resolve().is_relative_to(upload_dir.resolve())` to block directory traversal (`../`).

## 3. Safe Document Parsing & Database Access
- [x] Encrypted/password-protected PDFs, 0-page PDFs, corrupt archives, and scanned/image-only PDFs raise explicit exceptions without indexing empty text or claiming OCR.
- [x] All SQLite operations in `database/db.py` use parameterized `?` placeholders (`PRAGMA foreign_keys = ON`)—zero string-interpolated SQL.
- [x] `utils/helpers.py::sanitize_error_message()` strips Windows/Unix local filesystem paths and credentials before rendering errors in the Streamlit UI; raw Python stack traces are never exposed to end users.
- [x] Application logs avoid logging full medical documents or complete user queries.

## 4. Deletion & Data Lifecycle Guarantees
- [x] **Delete Document**: `IngestionService.delete_document_completely()` synchronously removes:
  1. All matching vector records in ChromaDB (`document_id`)
  2. The stored file inside `data/uploads/`
  3. The metadata row in SQLite `documents`
  4. Retrieval cache entries (`invalidate_on_index_change()`)
- [x] **Delete Conversation**: `DatabaseManager.delete_conversation()` cascades via `ON DELETE CASCADE` to remove all child rows in `messages` and `feedback`.
- [x] **Delete Feedback**: Supported per feedback record (`delete_feedback`) and via bulk purge in **Settings & Security**.

## 5. Authorization & PHI Governance Expectations
- MediSage is designed strictly for **permitted educational and clinical reference documents** (e.g., public guidelines, textbook chapters, de-identified protocols).
- **Do NOT** upload identifiable patient health information (PHI / PII) unless operating in an authorized, encrypted, access-controlled environment meeting organizational compliance requirements.
- Because MediSage sends retrieved text passages to a configured cloud LLM API (Google Gemini) during chat generation, operators must ensure no unauthorized sensitive patient records are indexed when external cloud APIs are enabled.
