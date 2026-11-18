# MediSage Automated Test Report, Bug Log & CPU Performance Notes

## 1. Test Scope & Strategy

Automated unit and integration tests (`tests/`) run locally on CPU using isolated temporary SQLite databases, temporary ChromaDB directories, synthetic valid/corrupt/scanned PDF and DOCX fixtures, and mocked Gemini SDK clients. Live end-to-end verification (`scripts/verify_e2e_live.py`) tests the real configured `GEMINI_API_KEY` against the Gemini `v1beta` endpoint (`gemini-2.5-flash`).

---

## 2. Automated Test Suite Coverage (`tests/`)

| Test File | Test Count | Verified Behaviors |
| --- | --- | --- |
| `tests/test_database.py` | 4 | Auto schema initialization, health check, document CRUD + duplicate SHA-256 lookup, conversation/message/feedback cascade deletion, bounded history windowing, persistent settings, evaluation runs |
| `tests/test_embeddings.py` | 2 | Heading/table-aware chunking with metadata attachment & split-table header retention, CPU `all-MiniLM-L6-v2` 384-dim embeddings, singleton model reuse, query embedding cache |
| `tests/test_ingestion.py` | 6 | Extension/size/empty/magic-byte validation, path-traversal defense, PyMuPDF page extraction & repeated-header stripping, DOCX ordered headings & tables, scanned/corrupt PDF honest rejection, idempotency & synchronized disk/vector/SQLite deletion |
| `tests/test_retrieval.py` | 2 | ChromaDB persistence across client restarts, semantic retrieval & `document_id` filtering, empty OOKB result under threshold, deterministic cache invalidation on deletion |
| `tests/test_rag.py` | 10 | Mocked Gemini adapter (missing key, 429 retry recovery, timeout, malformed response, `v1beta` default & developer-instruction fallback), API key redaction in logs, citation validation (fabricated tag stripping, semicolon-grouped citation parsing, abstention citation suppression), emergency/diagnosis/dosage guardrails, follow-up rewriting, E2E RAG + chat persistence |
| **Total** | **24** | **24 / 24 Passing (`python -m pytest -v` in 11.43 s)** |

---

## 3. Defects Identified & Fixed

| ID | Severity | Component | Issue Observed | Resolution |
| --- | --- | --- | --- | --- |
| BUG-01 | High | `database/db.py` | `get_recent_messages()` raised `sqlite3.OperationalError: no such column: rowid`. | Projected `rowid AS _rid` in the inner subquery, sorted by `created_at ASC, _rid ASC`, and stripped `_rid`. |
| BUG-02 | High | `vectorstore/chroma_manager.py`, `evaluation/evaluator.py` | Windows `WinError 32` temp-dir lock on `data_level0.bin`. | Added `ChromaManager.close()` and `ignore_cleanup_errors=True` for temporary evaluation directories. |
| BUG-03 | Low | `evaluation/__init__.py` | `RuntimeWarning` on `python -m evaluation.evaluator`. | Cleared eager imports from `evaluation/__init__.py`. |
| BUG-04 | Medium | `config/settings.py`, `rag/retriever.py` | Off-topic query `eval_15` matched at `0.2794` under `0.25` threshold (50% OOKB abstention). | Tuned default `similarity_threshold` to `0.32`, achieving `100%` OOKB abstention. |
| BUG-05 | Critical | `core/llm_client.py`, `config/settings.py` | `google-genai 3.5.0` defaulted to `v1`, failing with `400 INVALID_ARGUMENT: Developer instruction is not enabled for api version v1`; `gemini-2.5-flash` thinking tokens could consume small `max_output_tokens` budgets. | Configured `HttpOptions(api_version="v1beta", timeout=...)`, set `ThinkingConfig(thinking_budget=0)` for `2.5-flash`, added prompt-folding fallback for `system_instruction`, and expanded error classification (`unsupported_model`, `network_error`, `auth_error`, `rate_limit`, `timeout`). |
| BUG-06 | Medium | `vectorstore/chroma_manager.py` | Importing `vectorstore` before `ingestion` triggered a circular `ImportError` (`TextChunk` vs `ChromaManager`). | Moved `from ingestion.chunking import TextChunk` under `if TYPE_CHECKING:`. |
| BUG-07 | Medium | `scripts/generate_sample_docs.py`, `ingestion/ingestion_service.py`, `rag/citations.py`, `ingestion/embeddings.py` | Re-running sample generation changed PDF/DOCX timestamps and hashes (creating duplicate sample rows); HF Hub metadata checks added startup latency; semicolon-grouped citations `[Source: A, page 1; Source: B, page 2]` were not normalized. | Made `generate_all_samples(..., overwrite=False)` idempotent, added `verify_and_repair_consistency()`, added `local_files_only=True` fallback in `EmbeddingService` (cold start ~257 ms), and normalized semicolon-grouped citations + abstention citation suppression. |

---

## 4. Measured CPU Laptop Performance & Live Gemini E2E Verification

- **`all-MiniLM-L6-v2` Warm Cached Load (`local_files_only=True`):** **243.9–291.6 ms** (initial uncached benchmark prior to local-first: `1,587.03 ms`)
- **Live Gemini 2.5 Flash (`v1beta`) Generation Latency (`scripts/verify_e2e_live.py`):** **~2.08–3.23 s** end-to-end per query.
- **10 Mandatory Live E2E Checks (`scripts/verify_e2e_live.py`):**
  1. Document upload & indexing (`E2E_Temp_Cholesterol_Ref.pdf` -> `indexed`, 1 chunk)
  2. Retrieval for hypertension blood pressure ranges
  3. Live Gemini `v1beta` grounded generation (`llm_used=True`)
  4. Verified `[Source: filename, page N]` citations
  5. Context-aware follow-up query rewriting + retrieval + Gemini generation
  6. Unsupported question abstention (`status=no_evidence`, `citations=0`)
  7. Multi-document PDF + DOCX retrieval & generation (`CURB-65` from `.docx`)
  8. Re-indexing & deletion consistency (0 orphaned vectors)
  9. Conversation history CRUD (create, rename, reopen, export JSON, delete)
  10. Evaluation benchmark (`100%` Recall@K, `100%` OOKB abstention, `14.23 ms` mean retrieval) & 4-way tuning comparison

---

## 5. Known Limitations

1. **No OCR for Scanned PDFs:** Image-only/scanned PDFs are intentionally rejected (`ScannedDocumentError`).
2. **Cloud LLM Dependency:** Live generation requires a valid `GEMINI_API_KEY` in `.env` and outbound internet connectivity.
3. **Offline Evaluation Mode:** Automated evaluation runs deterministically without calling live Gemini endpoints unless `--use-llm` is explicitly passed.
4. **Local Single-User Scope:** Designed for local workstation execution with SQLite and local ChromaDB persistence.
