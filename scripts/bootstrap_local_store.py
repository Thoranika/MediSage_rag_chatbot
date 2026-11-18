"""Bootstrap local persistent SQLite and ChromaDB with non-sensitive sample medical references."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ensure_directories, get_settings
from database.db import DatabaseManager
from evaluation.evaluator import DEFAULT_OUTPUT_PATH
from ingestion.embeddings import EmbeddingService
from ingestion.ingestion_service import IngestionService
from scripts.generate_sample_docs import generate_all_samples
from vectorstore.chroma_manager import ChromaManager


def main() -> None:
    settings = ensure_directories(get_settings())
    db = DatabaseManager(settings.sqlite_db_path)
    chroma = ChromaManager(settings.chroma_persist_dir, settings.chroma_collection_name)
    embeddings = EmbeddingService(
        model_name=settings.embedding_model_name, device=settings.embedding_device
    )
    ingestion = IngestionService(
        db=db, chroma=chroma, embeddings=embeddings, settings=settings
    )

    sample_paths = generate_all_samples(settings.samples_dir)
    for sp in sample_paths:
        outcome = ingestion.ingest_file_path(sp, source_origin="sample")
        print(f"{sp.name}: {outcome.status} -> {outcome.message}")

    eval_path = Path(DEFAULT_OUTPUT_PATH)
    if eval_path.exists() and not db.list_evaluation_runs(limit=1):
        report = json.loads(eval_path.read_text(encoding="utf-8"))
        db.save_evaluation_run(
            run_label=f"Baseline Verified ({report.get('mode', 'deterministic')})",
            config=report.get("config", {}),
            metrics=report.get("summary", {}),
        )
        print("Saved baseline evaluation run into SQLite evaluation_runs table.")

    print("Knowledge base stats:", json.dumps(db.get_knowledge_stats(), indent=2))
    chroma.close()


if __name__ == "__main__":
    main()
