"""Comparative retrieval tuning experiments comparing chunk sizes, overlap, top_k, and thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import run_evaluation
from utils.helpers import utc_now_iso

TUNING_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "tuning_results.json"

TUNING_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "Baseline (Large Chunk / Low Threshold / No Dedupe)",
        "chunk_size": 1000,
        "chunk_overlap": 50,
        "top_k": 2,
        "similarity_threshold": 0.10,
        "deduplicate": False,
        "enforce_doc_diversity": False,
        "use_query_rewriter": False,
    },
    {
        "name": "Fine-Grained Sub-Page Chunks (240 tokens)",
        "chunk_size": 240,
        "chunk_overlap": 45,
        "top_k": 4,
        "similarity_threshold": 0.32,
        "deduplicate": True,
        "enforce_doc_diversity": True,
        "use_query_rewriter": True,
    },
    {
        "name": "High Top-K / Permissive Threshold (600 / k=6 / thr=0.25)",
        "chunk_size": 600,
        "chunk_overlap": 100,
        "top_k": 6,
        "similarity_threshold": 0.25,
        "deduplicate": True,
        "enforce_doc_diversity": True,
        "use_query_rewriter": True,
    },
    {
        "name": "Tuned Default (600 tokens / k=4 / thr=0.32)",
        "chunk_size": 600,
        "chunk_overlap": 100,
        "top_k": 4,
        "similarity_threshold": 0.32,
        "deduplicate": True,
        "enforce_doc_diversity": True,
        "use_query_rewriter": True,
    },
]


def run_tuning_comparison(output_path: Path | None = None) -> dict[str, Any]:
    """Execute evaluation across baseline and tuned configurations and save report."""
    runs: list[dict[str, Any]] = []
    for cfg in TUNING_EXPERIMENTS:
        report = run_evaluation(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            top_k=cfg["top_k"],
            similarity_threshold=cfg["similarity_threshold"],
            deduplicate=cfg["deduplicate"],
            enforce_doc_diversity=cfg["enforce_doc_diversity"],
            use_query_rewriter=cfg["use_query_rewriter"],
            use_llm=False,
        )
        runs.append(
            {
                "experiment_name": cfg["name"],
                "config": report["config"],
                "total_chunks": report["corpus"]["total_chunks"],
                "summary": report["summary"],
            }
        )

    payload = {
        "generated_at": utc_now_iso(),
        "experiments": runs,
    }
    target = output_path or TUNING_OUTPUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    res = run_tuning_comparison()
    for exp in res["experiments"]:
        print(f"=== {exp['experiment_name']} (chunks={exp['total_chunks']}) ===")
        print(json.dumps(exp["summary"], indent=2))
