"""RAG Evaluation & Retrieval Tuning Dashboard page with KPI cards and Altair visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from evaluation.evaluator import DEFAULT_OUTPUT_PATH, run_evaluation
from evaluation.tuning import TUNING_OUTPUT_PATH, run_tuning_comparison
from ui.styles import render_empty_state, render_kpi_card, render_page_header, status_badge


def render_evaluation(services: dict[str, Any]) -> None:
    """Render evaluation metrics, case-level results, and tuning comparison experiments."""
    db = services["db"]
    settings = services["settings"]

    render_page_header(
        "RAG Evaluation & Retrieval Tuning",
        "Measure retrieval accuracy, citation integrity, safety compliance, and compare chunk/threshold configurations in an isolated index.",
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.subheader("16-Question RAG Benchmark")
            st.caption(
                "Runs against `evaluation/eval_dataset.json` inside an isolated temporary ChromaDB index, keeping user conversations separate."
            )
            use_live_llm = st.toggle(
                "Include live Gemini API generation (uses configured API key)",
                value=False,
            )
            if st.button("Run Evaluation Benchmark", type="primary", use_container_width=True):
                top_k = int(db.get_setting("retrieval_top_k", settings.retrieval_top_k))
                thresh = float(
                    db.get_setting("similarity_threshold", settings.similarity_threshold)
                )
                chunk_size = int(
                    db.get_setting("chunk_size_tokens", settings.chunk_size_tokens)
                )
                chunk_overlap = int(
                    db.get_setting("chunk_overlap_tokens", settings.chunk_overlap_tokens)
                )
                with st.spinner("Running 16-question RAG benchmark on CPU..."):
                    report = run_evaluation(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                        similarity_threshold=thresh,
                        use_llm=use_live_llm,
                    )
                    db.save_evaluation_run(
                        run_label=f"UI Benchmark ({report['mode']})",
                        config=report["config"],
                        metrics=report["summary"],
                    )
                st.success("Evaluation benchmark completed and saved.")
                st.rerun()

    with c2:
        with st.container(border=True):
            st.subheader("Retrieval Parameter Tuning")
            st.caption(
                "Compares chunk sizes (1000 vs 240 vs 600 tokens), Top-K values (2, 4, 6), and similarity thresholds (0.10, 0.25, 0.32)."
            )
            st.markdown("<div style='height: 1.45rem'></div>", unsafe_allow_html=True)
            if st.button("Run Retrieval Tuning Comparison", use_container_width=True):
                with st.spinner("Executing multi-configuration retrieval tuning suite..."):
                    run_tuning_comparison()
                st.success("Tuning experiment comparison saved.")
                st.rerun()

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    eval_path = Path(DEFAULT_OUTPUT_PATH)
    if eval_path.exists():
        report = json.loads(eval_path.read_text(encoding="utf-8"))
        summ = report.get("summary", {})
        mode_label = report.get("mode", "deterministic_retrieval_and_safety")
        mode_desc = (
            "Measured offline retrieval, safety, and lexical grounding proxy"
            if mode_label == "deterministic_retrieval_and_safety"
            else "Measured end-to-end with live Gemini generation"
        )
        corpus_info = report.get("corpus", {})

        st.subheader("Latest Benchmark Summary")
        st.markdown(
            status_badge(mode_label, "lavender")
            + status_badge(f"{corpus_info.get('documents', 0)} docs / {corpus_info.get('total_chunks', 0)} chunks", "info")
            + status_badge(f"Generated {report.get('generated_at', '')[:19]} UTC", "neutral"),
            unsafe_allow_html=True,
        )
        st.caption(mode_desc)

        r1_1, r1_2, r1_3, r1_4 = st.columns(4)
        with r1_1:
            render_kpi_card("Page Recall@K", f"{summ.get('recall_at_k', 0)*100:.1f}%", "Expected page retrieved", icon="🎯", accent="teal")
        with r1_2:
            render_kpi_card("Source Hit Rate", f"{summ.get('source_hit_rate', 0)*100:.1f}%", "Expected document hit", icon="📄", accent="teal")
        with r1_3:
            render_kpi_card("Mean Reciprocal Rank", f"{summ.get('mrr', 0):.3f}", "First relevant chunk rank", icon="🏅", accent="lavender")
        with r1_4:
            render_kpi_card("Context Relevance", f"{summ.get('context_relevance', 0)*100:.1f}%", "Expected clinical facts", icon="🔍", accent="green")

        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)

        r2_1, r2_2, r2_3, r2_4 = st.columns(4)
        with r2_1:
            render_kpi_card("Citation Correctness", f"{summ.get('citation_correctness', 0)*100:.1f}%", "Grounded [Source, page]", icon="📌", accent="green")
        with r2_2:
            render_kpi_card("Faithfulness Proxy", f"{summ.get('answer_faithfulness', 0)*100:.1f}%", "Lexical token grounding", icon="🛡️", accent="lavender")
        with r2_3:
            render_kpi_card("OOKB Abstention", f"{summ.get('unsupported_abstention_rate', 0)*100:.1f}%", "Out-of-KB refusal rate", icon="🚫", accent="coral")
        with r2_4:
            render_kpi_card("Mean Latency", f"{summ.get('mean_retrieval_latency_ms', 0):.1f} ms", f"P95: {summ.get('p95_retrieval_latency_ms', 0):.1f} ms", icon="⚡", accent="amber")

        st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Per-Question Evaluation Breakdown")
            cases = report.get("cases", [])
            categories = ["All"] + sorted({c["category"] for c in cases})
            cat_filter = st.selectbox("Filter by question category", categories)
            rows = [
                {
                    "ID": c["id"],
                    "Category": c["category"],
                    "Question": c["question"],
                    "Expected Behavior": c["expected_behavior"],
                    "Retrieval Status": c["retrieval_status"],
                    "Top Sim": c["top_similarity"],
                    "Source Hit": c["source_hit"],
                    "Page Recall": c["page_recall"],
                    "Context Rel": c["context_relevance"],
                    "Citation OK": c["citation_correctness"],
                    "Latency (ms)": c["retrieval_latency_ms"],
                }
                for c in cases
                if cat_filter == "All" or c["category"] == cat_filter
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        render_empty_state(
            "📊",
            "No Evaluation Report Generated Yet",
            "Click Run Evaluation Benchmark above to measure retrieval recall, citation accuracy, and safety compliance.",
        )

    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
    tuning_path = Path(TUNING_OUTPUT_PATH)
    with st.container(border=True):
        st.subheader("Before vs. After Retrieval Tuning Comparison")
        if tuning_path.exists():
            t_data = json.loads(tuning_path.read_text(encoding="utf-8"))
            exp_rows = []
            for exp in t_data.get("experiments", []):
                cfg = exp["config"]
                s = exp["summary"]
                exp_rows.append(
                    {
                        "Experiment": exp["experiment_name"],
                        "Chunk Size": cfg["chunk_size_tokens"],
                        "Overlap": cfg["chunk_overlap_tokens"],
                        "Top-K": cfg["top_k"],
                        "Threshold": cfg["similarity_threshold"],
                        "Total Chunks": exp["total_chunks"],
                        "Page Recall@K": round(s["recall_at_k"], 3),
                        "MRR": round(s["mrr"], 3),
                        "Context Relevance": round(s["context_relevance"], 3),
                        "OOKB Abstention": round(s["unsupported_abstention_rate"], 3),
                        "Latency (ms)": s["mean_retrieval_latency_ms"],
                    }
                )
            df_exp = pd.DataFrame(exp_rows)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

            melted = df_exp.melt(
                id_vars=["Experiment"],
                value_vars=["Page Recall@K", "MRR", "Context Relevance", "OOKB Abstention"],
                var_name="Metric",
                value_name="Score",
            )
            chart = (
                alt.Chart(melted)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=-12, labelLimit=180)),
                    xOffset="Metric:N",
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1.05]), title="Score (0–1)"),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["Page Recall@K", "MRR", "Context Relevance", "OOKB Abstention"],
                            range=["#269C96", "#8B82C8", "#78B99A", "#F28B82"],
                        ),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    tooltip=["Experiment", "Metric", "Score"],
                )
                .properties(height=320)
                .configure_view(strokeWidth=0)
                .configure_axis(gridColor="#EBF1F5", labelColor="#26334A", titleColor="#748196")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Click **Run Retrieval Tuning Comparison** above to generate comparative experiment metrics.")
