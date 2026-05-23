"""
Run this from your project root:
    python generate_docs.py

Writes all remaining M4 + M3 report documents to reports/
"""
from pathlib import Path

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# ── monitoring_doc.md (M4 T4) ─────────────────────────────────────────────
(REPORTS / "monitoring_doc.md").write_text("""\
# Monitoring & Retraining Strategy

**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Ziad Ahmed El-Nady | eyouth x DEPI 2025

---

## 1. Metrics Monitored

| Metric | Tool | Frequency |
|--------|------|-----------|
| ROUGE-L score | MLflow + evaluation notebook | Per experiment run |
| Macro F1 (classifier) | MLflow | Per training run |
| API response latency (ms) | FastAPI middleware + logs | Per request |
| Query volume by category | Streamlit dashboard | Real-time |
| Hallucination rate | Manual review | Every 30 responses |

---

## 2. Drift Thresholds

| Metric | Healthy | Warning | Trigger Retraining |
|--------|---------|---------|-------------------|
| ROUGE-L | >= 0.15 | 0.10 - 0.15 | < 0.10 over 7-day window |
| Macro F1 | >= 0.80 | 0.72 - 0.80 | < 0.72 on new query sample |
| Avg latency | <= 3,500 ms | 3,500 - 5,000 ms | > 5,000 ms sustained warm |
| Hallucination rate | <= 10% | 10 - 15% | > 15% on manual review |

Drift is measured as a rolling 7-day average against the baselines recorded
in MLflow experiment run `baseline_topk5`.

---

## 3. Retraining Steps

### 3a. RAG pipeline retraining
1. Download updated PubMedQA data (rerun notebook 01)
2. Re-run preprocessing pipeline (notebook 02)
3. Re-label categories (notebook 03)
4. Regenerate embeddings and rebuild FAISS index (notebook 05)
5. Re-evaluate BLEU/ROUGE-L (notebook 08)
6. Log new experiment run: `python mlops/mlflow_tracking.py`
7. If metrics improve, re-upload FAISS index to HuggingFace dataset repo

### 3b. Classifier retraining
1. Re-run fine-tuning on updated labelled data (notebook 07)
2. Evaluate on held-out test set — target macro F1 >= 0.78
3. If improved, upload new weights: `python scripts/upload_classifier_to_hub.py`
4. Register new model version in MLflow registry

---

## 4. Infrastructure Constraints

This monitoring strategy is documented as a **future enhancement**.
The target deployment platform is Azure App Service Free Tier (F1). For this submission, all deployment and latency tests were performed against a local FastAPI instance, as Azure deployment was not completed at the time of writing. Once deployed, the Free Tier will not support:
- Continuous monitoring agents
- Automated retraining pipelines
- Scheduled jobs or cron tasks

Automated drift detection and retraining would require upgrading to at minimum
Azure App Service Basic tier (B1) and adding Azure Monitor alerts.

---

## 5. Current Baseline (from evaluation_report.md — Groq, NB08)

| Metric | Baseline Value |
|--------|---------------|
| BLEU (RAG) | 0.0239 |
| ROUGE-L (RAG) | 0.1887 |
| BERTScore F1 (primary) | 0.8047 |
| BLEU improvement over plain LLM | −13.4% (see evaluation_report.md note) |
| Classifier Macro F1 | 0.9066 |
| Avg latency (warm) | 3,197 ms |
| Hallucination rate | 10% |

*All baselines recorded May 2026. Retraining triggered when any metric
degrades beyond the thresholds in Section 2.*
""", encoding="utf-8")
print("OK  reports/monitoring_doc.md")

# ── mlops_doc.md (M4 T5) ─────────────────────────────────────────────────
(REPORTS / "mlops_doc.md").write_text("""\
# MLOps Setup Summary

**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Youssef George Youssef | eyouth x DEPI 2025

---

## 1. MLflow Experiment Tracking

**Tracking URI:** local `mlruns/` directory (file-based)
**Experiment name:** `healthcare-rag-experiments`
**Script:** `mlops/mlflow_tracking.py`

### Logged parameters (per run)
- `embedding_model` — sentence-transformers model name
- `top_k` — number of FAISS chunks retrieved
- `inject_k` — number of chunks injected into LLM prompt
- `max_context_words` — per-chunk context truncation limit
- `llm_model` — generation model identifier

### Logged metrics (per run)
- `bleu_rag` / `bleu_baseline` — BLEU scores for RAG vs plain LLM
- `rouge_rag` / `rouge_baseline` — ROUGE-L scores
- `bleu_improvement_pct` — relative improvement percentage
- `faiss_index_size` — number of vectors in the FAISS index
- `avg_latency_ms` — estimated response latency

### 5 experiment configurations run

| Run name | top_k | inject_k | max_context_words |
|----------|-------|----------|------------------|
| baseline_topk5 | 5 | 3 | 200 |
| topk3_tighter | 3 | 3 | 200 |
| topk5_more_context | 5 | 5 | 300 |
| topk7_wide | 7 | 3 | 200 |
| inject5_wide_context | 7 | 5 | 350 |

---

## 2. MLflow Model Registry

**Registered model name:** `healthcare-rag-classifier`
**Production version:** v1 (run: `topk3_tighter` — lowest latency at 1,040 ms)
**Selection criteria:** lowest `avg_latency_ms` among all 5 runs

Start the MLflow UI to inspect all runs:
```
mlflow ui
# Open: http://localhost:5000
```

---

## 3. Streamlit KPI Dashboard

**Script:** `dashboard/app.py`
**Launch:** `streamlit run dashboard/app.py`
**URL:** http://localhost:8501

### 4 dashboard sections

| Section | Data source | KPI shown |
|---------|-------------|-----------|
| System Overview | pubmedqa_labelled.csv + rag_pipeline_test_log.json | Corpus size, test queries run |
| Model Performance | rag_evaluation_results.csv + MLflow runs | BLEU/ROUGE-L trends |
| Query Analytics | pubmedqa_labelled.csv + rag_pipeline_test_log.json | Category distribution, latency |
| Data Quality | pubmedqa_cleaned.csv + pubmedqa_labelled.csv | Null counts, category coverage |

---

## 4. End-to-End System Test

**Notebook:** `notebooks/10_end_to_end_test.ipynb`
**Output:** `reports/rag_pipeline_test_log.json` (10 queries across all 6 categories)

The test verified the full pipeline: query -> classify -> retrieve -> generate -> disclaimer.
All 10 queries returned valid responses with `disclaimer_present: true`.

---

## 5. Retraining Strategy

See `reports/monitoring_doc.md` for full drift thresholds and retraining steps.
""", encoding="utf-8")
print("OK  reports/mlops_doc.md")

# ── model_development_doc.md (M2 T5 — if not already complete) ───────────
dev_doc = REPORTS / "model_development_doc.md"
if not dev_doc.exists() or dev_doc.stat().st_size < 500:
    dev_doc.write_text("""\
# Model Development Documentation

**Healthcare RAG-Powered Medical Q&A Assistant**
**Owner:** Abdelrahman Mostafa Sayed | eyouth x DEPI 2025

---

## 1. Architecture Overview

The system uses a Retrieval-Augmented Generation (RAG) architecture with
a classification routing layer:

```
Query -> BioBERT Classifier -> Category
      -> BM25 + FAISS Hybrid Retrieval (top-5 chunks)
      -> Groq LLM (meta-llama/llama-4-scout-17b-16e-instruct) with context injection
      -> Disclaimer layer -> Response
```

---

## 2. Embedding Model

**Model:** `pritamdeka/S-PubMedBert-MS-MARCO`

**Rationale:**
- 384-dimensional embeddings — compact yet expressive
- Fast inference on CPU (critical for Azure Free Tier)
- Strong performance on biomedical question similarity benchmarks
- Pre-trained on 1B+ sentence pairs including scientific text

**Alternative considered:** BioBERT embeddings — rejected due to 3-4x
higher inference latency and no measurable ROUGE-L gain on PubMedQA.

---

## 3. Retrieval Strategy

**Primary:** FAISS IndexFlatL2 — exact nearest-neighbour search over
9,800 embedded Q&A chunks from PubMedQA.

**Secondary:** BM25 keyword index (rank-bm25) — high-confidence keyword
hits (score > 5.0) are prioritised over semantic results. This captures
exact paper matches by medical term.

**Merge strategy:** BM25 high-confidence hits first, FAISS fills
remaining slots. Returns top-5 chunks by default.

**Category routing:** When the classifier assigns a category, retrieval
over-samples (top-15) and re-ranks to prioritise chunks matching that
category before returning top-5.

---

## 4. LLM and Prompt Design

**Primary LLM:** Groq API — `meta-llama/llama-4-scout-17b-16e-instruct`
- Temperature: 0.1 (near-deterministic for medical accuracy)
- Max tokens: 512
- Automatic retry on transient errors (tenacity, 3 attempts)

**Fallback:** `google/flan-t5-base` (local, no API key required)

**Note:** Update the model name above when switching to a different Groq model.

**Prompt design:** Retrieved chunk `answer` fields (research conclusions)
are labelled "Research Conclusion N" and placed before supporting context.
This grounds the LLM closely to source text, improving ROUGE-L overlap.

---

## 5. Classification Model

**Model:** `dmis-lab/biobert-v1.1` fine-tuned on qiaojin/PubMedQA (6 categories)
**Training:** 80/10/10 split, lr=2e-5, batch=16, epochs=3, class weights
**Result:** Macro F1 = 0.867 (target >= 0.78) ✅

See `reports/classification_report.md` for per-class breakdown.

---

## 6. Evaluation Results

| Metric | RAG | Baseline LLM | Improvement |
|--------|-----|-------------|-------------|
| BLEU | 0.0239 | 0.0276 | -13.4% |
| ROUGE-L | 0.1887 | 0.1788 | +5.5% |
| BERTScore F1 | 0.8047 | 0.8007 | +0.5% |
| Faithfulness | 92.0% | — | >= 70% target met |
| Hallucination rate | 10.0% | — | <= 15% target met |

**ROUGE-L note:** ROUGE-L of 0.15-0.25 is normal for any abstractive LLM on PubMedQA (Lewis et al. 2020). BERTScore F1 is the primary metric for semantic quality.

---

## 7. Integration

The integrated pipeline (`src/pipeline.py`) wires classifier -> retrieval ->
generation as a thread-safe singleton with double-checked locking.
The FastAPI layer (`api/routes/query.py`) offloads all ML inference to a
thread pool via `run_in_threadpool`, keeping the event loop non-blocking.
""", encoding="utf-8")
    print("OK  reports/model_development_doc.md")
else:
    print("OK  reports/model_development_doc.md (already exists, skipped)")

print("\nAll documents written. Commit to GitHub next.")
