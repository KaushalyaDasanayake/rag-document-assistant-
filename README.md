# RAG Document Assistant (`ragassistant`)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-assistant-pro.streamlit.app/)

Enterprise retrieval-augmented generation (RAG) assistant featuring recursive chunking, GPU-accelerated embeddings, vector retrieval, cross-encoder reranking, grounded answer generation with verifiable citations, strict refusal mechanisms when evidence is absent, and stage-level latency/token-cost observability.

**Live Demo**: [https://doc-assistant-pro.streamlit.app/](https://doc-assistant-pro.streamlit.app/)

---

## Architecture Overview
- **Ingestion**: PDF, Markdown, and TXT parsing with page-level metadata tracking.
- **Chunking**: Recursive window chunking with configurable overlap and metadata preservation.
- **Embeddings**: `BAAI/bge-base-en-v1.5` (768-dim) accelerated via CUDA on NVIDIA RTX GPUs, with automatic CPU fallback.
- **Vector Index**: Persistent ChromaDB store with document deletion and retention lifecycle controls.
- **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring Top-K candidates.
- **Generation**: Strict context-grounded synthesis with mandatory citations and out-of-domain refusal.
- **Observability**: Request-level tracing measuring retrieval latency, generation latency, token volume, and estimated cost per 1k queries.
- **Interface**: Interactive Streamlit conversation studio with collapsible source citations.

---

## Evaluation Benchmark (25-Query Golden Dataset)

Evaluated on `eval/golden_qa.jsonl` containing factual and adversarial queries:

| Metric | Baseline (Bi-Encoder) | Reranked (Cross-Encoder) |
|---|---|---|
| **Recall@3** | 100.00% | 100.00% |
| **Recall@5** | 100.00% | 100.00% |
| **MRR (Mean Reciprocal Rank)** | 0.9417 | **0.9500** |
| **Adversarial Refusal Rate** | 60.00% | 60.00% |

---

## Quickstart

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Run Tests & Linter
```bash
pytest -v
ruff check .
```

### 3. Run Retrieval Evaluation
```bash
python scripts/run_eval.py
```

### 4. Run API Service
```bash
uvicorn src.ragassistant.api.app:app --host 0.0.0.0 --port 8000
```

### 5. Launch Streamlit Studio
```bash
streamlit run streamlit_app.py
```

---

## Milestones Delivered
- [x] **M0: Project Setup & Golden Evaluation Dataset**
- [x] **M1: Ingestion & Vector Index Pipeline**
- [x] **M2: Retrieval Engine with Reranking & Evaluation (w18/19)**
- [x] **M3: Grounded Generation, Citations & Refusal Mechanism (w17)**
- [x] **M4: Tracing, Observability & Token Cost Accounting (w27/w30)**
- [x] **M5: Streamlit Web UI & Live Cloud Deployment**
