# RAG Document Assistant (`ragassistant`)

Enterprise retrieval-augmented generation (RAG) assistant featuring chunking, local GPU-accelerated embeddings, vector retrieval, cross-field reranking, grounded answer generation with verifiable citations, strict refusal mechanisms when evidence is absent, and stage-level latency/token-cost observability.

---

## Architecture Overview
- **Ingestion**: PDF, Markdown, and TXT parsing with page-level metadata tracking.
- **Chunking**: Recursive window chunking with configurable overlap.
- **Embeddings**: `BAAI/bge-base-en-v1.5` (768-dim) accelerated via CUDA on NVIDIA RTX GPUs, with automatic CPU fallback.
- **Vector Index**: In-process ChromaDB with metadata filtering.
- **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring Top-K candidates.
- **Generation**: Strict context-grounded synthesis with mandatory citations and out-of-domain refusal.
- **Evaluation**: Quantitative benchmarks on a frozen 25-query golden dataset measuring Recall@K, MRR, and Faithfulness.
- **Interface**: Interactive Streamlit conversation studio with collapsible source citations.

---

## Milestones & Delivery Status
- [ ] **M0: Project Setup & Golden Evaluation Dataset**
- [ ] **M1: Ingestion & Vector Index Pipeline**
- [ ] **M2: Retrieval Engine with Reranking & Evaluation (w18/19)**
- [ ] **M3: Grounded Generation, Citations & Refusal Mechanism (w17)**
- [ ] **M4: Tracing, Observability & Token Cost Accounting (w27/w30)**
- [ ] **M5: Streamlit Web UI & Live Cloud Deployment**
