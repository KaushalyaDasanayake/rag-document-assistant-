# Live Cloud Deployment Guide: Streamlit Community Cloud

This guide explains how to deploy the **Enterprise RAG Document Assistant** to **Streamlit Community Cloud** (100% free, no credit card required) to provide a permanent live interactive portfolio demo for recruiters and Fiverr clients.

---

## 1. Prerequisites

1. Your repository pushed to GitHub:
   - Repository: `https://github.com/KaushalyaDasanayake/rag-document-assistant-`
   - Branch: `main`
2. A free [Streamlit Community Cloud](https://share.streamlit.io/) account (sign in using your GitHub account).
3. *(Optional)* A free [Groq API Key](https://console.groq.com/keys) (Groq offers ultra-fast free tier LPU inference with `llama-3.1-8b-instant`). If omitted, the app automatically runs in deterministic fallback mode with zero API keys required.

---

## 2. Step-by-Step Deployment

### Step 1: Open Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/).
2. Click **"New app"** in the top right corner.

### Step 2: Configure Repository Settings
Fill in the deployment form:
- **Repository**: `KaushalyaDasanayake/rag-document-assistant-`
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`
- **App URL**: Choose a custom subdomain if available (e.g. `enterprise-rag-assistant.streamlit.app`).

### Step 3: Configure Environment Secrets (Optional)
Click **"Advanced settings..."** at the bottom of the form:
- Python version: Select **3.11** or **3.10**
- Under **Secrets**, add your API keys:
  ```toml
  GROQ_API_KEY = "gsk_your_groq_api_key_here"
  # Optional:
  # OPENAI_API_KEY = "sk-your-openai-api-key"
  ```
- Click **Save**.

### Step 4: Launch Deployment
Click **"Deploy!"**.
Streamlit will clone the repository, install dependencies from `requirements.txt`, load the local CPU embeddings, and launch your application.

Your permanent public URL will be ready at:
`https://<your-subdomain>.streamlit.app`

---

## 3. Live Demo Features for Clients and Recruiters

When sharing your link with Fiverr prospective clients or engineering hiring managers, highlight these enterprise capabilities:

1. **Document Ingestion & File Upload**:
   - Upload any custom corporate PDF, Markdown policy, or plain text memo in the sidebar.
   - The system automatically slices the document into recursive semantic chunks with page provenance tracking.

2. **Data Retention & Lifecycle Management (w30 Compliance)**:
   - View active documents in the repository.
   - Click the 🗑️ icon next to any document to delete all its index entries under enterprise data retention rules.
   - Execute a full compliance wipe with "Wipe Index".

3. **Verbatim Provenance Citations**:
   - Ask any factual question or click one of the quick prompt chips.
   - Expand the **"Source Citations"** accordion to view the exact document name, page number, and verbatim quote verified against the source text.

4. **Hallucination Prevention & Refusal Mechanism**:
   - Ask an unanswerable or adversarial question (e.g., *"What is the gym reimbursement amount in Tokyo?"*).
   - The assistant flags the query with `🛡️ REFUSED` and returns a standard refusal rather than hallucinating false claims.

5. **Token Cost Accounting & Latency Telemetry**:
   - Every answer displays a live telemetry badge showing:
     - Total Latency (with Retrieval vs Generation breakdown)
     - Token Accounting (Input tokens vs Output tokens)
     - Estimated Financial Cost and extrapolated cost per 1,000 queries.
