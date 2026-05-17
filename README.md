# ISA Audit Intelligence

A RAG-powered audit intelligence system that maps ERP control failures and SHAP anomaly features to the relevant ISA (International Standards on Auditing) paragraphs and generates plain-language audit findings via a local LLM.

## How it works

1. **Indexing** — ISA standard PDFs are chunked by paragraph and embedded into a local ChromaDB vector store (runs once on first launch).
2. **Retrieval** — Given a SHAP anomaly feature and an ERP control failure, the system retrieves the most relevant ISA paragraph for each using semantic similarity search.
3. **Generation** — A local Ollama LLM synthesises both paragraphs into a concise, standards-grounded audit finding.
4. **Display** — The Streamlit UI shows both retrieved ISA paragraphs alongside the generated narrative.

## Architecture

```
src/
├── app.py          # Streamlit entry point
├── config.py       # Constants & dropdown options
├── models.py       # Data classes & TypedDicts
├── indexer.py      # PDF chunking & corpus indexing
├── vector_store.py # ChromaDB vector index (ISAVectorIndex)
├── retrieval.py    # RAG retrieval node (ISARAGRetrieval)
├── llm.py          # Local LLM via Ollama (LocalLLM)
├── ui.py           # Streamlit UI components
├── pipeline.py     # LangGraph orchestrator
└── Data/           # ISA PDF files (not committed — see Setup)
```

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| [Ollama](https://ollama.com) | latest | Local LLM server |
| `llama3.1:8b-instruct-q4_K_M` | — | LLM model (pulled via Ollama) |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/georgeesamy/ISA-Audit-Intelligence.git
cd ISA-Audit-Intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add ISA PDF files
mkdir -p src/Data
# Copy your ISA standard PDFs into src/Data/:
#   ISA 240.pdf, ISA 265.PDF, ISA 315.pdf, ISA 330.pdf
#   ISA530 (amended) - Audit Sampling.pdf

# 5. Start Ollama and pull the model
ollama serve                                        # in a separate terminal
ollama pull llama3.1:8b-instruct-q4_K_M

# 6. Run the app
streamlit run src/app.py
```

The ISA corpus is indexed automatically on first launch (takes ~1–2 minutes). Subsequent launches load from the cached ChromaDB store.

## Usage

1. Open the app in your browser (default: `http://localhost:8501`).
2. Select an **ERP Control Attribute** (e.g. SoD Violation, Management Override).
3. Select a **SHAP Feature Column** (e.g. `posting_hour`, `amount_ratio`).
4. Click **Analyse & Generate Narrative**.
5. The app displays:
   - The retrieved ISA paragraph for the ERP control failure
   - The retrieved ISA paragraph for the SHAP anomaly feature
   - An AI-generated audit finding grounded in both paragraphs

## Tech stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[ChromaDB](https://www.trychroma.com)** — Local vector database
- **[sentence-transformers](https://www.sbert.net)** — `BAAI/bge-large-en-v1.5` embeddings
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Pipeline orchestration
- **[pdfplumber](https://github.com/jsvine/pdfplumber)** — PDF text extraction
- **[Ollama](https://ollama.com)** — Local LLM inference

## Notes

- `chroma_db/` and `src/Data/` are gitignored. Both are generated/provided locally.
- Ollama must be running on `localhost:11434` before clicking Analyse.
- The LLM model can be changed in `src/config.py` (`LLM_MODEL`).
