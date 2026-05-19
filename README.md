# An Explainable Artificial Intelligence Based Conceptual Model for Financial Auditing to Enhance Audit Transparency and Quality

**Bachelor Thesis** — German University in Cairo (GUC)  
**Faculty:** Management Technology (MNGT) — Department of Business Informatics  
**Author:** George Samy Nabih (ID: 58-21333)  
**Supervisor:** Dr. Gamal Kassem  
**Submission Date:** May 2026

---

## Project Overview

Financial auditing faces growing challenges as data volumes increase and traditional methods struggle to keep up. While AI has shown strong results in fraud detection and anomaly identification, most AI systems used in auditing operate as **black boxes**, making their outputs difficult to justify.

This project addresses that problem by developing a **conceptual model** that integrates **Explainable Artificial Intelligence (XAI)** into the financial auditing process to improve transparency and audit quality. The model follows the **Design Science Research** methodology and defines a **27-step pipeline** covering:

1. Data ingestion
2. Anomaly detection
3. SHAP-based explanation
4. ERP control attribution
5. ISA standard retrieval
6. Narrative generation

A **focused prototype** (steps 19–25) was built and evaluated through verification tests, validation tests, retrieval accuracy analysis, and expert interviews using the Delphi method.

## How the Prototype Works

1. **Indexing** — ISA standard PDFs are chunked by paragraph and embedded into a local ChromaDB vector store (runs once on first launch).
2. **Retrieval** — Given a SHAP anomaly feature and an ERP control failure, the system retrieves the most relevant ISA paragraph for each using semantic similarity search.
3. **Generation** — A local Ollama LLM synthesises both paragraphs into a concise, standards-grounded audit finding.
4. **Display** — The Streamlit UI shows both retrieved ISA paragraphs alongside the generated narrative.

## Repository Structure

```
ISA-Audit-Intelligence/
├── README.md
├── requirements.txt
├── src/                          # Source code
│   ├── app.py                    # Streamlit entry point
│   ├── config.py                 # Constants & dropdown options
│   ├── models.py                 # Data classes & TypedDicts
│   ├── indexer.py                # PDF chunking & corpus indexing
│   ├── vector_store.py           # ChromaDB vector index
│   ├── retrieval.py              # RAG retrieval node
│   ├── llm.py                    # Local LLM via Ollama
│   ├── ui.py                     # Streamlit UI components
│   └── pipeline.py               # LangGraph orchestrator
├── docs/                         # Documentation
│   ├── George_Samy_Nabih_58-21333.pdf          # Thesis report
│   ├── AI in Financial Reporting...Final Report.docx
│   └── diagrams/                 # UML & architecture diagrams
│       ├── ClassDiagram.drawio.png              # Class diagram
│       ├── SequanceDiagram.png                  # Sequence diagram
│       ├── RefinedSequanceDiagram.png           # Refined sequence diagram
│       ├── SolutionApproach.drawio.png          # Solution approach diagram
│       └── *.drawio                             # Editable draw.io source files
└── resources/
    └── ISA-Standards/            # ISA PDF source documents
        ├── ISA 240.pdf
        ├── ISA 265.PDF
        ├── ISA 315.pdf
        ├── ISA 330.pdf
        └── ISA530 (amended) - Audit Sampling.pdf
```

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| [Ollama](https://ollama.com) | latest | Local LLM server |
| `llama3.1:8b-instruct-q4_K_M` | — | LLM model (pulled via Ollama) |

## Setup and Installation

```bash
# 1. Clone the repo
git clone https://github.com/georgeesamy/ISA-Audit-Intelligence.git
cd ISA-Audit-Intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy ISA PDFs into the runtime data folder
# The ISA source PDFs are in resources/ISA-Standards/.
# Copy them to src/Data/ so the indexer can find them:
mkdir -p src/Data
cp resources/ISA-Standards/* src/Data/
# Windows: copy resources\ISA-Standards\* src\Data\

# 5. Start Ollama and pull the model
ollama serve                                    # in a separate terminal
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

## Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[ChromaDB](https://www.trychroma.com)** — Local vector database
- **[sentence-transformers](https://www.sbert.net)** — `BAAI/bge-large-en-v1.5` embeddings
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Pipeline orchestration
- **[pdfplumber](https://github.com/jsvine/pdfplumber)** — PDF text extraction
- **[Ollama](https://ollama.com)** — Local LLM inference (Llama 3.1 8B)
- **[SHAP](https://shap.readthedocs.io)** — Explainable AI feature attribution

## Methodology

This project follows the **Design Science Research Methodology (DSRM)** by Peffers et al. (2007), combined with **CRISP-DM** for data processing stages. Evaluation was conducted through:

- **Verification tests** — confirming each pipeline step produces correct outputs
- **Validation tests** — confirming end-to-end system behavior meets requirements
- **Retrieval accuracy analysis** — measuring precision/recall of ISA paragraph retrieval
- **Expert interviews** — four domain professionals evaluated the system using the Delphi method

## Notes

- `chroma_db/` is generated at runtime and gitignored.
- Ollama must be running on `localhost:11434` before clicking Analyse.
- The LLM model can be changed in `src/config.py` (`LLM_MODEL`).
- All processing happens locally — no data is transmitted to external servers.
