import os
import re
import requests
import streamlit as st
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field


# ── Dropdown options ──────────────────────────────────────────────────────────

ERP_CONTROL_OPTIONS = [
    "SoD Violation — identified deficiencies internal control determine whether individually combination constitute significant deficiencies",
    "Management Override of Controls — irrespective auditor assessment management override shall design perform audit procedures accounting estimates",
    "Missing Approval — auditor shall design perform tests of controls sufficient audit evidence operating effectiveness relevant controls",
]

SHAP_FEATURE_OPTIONS = [
    "posting_hour — journal entries end of reporting period unusual posting hour select adjustments",
    "amount_ratio — substantive procedures material class transactions account balance irrespective assessed risk",
    "account_deviation — auditor shall evaluate unusual unexpected relationships analytical procedures investigate misstatement material",
]


# ── Fig 17: Chunking ──────────────────────────────────────────────────────────

def chunk_isa_document(pdf_path: str) -> list[dict]:
    chunks = []
    current_chunk_lines = []
    current_para_id = None
    para_pattern = re.compile(r'^(\d+\.|A\d+\.)\s')

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                stripped = line.strip()
                match = para_pattern.match(stripped)
                if match:
                    if current_chunk_lines and current_para_id:
                        chunks.append({
                            "para_id": current_para_id,
                            "text": " ".join(current_chunk_lines).strip(),
                            "source": pdf_path
                        })
                    current_para_id = match.group(1)
                    current_chunk_lines = [stripped]
                else:
                    if current_chunk_lines:
                        current_chunk_lines.append(stripped)

    if current_chunk_lines and current_para_id:
        chunks.append({
            "para_id": current_para_id,
            "text": " ".join(current_chunk_lines).strip(),
            "source": pdf_path
        })
    return chunks


# ── Fig 18: ISA Vector Index ──────────────────────────────────────────────────

class ISAVectorIndex:
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    similarity_metric: str = "cosine"
    index_name: str = "isa_corpus"

    def __init__(self):
        self._client = chromadb.PersistentClient(path="chroma_db")
        self._encoder = SentenceTransformer(self.embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=self.index_name,
            metadata={"hnsw:space": self.similarity_metric}
        )

    def similarity_search(self, query: str) -> str:
        query_vector = self._encoder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=["documents", "metadatas"]
        )
        if results["documents"] and results["documents"][0]:
            text = results["documents"][0][0]
            meta = results["metadatas"][0][0] if results["metadatas"] else {}
            source = meta.get("source", "")
            para_id = meta.get("para_id", "")
            m = re.match(r'ISA\s*(\d+)', os.path.basename(source), re.IGNORECASE)
            isa_name = f"ISA {m.group(1)}" if m else os.path.splitext(os.path.basename(source))[0]
            citation = f"[{isa_name}, Para {para_id}]" if isa_name and para_id else ""
            return f"{citation}\n{text}" if citation else text
        return ""


def index_isa_corpus(pdf_paths: list[str], isa_index: ISAVectorIndex) -> None:
    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(chunk_isa_document(pdf_path))

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = isa_index._encoder.encode(
        texts, batch_size=32, show_progress_bar=True
    )
    isa_index._collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[f"{chunk['source']}_{i}" for i, chunk in enumerate(all_chunks)],
        metadatas=[
            {"source": chunk["source"], "para_id": chunk["para_id"]}
            for chunk in all_chunks
        ]
    )


# ── Fig 19: Data Classes ──────────────────────────────────────────────────────

@dataclass
class ControlAttribution:
    gl_row_id: str = ""
    attribution_type: str = ""
    traversal_path: list = field(default_factory=list)


@dataclass
class MergedExplanation:
    gl_row_id: str
    ensemble_score: float
    shap_column_name: str
    erp_attribution: ControlAttribution


@dataclass
class EnrichedExplanation:
    gl_row_id: str = ""
    isa_paragraph: str = ""
    shap_column_name: str = ""
    erp_attribution: ControlAttribution = field(default_factory=ControlAttribution)


class PipelineState(TypedDict):
    shap_column_name: str
    erp_attribution_type: str
    isa_paragraph_shap: str
    isa_paragraph_erp: str
    enriched_explanation: Optional[EnrichedExplanation]
    final_output: Optional[str]


# ── Fig 22: ISA RAG Retrieval ─────────────────────────────────────────────────

class ISARAGRetrieval:
    def __init__(self, isa_index: ISAVectorIndex):
        self._isa_index = isa_index

    def retrieve(
        self, merged: list[MergedExplanation]
    ) -> list[EnrichedExplanation]:
        results = []
        for entry in merged:
            state = {
                "shap_column_name": entry.shap_column_name,
                "erp_attribution_type": entry.erp_attribution.attribution_type
            }
            node_result = self.retrieve_node(state)
            results.append(node_result["enriched_explanation"])
        return results

    def retrieve_node(self, state: PipelineState) -> dict:
        shap_query = state['shap_column_name'].split(' — ')[-1] if ' — ' in state['shap_column_name'] else state['shap_column_name']
        erp_query = state['erp_attribution_type'].split(' — ')[-1] if ' — ' in state['erp_attribution_type'] else state['erp_attribution_type']
        isa_paragraph_shap = self._isa_index.similarity_search(shap_query)
        isa_paragraph_erp = self._isa_index.similarity_search(erp_query)
        erp_attribution = ControlAttribution(
            attribution_type=state["erp_attribution_type"]
        )
        enriched = EnrichedExplanation(
            isa_paragraph=isa_paragraph_shap,
            shap_column_name=state["shap_column_name"],
            erp_attribution=erp_attribution
        )
        return {
            "isa_paragraph_shap": isa_paragraph_shap,
            "isa_paragraph_erp": isa_paragraph_erp,
            "enriched_explanation": enriched
        }


# ── Fig 23: Local LLM ─────────────────────────────────────────────────────────

class LocalLLM:
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    max_chars: int = 4000
    timeout: int = 120

    def generate(
        self,
        enriched: list[EnrichedExplanation],
        isa_paragraph_erp: str = ""
    ) -> list[str]:
        outputs = []
        for entry in enriched:
            shap_label = entry.shap_column_name.split(" — ")[0].strip()
            erp_label  = entry.erp_attribution.attribution_type.split(" — ")[0].strip()
            prompt = (
                "You are a financial audit analysis system.\n\n"
                "Instructions:\n"
                "- Write a short audit finding in third person (no 'I' or 'we').\n"
                "- Two to three sentences only. No preamble, no headers, no bullet points.\n"
                "- Explain what anomaly type was detected and which ERP control failed, "
                "grounded strictly in the two ISA paragraphs below.\n"
                "- In your explanation sentences, do not use the word 'auditor'. "
                "Use 'the findings' instead wherever you would say 'auditor'.\n"
                "- Do NOT invent specific values, times, amounts, or any detail not present in the inputs.\n"
                "- After the explanation, reproduce both ISA paragraphs exactly "
                "word for word, each on its own line, unchanged.\n\n"
                f"Anomaly Feature: {shap_label}\n"
                f"ERP Control Failure: {erp_label}\n"
                f"ISA Paragraph (Anomaly): {entry.isa_paragraph[:self.max_chars]}\n"
                f"ISA Paragraph (Control): {isa_paragraph_erp[:self.max_chars]}\n\n"
                "Audit Finding:"
            )
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0}
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                outputs.append(response.json().get("response", ""))
            except requests.exceptions.ConnectionError:
                outputs.append("ERROR: Local LLM is unavailable. Please ensure Ollama is running and try again.")
            except requests.exceptions.Timeout:
                outputs.append("ERROR: LLM request timed out. The model took too long to respond.")
            except Exception as e:
                outputs.append(f"ERROR: Narrative generation failed. {str(e)}")
        return outputs

    def generate_node(
        self,
        state: PipelineState
    ) -> dict:
        enriched = state["enriched_explanation"]
        outputs = self.generate(
            [enriched],
            isa_paragraph_erp=state["isa_paragraph_erp"]
        )
        final_output = outputs[0] if outputs else ""
        if final_output.startswith("ERROR:"):
            st.error(final_output)
            return {"final_output": ""}
        final_output = final_output.replace("The auditor", "The findings") \
                                   .replace("the auditor", "the findings") \
                                   .replace("The Auditor", "The findings") \
                                   .replace("auditor", "findings")
        return {"final_output": final_output}


# ── UI Dashboard ──────────────────────────────────────────────────────────────

class UI_Dashboard:
    def render_findings(self, final_state: PipelineState) -> None:
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ISA Basis — ERP Control**")
            st.info(final_state["isa_paragraph_erp"] or "No paragraph retrieved.")
        with col2:
            st.markdown("**ISA Basis — SHAP Feature**")
            st.info(final_state["isa_paragraph_shap"] or "No paragraph retrieved.")

        st.markdown("**Audit Observation**")
        st.success(final_state["final_output"] or "No narrative generated.")

    def render_no_findings(self) -> None:
        st.info("No entries exceed threshold")

    def export(self, format: str) -> None:
        pass


# ── Fig 20: LangGraph Orchestrator ───────────────────────────────────────────

class LangGraphOrchestrator:
    def __init__(self, isa_index: ISAVectorIndex):
        self._isa_retrieval = ISARAGRetrieval(isa_index)
        self._local_llm = LocalLLM()
        self._ui_dashboard = UI_Dashboard()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("retrieve_isa", self._isa_retrieval.retrieve_node)
        graph.add_node("generate_narrative", self._local_llm.generate_node)
        graph.set_entry_point("retrieve_isa")
        graph.add_edge("retrieve_isa", "generate_narrative")
        graph.add_edge("generate_narrative", END)
        return graph.compile()

    def start_pipeline(
        self, shap_column_name: str, erp_attribution_type: str
    ) -> PipelineState:
        initial_state: PipelineState = {
            "shap_column_name": shap_column_name,
            "erp_attribution_type": erp_attribution_type,
            "isa_paragraph_shap": "",
            "isa_paragraph_erp": "",
            "enriched_explanation": None,
            "final_output": None
        }
        final_state = self._graph.invoke(initial_state)
        self._ui_dashboard.render_findings(final_state)
        return final_state


# ── App entry point ───────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    isa_index = ISAVectorIndex()
    if isa_index._collection.count() == 0:
        pdf_paths = [
            "src/Data/ISA 240.pdf",
            "src/Data/ISA 265.PDF",
            "src/Data/ISA 315.pdf",
            "src/Data/ISA 330.pdf",
            "src/Data/ISA530 (amended) - Audit Sampling.pdf",
        ]
        existing = [p for p in pdf_paths if os.path.exists(p)]
        if existing:
            with st.spinner("Indexing ISA corpus (first launch only) …"):
                index_isa_corpus(existing, isa_index)
    orchestrator = LangGraphOrchestrator(isa_index)
    return isa_index, orchestrator


def main():
    st.set_page_config(
        page_title="ISA Audit Intelligence",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
        .stButton > button {
            background-color: #E8724A;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .stButton > button:hover {
            background-color: #D4613A;
            color: white;
        }
        .section-label {
            color: #E8724A;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .col-label {
            font-weight: 600;
            font-size: 0.82rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #444;
            margin-bottom: 0.25rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header bar
    st.markdown("""
    <div style="background-color:#1a1a2e;padding:1rem 1.5rem;border-radius:8px;margin-bottom:1.75rem;">
        <div style="display:flex;align-items:center;gap:0.85rem;">
            <div style="background-color:#E8724A;width:36px;height:36px;border-radius:6px;
                        display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
                📋
            </div>
            <div>
                <div style="color:white;font-weight:700;font-size:1.15rem;line-height:1.3;">
                    ISA Audit Intelligence
                </div>
                <div style="color:#888;font-size:0.76rem;letter-spacing:0.04em;">
                    ERP Attribution &nbsp;·&nbsp; SHAP Analysis &nbsp;·&nbsp; ISA Retrieval &nbsp;·&nbsp; AI Narrative
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, orchestrator = load_pipeline()

    st.markdown('<p class="section-label">Select Audit Inputs</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="col-label">ERP Control Attribute</p>', unsafe_allow_html=True)
        erp_attr = st.selectbox(
            label="ERP Control Attribute",
            options=ERP_CONTROL_OPTIONS,
            label_visibility="collapsed",
            format_func=lambda x: x.split(" — ")[0],
        )
    with col2:
        st.markdown('<p class="col-label">SHAP Feature Column</p>', unsafe_allow_html=True)
        shap_feat = st.selectbox(
            label="SHAP Feature Column",
            options=SHAP_FEATURE_OPTIONS,
            label_visibility="collapsed",
            format_func=lambda x: x.split(" — ")[0],
        )

    st.markdown("")
    if st.button("Analyse & Generate Narrative"):
        with st.spinner("Running pipeline …"):
            orchestrator.start_pipeline(shap_feat, erp_attr)


if __name__ == "__main__":
    main()
