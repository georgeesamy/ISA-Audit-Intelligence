import os
import streamlit as st
from config import ERP_CONTROL_OPTIONS, SHAP_FEATURE_OPTIONS, PDF_PATHS
from vector_store import ISAVectorIndex
from indexer import index_isa_corpus
from pipeline import LangGraphOrchestrator
from ui import CUSTOM_CSS, HEADER_HTML


@st.cache_resource
def load_pipeline() -> LangGraphOrchestrator:
    isa_index = ISAVectorIndex()
    if isa_index._collection.count() == 0:
        existing = [p for p in PDF_PATHS if os.path.exists(p)]
        if existing:
            with st.spinner("Indexing ISA corpus (first launch only) …"):
                index_isa_corpus(existing, isa_index)
    return LangGraphOrchestrator(isa_index)


def main():
    st.set_page_config(
        page_title="ISA Audit Intelligence",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    orchestrator = load_pipeline()

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
