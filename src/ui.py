import streamlit as st
from models import PipelineState

HEADER_HTML = """
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
"""

CUSTOM_CSS = """
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
"""


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
