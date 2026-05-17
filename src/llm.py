import requests
import streamlit as st
from models import EnrichedExplanation, PipelineState
from config import LLM_MODEL, LLM_MAX_CHARS, LLM_TIMEOUT, OLLAMA_URL


class LocalLLM:
    def generate(self, enriched: list[EnrichedExplanation], isa_paragraph_erp: str = "") -> list[str]:
        outputs = []
        for entry in enriched:
            shap_label = entry.shap_column_name.split(" — ")[0].strip()
            erp_label = entry.erp_attribution.attribution_type.split(" — ")[0].strip()
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
                f"ISA Paragraph (Anomaly): {entry.isa_paragraph[:LLM_MAX_CHARS]}\n"
                f"ISA Paragraph (Control): {isa_paragraph_erp[:LLM_MAX_CHARS]}\n\n"
                "Audit Finding:"
            )
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                    timeout=LLM_TIMEOUT,
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

    def generate_node(self, state: PipelineState) -> dict:
        enriched = state["enriched_explanation"]
        outputs = self.generate([enriched], isa_paragraph_erp=state["isa_paragraph_erp"])
        final_output = outputs[0] if outputs else ""
        if final_output.startswith("ERROR:"):
            st.error(final_output)
            return {"final_output": ""}
        final_output = (
            final_output
            .replace("The auditor", "The findings")
            .replace("the auditor", "the findings")
            .replace("The Auditor", "The findings")
            .replace("auditor", "findings")
        )
        return {"final_output": final_output}
