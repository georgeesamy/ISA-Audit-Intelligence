from langgraph.graph import StateGraph, END
from models import PipelineState
from retrieval import ISARAGRetrieval
from llm import LocalLLM
from ui import UI_Dashboard


class LangGraphOrchestrator:
    def __init__(self, isa_index):
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

    def start_pipeline(self, shap_column_name: str, erp_attribution_type: str) -> PipelineState:
        initial_state: PipelineState = {
            "shap_column_name": shap_column_name,
            "erp_attribution_type": erp_attribution_type,
            "isa_paragraph_shap": "",
            "isa_paragraph_erp": "",
            "enriched_explanation": None,
            "final_output": None,
        }
        final_state = self._graph.invoke(initial_state)
        self._ui_dashboard.render_findings(final_state)
        return final_state
