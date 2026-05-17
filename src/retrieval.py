from models import ControlAttribution, EnrichedExplanation, MergedExplanation, PipelineState


class ISARAGRetrieval:
    def __init__(self, isa_index):
        self._isa_index = isa_index

    def retrieve(self, merged: list[MergedExplanation]) -> list[EnrichedExplanation]:
        results = []
        for entry in merged:
            state = {
                "shap_column_name": entry.shap_column_name,
                "erp_attribution_type": entry.erp_attribution.attribution_type,
            }
            node_result = self.retrieve_node(state)
            results.append(node_result["enriched_explanation"])
        return results

    def retrieve_node(self, state: PipelineState) -> dict:
        shap_query = state["shap_column_name"].split(" — ")[-1] if " — " in state["shap_column_name"] else state["shap_column_name"]
        erp_query = state["erp_attribution_type"].split(" — ")[-1] if " — " in state["erp_attribution_type"] else state["erp_attribution_type"]

        isa_paragraph_shap = self._isa_index.similarity_search(shap_query)
        isa_paragraph_erp = self._isa_index.similarity_search(erp_query)

        enriched = EnrichedExplanation(
            isa_paragraph=isa_paragraph_shap,
            shap_column_name=state["shap_column_name"],
            erp_attribution=ControlAttribution(attribution_type=state["erp_attribution_type"]),
        )
        return {
            "isa_paragraph_shap": isa_paragraph_shap,
            "isa_paragraph_erp": isa_paragraph_erp,
            "enriched_explanation": enriched,
        }
