from dataclasses import dataclass, field
from typing import TypedDict, Optional


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
