from dataclasses import dataclass, field
from typing import TypedDict, Optional


@dataclass
class ControlAttribution:
    attribution_type: str = ""


@dataclass
class EnrichedExplanation:
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
