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

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "isa_corpus"
SIMILARITY_METRIC = "cosine"

LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"
LLM_MAX_CHARS = 4000
LLM_TIMEOUT = 120
OLLAMA_URL = "http://localhost:11434/api/generate"

PDF_PATHS = [
    "src/Data/ISA 240.pdf",
    "src/Data/ISA 265.PDF",
    "src/Data/ISA 315.pdf",
    "src/Data/ISA 330.pdf",
    "src/Data/ISA530 (amended) - Audit Sampling.pdf",
]
