"""
ISA Demo Retrieval Evaluation -- Thesis Confusion Matrix
Run from project root: python src/evaluate_demo.py
"""

import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from app import ISAVectorIndex


SHAP_CASES = [
    {
        "name": "posting_hour",
        "query": "auditor select journal entries adjustments end reporting period inquiries financial reporting process unusual activity throughout period",
        "expected_isa": "ISA 240",
        "expected_para": "A45",
    },
    {
        "name": "amount_ratio",
        "query": "auditor design perform substantive procedures material class transactions account balance irrespective assessed risk expectation recorded amounts differences",
        "expected_isa": "ISA 330",
        "expected_para": "18",
    },
    {
        "name": "account_deviation",
        "query": "account balance deviation",
        "expected_isa": "ISA 240",
        "expected_para": "23",
    },
]

ERP_CASES = [
    {
        "name": "SoD_Violation",
        "query": "identified deficiencies internal control individually combination constitute significant deficiencies those charged governance",
        "expected_isa": "ISA 265",
        "expected_para": "8",
    },
    {
        "name": "Management_Override",
        "query": "auditor shall irrespective assessment risks management override controls design perform audit procedures review accounting estimates evaluate significant transactions outside normal course business",
        "expected_isa": "ISA 240",
        "expected_para": "33",
    },
    {
        "name": "Missing_Approval",
        "query": "missing approval",
        "expected_isa": "ISA 330",
        "expected_para": "8",
    },
]

OUTPUT_SHAP = os.path.join(os.path.dirname(__file__), "..", "confusion_matrix_demo_shap.png")
OUTPUT_ERP  = os.path.join(os.path.dirname(__file__), "..", "confusion_matrix_demo_erp.png")


def parse_citation(result: str) -> str:
    m = re.match(r"\[ISA\s*(\d+),\s*Para\s*([^\]]+)\]", result.strip())
    if m:
        return f"ISA {m.group(1)} Para {m.group(2).strip().rstrip('.')}"
    return "Unknown"


def plot_cm(y_true, y_pred, case_names, title, output_path):
    # Only expected labels -- wrong retrievals show as a miss (zero row) not an extra column
    all_labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    n = len(all_labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 2.2), max(5, n * 1.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)

    # Annotate y-axis: add case name and PASS/FAIL to each expected-class row
    y_tick_labels = []
    for label in all_labels:
        matched = [(name, pred) for name, exp, pred in zip(case_names, y_true, y_pred) if exp == label]
        if matched:
            name, pred = matched[0]
            mark = "PASS" if label == pred else "FAIL"
            y_tick_labels.append(f"{label}\n[{name}: {mark}]")
        else:
            y_tick_labels.append(label)

    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.set_title(
        f"{title}\nRows = Expected  |  Columns = Retrieved\n"
        f"Diagonal = correct retrieval  |  Off-diagonal = wrong retrieval",
        fontsize=10, pad=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.abspath(output_path)}\n")


def evaluate_group(cases: list, label: str, output_path: str, isa_index: ISAVectorIndex):
    print(f"\n{'=' * 62}")
    print(f"  {label}  ({len(cases)} queries)")
    print(f"{'=' * 62}")

    y_true, y_pred, names = [], [], []

    for case in cases:
        expected  = f"{case['expected_isa']} Para {case['expected_para']}"
        raw       = isa_index.similarity_search(case["query"])
        predicted = parse_citation(raw)

        y_true.append(expected)
        y_pred.append(predicted)
        names.append(case["name"])

        mark = "PASS" if predicted == expected else "FAIL"
        print(f"  [{mark}]  {case['name']:<22}  expected={expected:<20}  got={predicted}")

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    print(f"\n  Accuracy: {correct}/{total}  ({100 * correct / total:.0f}%)\n")

    exp_labels = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=exp_labels, zero_division=0)
    print("  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")
    print()

    plot_cm(y_true, y_pred, names, label, output_path)


def run():
    print("\nLoading ISA Vector Index ...\n")
    isa_index = ISAVectorIndex()

    evaluate_group(SHAP_CASES, "SHAP Feature Retrieval", OUTPUT_SHAP, isa_index)
    evaluate_group(ERP_CASES,  "ERP Attribution Retrieval", OUTPUT_ERP, isa_index)


if __name__ == "__main__":
    run()
