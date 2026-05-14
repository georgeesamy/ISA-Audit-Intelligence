"""
ISA Retrieval Evaluation — Confusion Matrix (SHAP + ERP)
Run from the project root: python src/evaluate_retrieval.py
"""

import os
import sys
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from app import ISAVectorIndex


GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "ground_truth.json")
OUTPUT_SHAP = os.path.join(os.path.dirname(__file__), "..", "confusion_matrix_shap.png")
OUTPUT_ERP  = os.path.join(os.path.dirname(__file__), "..", "confusion_matrix_erp.png")


def parse_citation(result: str) -> str:
    """Extract 'ISA XXX Para YY' from similarity_search output."""
    m = re.match(r"\[ISA\s*(\d+),\s*Para\s*([^\]]+)\]", result.strip())
    if m:
        para = m.group(2).strip().rstrip(".")
        return f"ISA {m.group(1)} Para {para}"
    return "Unknown"


def plot_cm(y_true, y_pred, title, output_path):
    all_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(
        f"{title}\nRows = Expected  |  Columns = Retrieved",
        fontsize=11, pad=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {os.path.abspath(output_path)}\n")


def evaluate_group(cases: dict, label: str, output_path: str, isa_index: ISAVectorIndex):
    print(f"{'=' * 60}")
    print(f"  {label} ({len(cases)} queries)")
    print(f"{'=' * 60}")

    y_true, y_pred = [], []

    for case_name, case in cases.items():
        expected = f"{case['expected_isa']} Para {case['expected_para']}"
        result = isa_index.similarity_search(case["query"])
        predicted = parse_citation(result)

        y_true.append(expected)
        y_pred.append(predicted)

        mark = "PASS" if predicted == expected else "FAIL"
        print(f"  [{mark}]  {case_name:<22}  expected={expected:<18}  got={predicted}")

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    print(f"\n  Accuracy: {correct}/{total} ({100 * correct / total:.0f}%)\n")

    print("  Classification Report:")
    all_labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0)
    for line in report.splitlines():
        print(f"    {line}")
    print()

    plot_cm(y_true, y_pred, label, output_path)


def run_evaluation():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    shap_cases = {k: v for k, v in ground_truth.items() if v.get("type") == "shap"}
    erp_cases  = {k: v for k, v in ground_truth.items() if v.get("type") == "erp"}

    print("\nLoading ISA Vector Index …\n")
    isa_index = ISAVectorIndex()

    evaluate_group(shap_cases, "SHAP Feature Retrieval", OUTPUT_SHAP, isa_index)
    evaluate_group(erp_cases,  "ERP Attribution Retrieval", OUTPUT_ERP, isa_index)


if __name__ == "__main__":
    run_evaluation()
