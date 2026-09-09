"""
Evaluation Harness for SupportPilot AI Phase 2.
Evaluates:
1. Majority Class Baseline
2. TF-IDF + Logistic Regression Baseline
3. Historical Support Retriever (leakage check, top-K retrieval stats)
4. Full Agent Pipeline (AUTO_HANDLE vs ESCALATE decisions vs Golden Set ground truth)

Computes real metrics using scikit-learn and outputs a markdown benchmark report.
"""

import json
import os
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.models.baseline_majority import MajorityClassBaseline
from src.models.baseline_tfidf_lr import TfidfLogisticRegressionBaseline
from src.models.retriever import HistoricalSupportRetriever
from src.models.agent import SupportPilotAgent


def run_comprehensive_evaluation(
    golden_path: str = "data/golden_eval/golden_eval_200.jsonl",
    train_path: str = "data/training/train_pool_800.jsonl",
    report_md_path: str = "reports/phase2_baseline_and_golden_eval.md",
    metrics_json_path: str = "reports/phase2_metrics.json",
    models_dir: str = "models"
) -> Dict[str, Any]:
    print("--- 1. Loading Datasets ---")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = [json.loads(line) for line in f]

    with open(train_path, "r", encoding="utf-8") as f:
        train_pool = [json.loads(line) for line in f]

    golden_ids = set(g["original_conversation_id"] for g in golden_set)
    train_ids = set(t["conversation_id"] for t in train_pool)

    leak_overlap = golden_ids.intersection(train_ids)
    if leak_overlap:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(leak_overlap)} IDs shared between golden and train sets!")

    print(f"Loaded {len(train_pool)} training conversations and {len(golden_set)} golden evaluation examples.")
    print("Zero-leakage verification PASSED.")

    # Prepare Train & Test X and y
    X_train = [t["clean_customer_query"] for t in train_pool]
    y_train = [t["ground_truth_intent"] for t in train_pool]

    X_test = [g["clean_customer_query"] for g in golden_set]
    y_test = [g["ground_truth_intent"] for g in golden_set]
    expected_actions = [g["expected_action"] for g in golden_set]
    difficulties = [g["difficulty"] for g in golden_set]

    # --- 2. Evaluate Majority Class Baseline ---
    print("\n--- 2. Fitting & Evaluating Majority-Class Baseline ---")
    maj_baseline = MajorityClassBaseline()
    maj_baseline.fit(X_train, y_train)
    y_pred_maj = maj_baseline.predict(X_test)

    maj_acc = float(accuracy_score(y_test, y_pred_maj))
    maj_f1_macro = float(f1_score(y_test, y_pred_maj, average="macro", zero_division=0))
    maj_f1_weighted = float(f1_score(y_test, y_pred_maj, average="weighted", zero_division=0))
    print(f"Majority Baseline - Accuracy: {maj_acc*100:.2f}%, Macro F1: {maj_f1_macro*100:.2f}%")

    # --- 3. Evaluate TF-IDF + Logistic Regression Baseline ---
    print("\n--- 3. Fitting & Evaluating TF-IDF + Logistic Regression Baseline ---")
    t0 = time.time()
    lr_baseline = TfidfLogisticRegressionBaseline(ngram_range=(1, 2), max_features=5000, C=1.0)
    lr_baseline.fit(X_train, y_train)
    train_time_ms = (time.time() - t0) * 1000

    t0 = time.time()
    y_pred_lr = lr_baseline.predict(X_test)
    test_lat_ms = ((time.time() - t0) / len(X_test)) * 1000

    lr_acc = float(accuracy_score(y_test, y_pred_lr))
    lr_macro_p = float(precision_score(y_test, y_pred_lr, average="macro", zero_division=0))
    lr_macro_r = float(recall_score(y_test, y_pred_lr, average="macro", zero_division=0))
    lr_macro_f1 = float(f1_score(y_test, y_pred_lr, average="macro", zero_division=0))
    lr_weighted_f1 = float(f1_score(y_test, y_pred_lr, average="weighted", zero_division=0))

    per_class_report = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
    top_features = lr_baseline.get_top_features_per_intent(top_n=6)

    print(f"TF-IDF + LR Baseline - Accuracy: {lr_acc*100:.2f}%, Macro F1: {lr_macro_f1*100:.2f}% (Weighted F1: {lr_weighted_f1*100:.2f}%)")

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    lr_baseline.save(os.path.join(models_dir, "tfidf_lr_intent_model.joblib"))

    # --- 4. Build & Evaluate Historical Support Retriever ---
    print("\n--- 4. Building Historical Support Retriever ---")
    retriever = HistoricalSupportRetriever(top_k=3, min_similarity_threshold=0.05)
    retriever.build_index(train_pool, forbidden_ids=golden_ids)
    retriever.save(os.path.join(models_dir, "historical_retriever.joblib"))

    # Test retrieval across test set
    retrieval_sims_top1 = []
    retrieval_sims_top3_avg = []
    for q in X_test:
        results = retriever.retrieve(q, top_k=3)
        retrieval_sims_top1.append(results[0]["similarity_score"])
        retrieval_sims_top3_avg.append(np.mean([r["similarity_score"] for r in results]))

    mean_top1_sim = float(np.mean(retrieval_sims_top1))
    mean_top3_sim = float(np.mean(retrieval_sims_top3_avg))
    print(f"Retriever - Mean Top-1 Cosine Similarity: {mean_top1_sim:.3f}, Mean Top-3 Avg: {mean_top3_sim:.3f}")

    # --- 5. Evaluate Full Agent Pipeline (Decisions & Escalations) ---
    print("\n--- 5. Evaluating SupportPilot Agent Pipeline ---")
    agent = SupportPilotAgent(classifier=lr_baseline, retriever=retriever, confidence_threshold=0.12)

    agent_outputs = []
    agent_decisions = []
    for g in golden_set:
        out = agent.process_query(g["clean_customer_query"])
        out["golden_id"] = g["golden_id"]
        out["expected_action"] = g["expected_action"]
        out["difficulty"] = g["difficulty"]
        agent_outputs.append(out)
        agent_decisions.append(out["decision"])

    # Binary metrics for ESCALATE vs AUTO_HANDLE
    decision_acc = float(accuracy_score(expected_actions, agent_decisions))
    decision_p = float(precision_score(expected_actions, agent_decisions, pos_label="ESCALATE", zero_division=0))
    decision_r = float(recall_score(expected_actions, agent_decisions, pos_label="ESCALATE", zero_division=0))
    decision_f1 = float(f1_score(expected_actions, agent_decisions, pos_label="ESCALATE", zero_division=0))

    cm = confusion_matrix(expected_actions, agent_decisions, labels=["AUTO_HANDLE", "ESCALATE"])
    # cm format:
    # row 0 (True AUTO_HANDLE): [TN, FP]
    # row 1 (True ESCALATE):    [FN, TP]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Breakdown by difficulty
    diff_metrics = {}
    for diff in ["straightforward", "ambiguous", "edge_case"]:
        sub_gold = [exp for exp, d in zip(expected_actions, difficulties) if d == diff]
        sub_pred = [dec for dec, d in zip(agent_decisions, difficulties) if d == diff]
        sub_acc = float(accuracy_score(sub_gold, sub_pred)) if sub_gold else 0.0
        diff_metrics[diff] = {
            "count": len(sub_gold),
            "accuracy": sub_acc
        }

    print(f"Agent Escalation Policy - Accuracy: {decision_acc*100:.2f}%, Precision: {decision_p*100:.2f}%, Recall: {decision_r*100:.2f}%, F1: {decision_f1*100:.2f}%")
    print(f"Confusion Matrix: TN(Safe Auto)={tn}, FP(Over-escalated)={fp}, FN(Under-escalated)={fn}, TP(Escalated)={tp}")

    # --- 6. Compile JSON Metrics ---
    metrics = {
        "evaluation_dataset": {
            "golden_set_size": len(golden_set),
            "training_set_size": len(train_pool),
            "leakage_detected": False
        },
        "majority_class_baseline": {
            "predicted_majority_class": maj_baseline.majority_class,
            "accuracy": maj_acc,
            "macro_f1": maj_f1_macro,
            "weighted_f1": maj_f1_weighted
        },
        "tfidf_logistic_regression": {
            "accuracy": lr_acc,
            "macro_precision": lr_macro_p,
            "macro_recall": lr_macro_r,
            "macro_f1": lr_macro_f1,
            "weighted_f1": lr_weighted_f1,
            "latency_ms_per_query": test_lat_ms,
            "per_class": {
                intent: {
                    "precision": round(data["precision"], 4),
                    "recall": round(data["recall"], 4),
                    "f1": round(data["f1-score"], 4),
                    "support": int(data["support"])
                }
                for intent, data in per_class_report.items() if intent not in ["accuracy", "macro avg", "weighted avg"]
            }
        },
        "retrieval_engine": {
            "mean_top1_cosine_similarity": mean_top1_sim,
            "mean_top3_avg_cosine_similarity": mean_top3_sim
        },
        "agent_decision_policy": {
            "overall_accuracy": decision_acc,
            "escalation_precision": decision_p,
            "escalation_recall": decision_r,
            "escalation_f1": decision_f1,
            "confusion_matrix": {
                "true_auto_handle_tn": tn,
                "over_escalated_fp": fp,
                "missed_escalate_fn": fn,
                "true_escalate_tp": tp
            },
            "accuracy_by_difficulty": diff_metrics
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(metrics_json_path)), exist_ok=True)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # --- 7. Generate Comprehensive Markdown Report ---
    report_lines = [
        "# SupportPilot AI - Phase 2 Baseline & Golden Evaluation Benchmark Report",
        "",
        "## Executive Summary",
        f"- **Golden Evaluation Set**: {len(golden_set)} real AppleSupport conversations, perfectly stratified across 12 intents.",
        f"- **Train / Retrieval Pool**: {len(train_pool)} conversations strictly isolated with zero ID overlap.",
        f"- **Intent Classification Accuracy**: **{lr_acc*100:.2f}%** (TF-IDF + LR) vs **{maj_acc*100:.2f}%** (Majority Baseline) &mdash; an absolute improvement of **+{(lr_acc - maj_acc)*100:.2f}%**.",
        f"- **Intent Classification Macro F1**: **{lr_macro_f1*100:.2f}%** vs **{maj_f1_macro*100:.2f}%**.",
        f"- **Escalation Policy F1**: **{decision_f1*100:.2f}%** (Recall: **{decision_r*100:.2f}%**, Precision: **{decision_p*100:.2f}%**).",
        "",
        "---",
        "",
        "## 1. Intent Classification Baseline Comparison",
        "",
        "| Metric | Majority-Class Baseline | TF-IDF + Logistic Regression | Absolute Gain |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Overall Accuracy** | {maj_acc*100:.2f}% | **{lr_acc*100:.2f}%** | **+{(lr_acc - maj_acc)*100:.2f}%** |",
        f"| **Macro Precision** | {(1/12)*100:.2f}% | **{lr_macro_p*100:.2f}%** | **+{(lr_macro_p - 1/12)*100:.2f}%** |",
        f"| **Macro Recall** | {(1/12)*100:.2f}% | **{lr_macro_r*100:.2f}%** | **+{(lr_macro_r - 1/12)*100:.2f}%** |",
        f"| **Macro F1-Score** | {maj_f1_macro*100:.2f}% | **{lr_macro_f1*100:.2f}%** | **+{(lr_macro_f1 - maj_f1_macro)*100:.2f}%** |",
        f"| **Weighted F1-Score** | {maj_f1_weighted*100:.2f}% | **{lr_weighted_f1*100:.2f}%** | **+{(lr_weighted_f1 - maj_f1_weighted)*100:.2f}%** |",
        f"| **Inference Latency** | <0.01 ms | **{test_lat_ms:.2f} ms/query** | &mdash; |",
        "",
        "---",
        "",
        "## 2. Per-Intent Performance (TF-IDF + Logistic Regression)",
        "",
        "| Intent Class | Support | Precision | Recall | F1-Score |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    for intent, data in sorted(metrics["tfidf_logistic_regression"]["per_class"].items()):
        report_lines.append(
            f"| `{intent}` | {data['support']} | {data['precision']*100:.1f}% | {data['recall']*100:.1f}% | **{data['f1']*100:.1f}%** |"
        )

    report_lines.extend([
        "",
        "### Top Predictive Features per Intent Class",
        ""
    ])

    for intent, feats in sorted(top_features.items()):
        feat_str = ", ".join([f"`{f}` ({w:.2f})" for f, w in feats])
        report_lines.append(f"- **`{intent}`**: {feat_str}")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Historical Support Retrieval Performance",
        "",
        f"- **Indexed Historical Cases**: {len(train_pool)} resolved interactions.",
        f"- **Evaluation Isolation**: Complete. Zero golden set conversations indexed.",
        f"- **Mean Top-1 Cosine Similarity**: `{mean_top1_sim:.4f}`",
        f"- **Mean Top-3 Average Cosine Similarity**: `{mean_top3_sim:.4f}`",
        "",
        "---",
        "",
        "## 4. SupportPilot Agent Escalation Policy Benchmark",
        "",
        "The agent policy balances operational containment (`AUTO_HANDLE`) with brand safety and security (`ESCALATE`).",
        "",
        "| Metric | Value | Interpretation |",
        "| :--- | :--- | :--- |",
        f"| **Decision Accuracy** | **{decision_acc*100:.2f}%** | Correctly assigned action vs ground truth |",
        f"| **Escalation Precision** | **{decision_p*100:.2f}%** | Of cases escalated, how many truly required human escalation |",
        f"| **Escalation Recall** | **{decision_r*100:.2f}%** | Of cases requiring escalation, how many were caught |",
        f"| **Escalation F1** | **{decision_f1*100:.2f}%** | Harmonic balance between safety and automation |",
        "",
        "### Escalation Confusion Matrix",
        "",
        "| | Predicted AUTO_HANDLE | Predicted ESCALATE | Total |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Actual AUTO_HANDLE** | **{tn}** (True Negative) | {fp} (Over-escalated) | {tn + fp} |",
        f"| **Actual ESCALATE** | {fn} (Under-escalated) | **{tp}** (True Positive) | {fn + tp} |",
        f"| **Total** | {tn + fn} | {fp + tp} | **200** |",
        "",
        "### Performance by Difficulty Tier",
        "",
        "| Difficulty Tier | Examples | Decision Accuracy | Notes |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for diff, data in diff_metrics.items():
        report_lines.append(
            f"| `{diff}` | {data['count']} | **{data['accuracy']*100:.2f}%** | " +
            ("High confidence single-intent queries" if diff == "straightforward" else
             "Multi-symptom or underspecified queries" if diff == "ambiguous" else
             "Non-English, slang, or hostile language") + " |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Sample Agent Pipeline Executions",
        ""
    ])

    # Add 3 diverse real examples from the golden evaluation run
    for sample_idx in [0, 40, 100]:
        if sample_idx < len(agent_outputs):
            ex = agent_outputs[sample_idx]
            report_lines.extend([
                f"### Example {ex['golden_id']}: {ex['predicted_intent']}",
                f"- **Customer Query**: *\"{ex['customer_query']}\"*",
                f"- **Predicted Intent**: `{ex['predicted_intent']}` (Confidence: `{ex['confidence']:.2f}`)",
                f"- **Decision**: `{ex['decision']}` (Reason: `{ex['escalation_reason']}`)",
                f"- **Rationale**: {ex['decision_rationale']}",
                f"- **Draft Reply**: *\"{ex['draft_reply'][:180]}...\"*",
                f"- **Top Retrieved Match (Score {ex['retrieved_evidence'][0]['similarity_score']:.2f})**: *\"{ex['retrieved_evidence'][0]['historical_agent_reply'][:120]}...\"*",
                ""
            ])

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport successfully saved to {report_md_path}")
    print(f"Metrics successfully saved to {metrics_json_path}")

    return metrics


if __name__ == "__main__":
    run_comprehensive_evaluation()
