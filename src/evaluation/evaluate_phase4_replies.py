"""Run Phase 4 reply-quality checks on the real zero-leakage golden set."""

import json
from pathlib import Path
from typing import Any, Dict

from src.evaluation.reply_quality import REPLY_QUALITY_RUBRIC, deterministic_quality_checks
from src.models.agent import SupportPilotAgent
from src.models.baseline_tfidf_lr import TfidfLogisticRegressionBaseline
from src.models.dense_retriever import DenseSupportRetriever


def run_phase4_evaluation(
    golden_path: str = "data/golden_eval/golden_eval_200.jsonl",
    train_path: str = "data/training/train_pool_800.jsonl",
    dense_retriever_path: str = "models/dense_retriever.joblib",
    metrics_path: str = "reports/phase4_reply_quality_metrics.json",
    report_path: str = "reports/phase4_grounded_reply_quality.md",
) -> Dict[str, Any]:
    golden = [json.loads(line) for line in Path(golden_path).read_text(encoding="utf-8").splitlines()]
    train = [json.loads(line) for line in Path(train_path).read_text(encoding="utf-8").splitlines()]
    golden_ids = {item["original_conversation_id"] for item in golden}
    if golden_ids.intersection({item["conversation_id"] for item in train}):
        raise ValueError("CRITICAL LEAKAGE DETECTED")

    classifier = TfidfLogisticRegressionBaseline().fit(
        [item["clean_customer_query"] for item in train], [item["ground_truth_intent"] for item in train]
    )
    retriever = DenseSupportRetriever.load(dense_retriever_path)
    if {item["conversation_id"] for item in retriever.corpus}.intersection(golden_ids):
        raise ValueError("CRITICAL DENSE-RETRIEVER LEAKAGE DETECTED")
    agent = SupportPilotAgent(classifier, retriever, confidence_threshold=0.12, similarity_threshold=0.45)
    outputs = []
    for example in golden:
        output = agent.process_query(example["clean_customer_query"])
        output["golden_id"] = example["golden_id"]
        output["grounding_similarity_threshold"] = agent.similarity_threshold
        outputs.append(output)

    metrics = {
        "evaluation_dataset": {"golden_set_size": len(golden), "training_set_size": len(train), "leakage_detected": False},
        "deterministic_quality_checks": deterministic_quality_checks(outputs),
        "reply_evidence_retriever": {"type": "dense", "model_name": retriever.model_name, "similarity_threshold": agent.similarity_threshold},
        "llm_judge": {"status": "not_run", "reason": "No real LLM judge callable or human labels were supplied; no synthetic judge score is reported.", "rubric": REPLY_QUALITY_RUBRIC},
        "human_llm_agreement": {"status": "ready_for_real_annotations", "matched_examples": 0},
    }
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checks = metrics["deterministic_quality_checks"]
    Path(report_path).write_text("\n".join([
        "# SupportPilot AI — Phase 4 Grounded Reply Quality",
        "",
        "## Actual evaluation",
        f"- Evaluated {len(golden)} real golden AppleSupport examples against an 800-thread zero-leakage retrieval pool.",
        f"- Top-1 retrieval relevance rate (dense score ≥ 0.45 and predicted/evidence intent agreement): **{checks['top1_retrieval_relevance_rate'] * 100:.2f}%**.",
        f"- Grounded reply rate: **{checks['grounded_reply_rate'] * 100:.2f}%**.",
        f"- Appropriate abstention rate (no safe reply, escalated): **{checks['appropriate_abstention_rate'] * 100:.2f}%**.",
        f"- Unsafe/context-bound reply rate: **{checks['unsafe_or_context_bound_reply_rate'] * 100:.2f}%**.",
        "",
        "## Design",
        "Replies are sanitized copies of an intent-aligned, cited dense-retrieval match when its score is at least 0.45 and it contains no source-specific status or timing claim. Otherwise the system declines to create a policy-bearing reply and escalates for insufficient grounding. The agent output includes the cited conversation ID, matched query, original reply, rank, and similarity score.",
        "",
        "## LLM and human judging",
        "The repository includes a dependency-injected LLM-as-judge adapter with an explicit five-dimension 1–5 rubric: groundedness, relevance, helpfulness, tone, and safety. No LLM score is reported in this run because no real LLM endpoint or human annotations were supplied. `human_llm_agreement` computes per-dimension Cohen’s kappa when matching real labels are provided.",
    ]), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_phase4_evaluation(), indent=2))
