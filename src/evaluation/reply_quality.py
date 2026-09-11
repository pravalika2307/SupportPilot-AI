"""Reply-quality evaluation, rubric, and human/LLM agreement helpers.

The LLM judge is dependency-injected: a run records no LLM metric until a real
judge callable is supplied.  This avoids reporting invented model judgments.
"""

import math
import re
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from sklearn.metrics import cohen_kappa_score
from src.models.reply_generation import GroundedReplyGenerator


REPLY_QUALITY_RUBRIC = {
    "groundedness": "1: unsupported; 3: partly supported; 5: every support claim is backed by the cited historical reply.",
    "relevance": "1: unrelated; 3: addresses part of the issue; 5: directly addresses the customer issue.",
    "helpfulness": "1: no actionable next step; 3: some useful guidance; 5: clear, actionable, appropriate guidance.",
    "tone": "1: hostile or dismissive; 3: neutral; 5: empathetic and professional.",
    "safety": "1: unsafe or unsupported policy; 3: uncertain; 5: no unsupported policy and appropriate escalation.",
}


class LLMReplyJudge:
    """Adapter for any real LLM call that returns five integer rubric scores."""

    def __init__(self, judge_callable: Callable[[Dict[str, Any], Dict[str, str]], Dict[str, int]]) -> None:
        self.judge_callable = judge_callable

    def judge(self, agent_output: Dict[str, Any]) -> Dict[str, int]:
        scores = self.judge_callable(agent_output, REPLY_QUALITY_RUBRIC)
        normalized = dict(scores)
        if "relevance" in normalized and "correctness" not in normalized:
            normalized["correctness"] = normalized["relevance"]
        elif "correctness" in normalized and "relevance" not in normalized:
            normalized["relevance"] = normalized["correctness"]
        if set(REPLY_QUALITY_RUBRIC).difference(normalized):
            raise ValueError("LLM judge must score every explicit rubric dimension.")
        if any(not isinstance(score, int) or not 1 <= score <= 5 for score in scores.values()):
            raise ValueError("LLM judge scores must be integers from 1 to 5.")
        return scores


def deterministic_quality_checks(agent_outputs: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Auditable checks measured directly from agent output, not substitute LLM scores."""
    outputs = list(agent_outputs)
    count = len(outputs)
    if not count:
        return {"count": 0, "grounded_reply_rate": 0.0, "citation_integrity_rate": 0.0, "unsupported_policy_rate": 0.0}
    grounded = [out for out in outputs if out["grounding_status"] == "grounded"]
    top_relevant = [out for out in outputs if out["retrieved_evidence"] and out["retrieved_evidence"][0]["similarity_score"] >= out["grounding_similarity_threshold"] and out["retrieved_evidence"][0]["historical_intent"] == out["predicted_intent"]]
    valid_citations = [out for out in grounded if re.sub(r"@\w+\b", "", out["reply_evidence"]["historical_agent_reply"]).strip() == out["draft_reply"]]
    context_bound_candidates = sum(GroundedReplyGenerator.is_context_bound(item["historical_agent_reply"]) for out in outputs for item in out["retrieved_evidence"])
    unsafe_outputs = sum(bool(out["unsupported_policy_detected"]) for out in outputs)
    safe_abstentions = [out for out in outputs if out["grounding_status"] != "grounded" and out["decision"] == "ESCALATE"]
    return {
        "count": count,
        "top1_retrieval_relevance_rate": len(top_relevant) / count,
        "grounded_reply_rate": len(grounded) / count,
        "provenance_match_rate": len(valid_citations) / count,
        "appropriate_abstention_rate": len(safe_abstentions) / count,
        "unsafe_or_context_bound_reply_rate": unsafe_outputs / count,
        "context_bound_candidates_rejected": context_bound_candidates,
    }


def compute_dimension_kappa(human_scores: List[int], llm_scores: List[int]) -> Dict[str, Any]:
    """Calculates quadratic Cohen's kappa for a single dimension with explicit zero-variance handling."""
    if not human_scores or not llm_scores:
        return {"kappa": 0.0, "zero_variance": False, "note": "Empty score list"}

    all_same_h = len(set(human_scores)) <= 1
    all_same_l = len(set(llm_scores)) <= 1

    if all_same_h and all_same_l:
        if human_scores[0] == llm_scores[0]:
            return {
                "kappa": 1.0,
                "zero_variance": True,
                "note": "Zero variance: 100% concordance on constant rating",
            }
        return {
            "kappa": 0.0,
            "zero_variance": True,
            "note": "Zero variance: discordant constant ratings",
        }

    try:
        val = cohen_kappa_score(human_scores, llm_scores, labels=[1, 2, 3, 4, 5], weights="quadratic")
        if math.isnan(val) or np.isnan(val):
            val = 1.0 if human_scores == llm_scores else 0.0
            return {"kappa": val, "zero_variance": True, "note": "Zero chance variance resolved"}
        return {"kappa": round(float(val), 4), "zero_variance": False}
    except Exception as exc:
        val = 1.0 if human_scores == llm_scores else 0.0
        return {"kappa": val, "zero_variance": True, "note": f"Exception resolved: {exc}"}


def _index_and_validate_labels(labels: List[Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for row in labels:
        golden_id = row.get("golden_id")
        if not golden_id:
            raise ValueError(f"{source} annotation is missing golden_id")
        if golden_id in indexed:
            raise ValueError(f"Duplicate golden_id in {source} annotations: {golden_id}")

        # Support 'correctness' as an alias for 'relevance'
        normalized = dict(row)
        if "correctness" in normalized and "relevance" not in normalized:
            normalized["relevance"] = normalized["correctness"]
        elif "relevance" in normalized and "correctness" not in normalized:
            normalized["correctness"] = normalized["relevance"]

        missing = set(REPLY_QUALITY_RUBRIC).difference(normalized)
        if missing:
            raise ValueError(f"{source} annotation {golden_id} is missing rubric dimensions: {sorted(missing)}")
        if any(not isinstance(normalized[key], int) or not 1 <= normalized[key] <= 5 for key in REPLY_QUALITY_RUBRIC):
            raise ValueError(f"{source} annotation {golden_id} has an invalid 1-5 rubric score")
        indexed[golden_id] = normalized
    return indexed


def human_llm_agreement(human_labels: List[Dict[str, Any]], llm_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-dimension and overall quadratic Cohen's kappa for matched annotation IDs.

    Explicitly handles duplicate IDs (raises ValueError) and zero-variance ratings.
    """
    human_by_id = _index_and_validate_labels(human_labels, "human")
    llm_by_id = _index_and_validate_labels(llm_labels, "LLM")
    shared_ids = sorted(set(human_by_id).intersection(llm_by_id))
    if not shared_ids:
        return {
            "matched_examples": 0,
            "kappa_by_dimension": {},
            "kappa_details": {},
            "overall_macro_kappa": 0.0,
            "overall_pooled_kappa": 0.0,
            "kappa_weighting": "quadratic",
        }

    kappas = {}
    details = {}
    all_human_pooled = []
    all_llm_pooled = []

    for dimension in REPLY_QUALITY_RUBRIC:
        human_scores = [human_by_id[item][dimension] for item in shared_ids]
        llm_scores = [llm_by_id[item][dimension] for item in shared_ids]
        dim_res = compute_dimension_kappa(human_scores, llm_scores)
        kappas[dimension] = dim_res["kappa"]
        details[dimension] = dim_res
        all_human_pooled.extend(human_scores)
        all_llm_pooled.extend(llm_scores)

    macro_kappa = round(float(np.mean(list(kappas.values()))), 4)
    pooled_res = compute_dimension_kappa(all_human_pooled, all_llm_pooled)

    return {
        "matched_examples": len(shared_ids),
        "kappa_by_dimension": kappas,
        "kappa_details": details,
        "overall_macro_kappa": macro_kappa,
        "overall_pooled_kappa": pooled_res["kappa"],
        "kappa_weighting": "quadratic",
    }

