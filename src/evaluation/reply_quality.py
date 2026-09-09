"""Reply-quality evaluation, rubric, and human/LLM agreement helpers.

The LLM judge is dependency-injected: a run records no LLM metric until a real
judge callable is supplied.  This avoids reporting invented model judgments.
"""

import re
from typing import Any, Callable, Dict, Iterable, List

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
        if set(scores) != set(REPLY_QUALITY_RUBRIC):
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


def _index_and_validate_labels(labels: List[Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for row in labels:
        golden_id = row.get("golden_id")
        if not golden_id:
            raise ValueError(f"{source} annotation is missing golden_id")
        if golden_id in indexed:
            raise ValueError(f"Duplicate golden_id in {source} annotations: {golden_id}")
        missing = set(REPLY_QUALITY_RUBRIC).difference(row)
        if missing:
            raise ValueError(f"{source} annotation {golden_id} is missing rubric dimensions: {sorted(missing)}")
        if any(not isinstance(row[key], int) or not 1 <= row[key] <= 5 for key in REPLY_QUALITY_RUBRIC):
            raise ValueError(f"{source} annotation {golden_id} has an invalid 1-5 rubric score")
        indexed[golden_id] = row
    return indexed


def human_llm_agreement(human_labels: List[Dict[str, Any]], llm_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-dimension Cohen's kappa for matched, real annotation IDs."""
    human_by_id = _index_and_validate_labels(human_labels, "human")
    llm_by_id = _index_and_validate_labels(llm_labels, "LLM")
    shared_ids = sorted(set(human_by_id).intersection(llm_by_id))
    if not shared_ids:
        return {"matched_examples": 0, "kappa_by_dimension": {}}
    kappas = {}
    for dimension in REPLY_QUALITY_RUBRIC:
        human = [human_by_id[item][dimension] for item in shared_ids]
        llm = [llm_by_id[item][dimension] for item in shared_ids]
        kappas[dimension] = float(cohen_kappa_score(human, llm, weights="quadratic"))
    return {"matched_examples": len(shared_ids), "kappa_by_dimension": kappas, "kappa_weighting": "quadratic"}
