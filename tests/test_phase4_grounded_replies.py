"""Phase 4 tests: cited replies, no policy fallback, rubric and agreement plumbing."""

from src.evaluation.reply_quality import LLMReplyJudge, deterministic_quality_checks, human_llm_agreement
from src.models.reply_generation import GroundedReplyGenerator


def test_grounded_generator_cites_exact_historical_reply():
    evidence = [{"rank": 1, "similarity_score": 0.9, "conversation_id": "case-1", "matched_customer_query": "battery drains", "historical_agent_reply": "@customer Check Settings > Battery."}]
    evidence[0]["historical_intent"] = "battery_power"
    result = GroundedReplyGenerator(0.2).generate(evidence, "battery_power")
    assert result["grounding_status"] == "grounded"
    assert result["text"] == "Check Settings > Battery."
    assert result["citation"]["conversation_id"] == "case-1"
    assert result["unsupported_policy_detected"] is False


def test_generator_refuses_unsupported_fallback():
    result = GroundedReplyGenerator(0.2).generate([], "battery_power")
    assert result["grounding_status"] == "insufficient_evidence"
    assert result["citation"] is None
    assert "verified support example" in result["text"]


def test_quality_checks_and_real_label_agreement():
    output = {"grounding_status": "grounded", "reply_evidence": {"historical_agent_reply": "@customer Try restarting."}, "draft_reply": "Try restarting.", "unsupported_policy_detected": False, "decision": "AUTO_HANDLE", "predicted_intent": "battery_power", "grounding_similarity_threshold": 0.2, "retrieved_evidence": [{"similarity_score": 0.9, "historical_intent": "battery_power", "historical_agent_reply": "@customer Try restarting."}]}
    assert deterministic_quality_checks([output])["provenance_match_rate"] == 1.0
    labels = [
        {"golden_id": "G1", "groundedness": 5, "relevance": 4, "helpfulness": 4, "tone": 5, "safety": 5},
        {"golden_id": "G2", "groundedness": 3, "relevance": 3, "helpfulness": 2, "tone": 4, "safety": 4},
    ]
    agreement = human_llm_agreement(labels, labels)
    assert agreement["matched_examples"] == 2
    assert set(agreement["kappa_by_dimension"]) == {"groundedness", "relevance", "helpfulness", "tone", "safety"}
    assert agreement["kappa_weighting"] == "quadratic"


def test_generator_requires_intent_agreement_and_rejects_context_bound_reply():
    mismatch = [{"rank": 1, "similarity_score": 0.9, "conversation_id": "case-1", "matched_customer_query": "battery", "historical_agent_reply": "@customer Try restarting.", "historical_intent": "software_update"}]
    assert "intent_mismatch" in GroundedReplyGenerator(0.2).generate(mismatch, "battery_power")["rejection_reasons"]
    context = [{"rank": 1, "similarity_score": 0.9, "conversation_id": "case-2", "matched_customer_query": "battery", "historical_agent_reply": "@customer We received your DM and will be responding momentarily.", "historical_intent": "battery_power"}]
    assert "context_bound" in GroundedReplyGenerator(0.2).generate(context, "battery_power")["rejection_reasons"]


def test_duplicate_annotation_ids_are_rejected():
    import pytest
    duplicate = [{"golden_id": "G1", "groundedness": 5, "relevance": 5, "helpfulness": 5, "tone": 5, "safety": 5}] * 2
    with pytest.raises(ValueError, match="Duplicate golden_id"):
        human_llm_agreement(duplicate, [])


def test_llm_judge_requires_complete_rubric():
    judge = LLMReplyJudge(lambda output, rubric: {dimension: 5 for dimension in rubric})
    assert judge.judge({})["safety"] == 5
