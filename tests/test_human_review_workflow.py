"""
Tests for SupportPilot AI Golden Set Human Review Workflow and Presentation Tool.
"""

import os
import json
import pytest

from src.data.human_review_tool import (
    generate_recommendation,
    format_review_presentation,
    export_review_queue,
    load_jsonl,
    VALID_INTENTS,
    VALID_ACTIONS,
    VALID_ESCALATION_REASONS
)


def test_generate_recommendation_discrepancy_correction():
    sample = {
        "golden_id": "GOLDEN_002",
        "clean_customer_query": "Ever since I upgraded to High Sierra, my media controls on the keyboard don't work with iTunes. Please help or fix it @AppleSupport",
        "ground_truth_intent": "app_store_billing",
        "expected_action": "ESCALATE",
        "escalation_reason": "financial_and_billing",
        "difficulty": "straightforward"
    }
    rec = generate_recommendation(sample)
    assert rec["golden_id"] == "GOLDEN_002"
    assert rec["recommended_intent"] == "display_touch_keyboard"
    assert rec["recommended_action"] == "AUTO_HANDLE"
    assert rec["recommended_escalation_reason"] == "none"
    assert rec["is_changed_from_heuristic"] is True
    assert "keyboard" in rec["recommendation_rationale"].lower()


def test_generate_recommendation_alarm_consequential_loss():
    sample = {
        "golden_id": "GOLDEN_003",
        "clean_customer_query": "Neither iPhone alarm went off this morning, had to make a 5:30 AM flight. The 2am alarm on one phone finally went off at 5:40 on it's own. United just charged $640 per person to get on a later flight today @AppleSupport",
        "ground_truth_intent": "app_store_billing",
        "expected_action": "ESCALATE",
        "escalation_reason": "financial_and_billing",
        "difficulty": "straightforward"
    }
    rec = generate_recommendation(sample)
    assert rec["golden_id"] == "GOLDEN_003"
    assert rec["recommended_intent"] == "audio_sound"
    assert rec["recommended_action"] == "ESCALATE"
    assert rec["recommended_escalation_reason"] == "financial_and_billing"
    assert rec["difficulty"] == "edge_case"
    assert "financial loss" in rec["recommendation_rationale"].lower()


def test_format_review_presentation():
    sample = {
        "golden_id": "TEST_001",
        "clean_customer_query": "How do I turn on night shift?",
        "ground_truth_intent": "general_inquiry_other",
        "expected_action": "AUTO_HANDLE",
        "escalation_reason": "none",
        "difficulty": "straightforward"
    }
    rec = generate_recommendation(sample)
    presentation = format_review_presentation(rec, index=1, total=200)

    assert "TEST_001" in presentation
    assert "How do I turn on night shift?" in presentation
    assert "Current Heuristic Label" in presentation
    assert "Recommended Label" in presentation
    assert "Recommended Intent" in presentation
    assert "Recommended Action" in presentation


def test_reviewed_dataset_integrity_and_isolation():
    reviewed_path = "data/golden_eval/human_verified_golden_200.jsonl"
    train_path = "data/training/train_pool_800.jsonl"
    orig_path = "data/golden_eval/golden_eval_200.jsonl"

    assert os.path.exists(reviewed_path)
    assert os.path.exists(orig_path)
    assert os.path.exists(train_path)

    reviewed = load_jsonl(reviewed_path)
    orig = load_jsonl(orig_path)
    train = load_jsonl(train_path)

    assert len(reviewed) == 200
    assert len(orig) == 200

    # Ensure zero leakage against training pool
    reviewed_conv_ids = set(r["original_conversation_id"] for r in reviewed)
    train_conv_ids = set(t["conversation_id"] for t in train)
    assert len(reviewed_conv_ids.intersection(train_conv_ids)) == 0

    # Validate taxonomy compliance
    for r in reviewed:
        assert r["ground_truth_intent"] in VALID_INTENTS
        assert r["expected_action"] in VALID_ACTIONS
        assert r["escalation_reason"] in VALID_ESCALATION_REASONS


def test_save_human_review_decisions_validation(tmp_path):
    from src.data.human_review_tool import save_human_review_decisions

    test_golden = tmp_path / "test_golden.jsonl"
    test_output = tmp_path / "test_verified.jsonl"
    test_audit = tmp_path / "test_audit.json"

    records = [
        {
            "golden_id": "TEST_001",
            "original_conversation_id": "c1",
            "root_tweet_id": "t1",
            "clean_customer_query": "billing issue",
            "ground_truth_intent": "app_store_billing",
            "expected_action": "ESCALATE",
            "escalation_reason": "financial_and_billing",
            "difficulty": "straightforward"
        }
    ]
    with open(test_golden, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Valid save
    decisions = [
        {
            "golden_id": "TEST_001",
            "ground_truth_intent": "app_store_billing",
            "expected_action": "ESCALATE",
            "escalation_reason": "financial_and_billing",
            "difficulty": "straightforward"
        }
    ]
    summary = save_human_review_decisions(
        decisions=decisions,
        annotator_id="test_human",
        input_golden_path=str(test_golden),
        output_path=str(test_output),
        audit_path=str(test_audit)
    )

    assert summary["reviewer_status"]["actually_human_reviewed"] is True
    assert summary["reviewer_status"]["reviewer_id"] == "test_human"
    assert os.path.exists(test_output)
    saved = load_jsonl(str(test_output))
    assert len(saved) == 1
    assert saved[0]["human_verified"] is True
    assert saved[0]["annotator_id"] == "test_human"

    # Invalid intent should raise ValueError
    invalid_decisions = [
        {
            "golden_id": "TEST_001",
            "ground_truth_intent": "invalid_intent_xyz",
            "expected_action": "ESCALATE",
            "escalation_reason": "financial_and_billing"
        }
    ]
    with pytest.raises(ValueError, match="Invalid intent"):
        save_human_review_decisions(
            decisions=invalid_decisions,
            annotator_id="test_human",
            input_golden_path=str(test_golden),
            output_path=str(test_output),
            audit_path=str(test_audit)
        )

