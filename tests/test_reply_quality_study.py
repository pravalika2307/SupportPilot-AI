"""Tests for SupportPilot AI Reply Quality Validation Study.

Verifies:
1. Reproducible 40-example stratified subset selection.
2. Complete evaluation inputs and draft reply generation.
3. Human annotation validation and dimension bounds checking (1-5).
4. Duplicate golden_id rejection in agreement calculation.
5. Explicit zero-variance handling in quadratic Cohen's kappa.
6. Real LLM judge provider behavior and zero-fabrication safety guarantee.
"""

import io
import json
import urllib.error
from unittest.mock import MagicMock
import pytest

from src.evaluation.llm_judge_provider import (
    ApiKeyMissingError,
    PROMPT_VERSION,
    RUBRIC_DIMENSIONS,
    RealLLMReplyJudge,
)
from src.evaluation.reply_quality import compute_dimension_kappa, human_llm_agreement
from src.evaluation.reply_quality_study import (
    run_llm_judge_study,
    select_reproducible_subset,
    validate_human_labels,
)


def test_reproducible_40_subset_selection():
    golden_records = [
        {"golden_id": f"GOLDEN_{i:03d}", "ground_truth_intent": intent, "expected_action": action, "clean_customer_query": f"Query {i}"}
        for i, (intent, action) in enumerate([
            ("app_store_billing", "AUTO_HANDLE"),
            ("app_store_billing", "AUTO_HANDLE"),
            ("app_store_billing", "ESCALATE"),
            ("software_update", "AUTO_HANDLE"),
            ("software_update", "AUTO_HANDLE"),
            ("software_update", "AUTO_HANDLE"),
            ("software_update", "ESCALATE"),
            ("battery_power", "AUTO_HANDLE"),
            ("battery_power", "AUTO_HANDLE"),
            ("battery_power", "ESCALATE"),
        ] * 20, 1)
    ]
    allocation = {
        "app_store_billing": 3,
        "software_update": 4,
        "battery_power": 3,
    }
    subset1 = select_reproducible_subset(golden_records, target_allocation=allocation, seed=42)
    subset2 = select_reproducible_subset(golden_records, target_allocation=allocation, seed=42)

    assert len(subset1) == 10
    assert [x["golden_id"] for x in subset1] == [x["golden_id"] for x in subset2]
    # Check that both actions are sampled
    actions = [x["expected_action"] for x in subset1]
    assert "AUTO_HANDLE" in actions
    assert "ESCALATE" in actions


def test_duplicate_golden_ids_rejected_in_agreement():
    h_labels = [
        {"golden_id": "G1", "groundedness": 5, "relevance": 5, "helpfulness": 5, "tone": 5, "safety": 5},
        {"golden_id": "G1", "groundedness": 4, "relevance": 4, "helpfulness": 4, "tone": 4, "safety": 4},
    ]
    l_labels = [
        {"golden_id": "G1", "groundedness": 5, "relevance": 5, "helpfulness": 5, "tone": 5, "safety": 5},
    ]
    with pytest.raises(ValueError, match="Duplicate golden_id"):
        human_llm_agreement(h_labels, l_labels)


def test_zero_variance_kappa_concordance_and_discordance():
    # 1. Constant concordant scores (e.g. all 5s)
    res_concordant = compute_dimension_kappa([5, 5, 5, 5], [5, 5, 5, 5])
    assert res_concordant["kappa"] == 1.0
    assert res_concordant["zero_variance"] is True

    # 2. Constant discordant scores (e.g. human gave all 5s, LLM gave all 4s)
    res_discordant = compute_dimension_kappa([5, 5, 5, 5], [4, 4, 4, 4])
    assert res_discordant["kappa"] == 0.0
    assert res_discordant["zero_variance"] is True

    # 3. Non-zero variance with high agreement
    res_variable = compute_dimension_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert res_variable["kappa"] == 1.0
    assert res_variable["zero_variance"] is False


def test_agreement_handles_correctness_alias_and_computes_overall():
    human = [
        {"golden_id": "G1", "groundedness": 5, "correctness": 5, "helpfulness": 5, "safety": 5, "tone": 5},
        {"golden_id": "G2", "groundedness": 4, "correctness": 4, "helpfulness": 4, "safety": 5, "tone": 4},
    ]
    llm = [
        {"golden_id": "G1", "groundedness": 5, "relevance": 5, "helpfulness": 5, "safety": 5, "tone": 5},
        {"golden_id": "G2", "groundedness": 4, "relevance": 4, "helpfulness": 4, "safety": 5, "tone": 4},
    ]
    agreement = human_llm_agreement(human, llm)
    assert agreement["matched_examples"] == 2
    assert agreement["overall_macro_kappa"] == 1.0
    assert agreement["overall_pooled_kappa"] == 1.0
    assert agreement["kappa_weighting"] == "quadratic"
    assert "safety" in agreement["kappa_details"]
    assert agreement["kappa_details"]["safety"]["zero_variance"] is True


def test_real_llm_judge_halts_without_fabrication_when_no_api_key():
    # When no key is provided and environment keys are unset
    judge = RealLLMReplyJudge(api_key=None, provider="none")
    judge.api_key = None
    assert judge.is_available() is False

    with pytest.raises(ApiKeyMissingError, match="No authentic LLM API key detected"):
        judge.evaluate_example({"golden_id": "G1", "customer_query": "test"})


def test_real_llm_judge_prompt_construction():
    judge = RealLLMReplyJudge(api_key="test_dummy_key", provider="google", model_name="gemini-1.5-flash")
    eval_input = {
        "golden_id": "GOLDEN_003",
        "customer_query": "Neither iPhone alarm went off this morning",
        "predicted_intent": "audio_sound",
        "decision": "AUTO_HANDLE",
        "grounding_status": "grounded",
        "draft_reply": "Hello. We are here to help.",
        "retrieved_evidence": [
            {
                "rank": 1,
                "similarity_score": 0.5464,
                "matched_customer_query": "My alarm did not work",
                "historical_agent_reply": "Hello. We are here to help.",
            }
        ],
    }
    prompt = judge.build_prompt(eval_input)
    assert "GOLDEN_003" in prompt
    assert "Neither iPhone alarm went off" in prompt
    assert "audio_sound" in prompt
    assert "groundedness" in prompt
    assert "safety" in prompt
    assert judge.prompt_version == PROMPT_VERSION
    assert judge.temperature == 0.0


def test_human_labels_validation_detects_incomplete_and_invalid_entries(tmp_path):
    labels_file = tmp_path / "test_labels.jsonl"

    # Test 1: Incomplete count
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"golden_id": "G1", "groundedness": 5, "correctness": 5, "helpfulness": 5, "safety": 5, "tone": 5}) + "\n")
    valid, msg, _ = validate_human_labels(str(labels_file))
    assert valid is False
    assert "Expected exactly 40 records" in msg

    # Test 2: Out-of-bounds score (e.g. 6)
    valid_40 = []
    for i in range(1, 41):
        valid_40.append({
            "golden_id": f"GOLDEN_{i:03d}",
            "groundedness": 5,
            "correctness": 5,
            "helpfulness": 5,
            "safety": 5,
            "tone": 6 if i == 1 else 5,  # tone = 6 for record 1
        })
    with open(labels_file, "w", encoding="utf-8") as f:
        for r in valid_40:
            f.write(json.dumps(r) + "\n")
    valid, msg, _ = validate_human_labels(str(labels_file))
    assert valid is False
    assert "invalid score for 'tone': 6" in msg


def test_run_llm_judge_study_resumes_and_skips_existing_valid_records(tmp_path):
    inputs_file = tmp_path / "study_inputs.jsonl"
    output_file = tmp_path / "llm_output.jsonl"

    study_inputs = [
        {"golden_id": "GOLDEN_001", "customer_query": "query 1", "draft_reply": "reply 1"},
        {"golden_id": "GOLDEN_002", "customer_query": "query 2", "draft_reply": "reply 2"},
        {"golden_id": "GOLDEN_003", "customer_query": "query 3", "draft_reply": "reply 3"},
    ]
    with open(inputs_file, "w", encoding="utf-8") as f:
        for item in study_inputs:
            f.write(json.dumps(item) + "\n")

    # Simulate GOLDEN_001 already evaluated and persisted
    existing_record = {
        "golden_id": "GOLDEN_001",
        "provider": "google",
        "model_name": "gemini-3.5-flash",
        "scores": {"groundedness": 5, "correctness": 5, "helpfulness": 5, "safety": 5, "tone": 5},
        "rationale": "Previously completed.",
    }
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing_record) + "\n")

    # Mock judge that evaluates remaining items
    mock_judge = MagicMock()
    mock_judge.is_available.return_value = True
    mock_judge.provider = "google"
    mock_judge.model_name = "gemini-3.5-flash"
    mock_judge.max_retries = 5

    def fake_eval(item):
        return {
            "golden_id": item["golden_id"],
            "provider": "google",
            "model_name": "gemini-3.5-flash",
            "scores": {"groundedness": 4, "correctness": 4, "helpfulness": 4, "safety": 5, "tone": 5},
            "rationale": f"Evaluated {item['golden_id']}",
        }

    mock_judge.evaluate_example.side_effect = fake_eval

    success, msg, records = run_llm_judge_study(
        study_inputs_path=str(inputs_file),
        output_path=str(output_file),
        judge=mock_judge,
        request_delay=0.0,
    )

    assert success is True
    # mock_judge should only be called for GOLDEN_002 and GOLDEN_003 (GOLDEN_001 was skipped)
    assert mock_judge.evaluate_example.call_count == 2
    called_ids = [call.args[0]["golden_id"] for call in mock_judge.evaluate_example.call_args_list]
    assert called_ids == ["GOLDEN_002", "GOLDEN_003"]

    # File on disk must contain all 3 records, with GOLDEN_001 preserved
    with open(output_file, "r", encoding="utf-8") as f:
        disk_records = [json.loads(line) for line in f if line.strip()]
    assert len(disk_records) == 3
    assert disk_records[0]["golden_id"] == "GOLDEN_001"
    assert disk_records[0]["rationale"] == "Previously completed."
    assert disk_records[1]["golden_id"] == "GOLDEN_002"
    assert disk_records[2]["golden_id"] == "GOLDEN_003"


def test_run_llm_judge_study_all_completed_skips_all_and_makes_no_calls(tmp_path):
    inputs_file = tmp_path / "study_inputs.jsonl"
    output_file = tmp_path / "llm_output.jsonl"

    study_inputs = [
        {"golden_id": "GOLDEN_001", "customer_query": "query 1", "draft_reply": "reply 1"},
    ]
    with open(inputs_file, "w", encoding="utf-8") as f:
        for item in study_inputs:
            f.write(json.dumps(item) + "\n")

    existing_record = {
        "golden_id": "GOLDEN_001",
        "provider": "google",
        "model_name": "gemini-3.5-flash",
        "scores": {"groundedness": 5, "correctness": 5, "helpfulness": 5, "safety": 5, "tone": 5},
        "rationale": "Already done.",
    }
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing_record) + "\n")

    mock_judge = MagicMock()
    mock_judge.is_available.return_value = True
    mock_judge.provider = "google"
    mock_judge.model_name = "gemini-3.5-flash"
    mock_judge.max_retries = 5

    success, msg, records = run_llm_judge_study(
        study_inputs_path=str(inputs_file),
        output_path=str(output_file),
        judge=mock_judge,
        request_delay=0.0,
    )

    assert success is True
    assert "already completed" in msg.lower()
    assert mock_judge.evaluate_example.call_count == 0
    assert len(records) == 1
    assert records[0]["golden_id"] == "GOLDEN_001"


def test_real_llm_judge_backoff_on_429_retry_success(monkeypatch):
    judge = RealLLMReplyJudge(
        api_key="test-gemini-key",
        provider="google",
        model_name="gemini-3.5-flash",
        max_retries=3,
        initial_backoff=0.01,
        max_backoff=0.05,
    )

    attempts = 0
    sleeps = []

    def mock_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("time.sleep", mock_sleep)

    valid_response_body = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "groundedness": 5,
                        "correctness": 5,
                        "helpfulness": 5,
                        "safety": 5,
                        "tone": 5,
                        "rationale": "Passed after backoff.",
                    })
                }]
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def read(self):
            return valid_response_body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=30):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            # Raise HTTP 429 for the first 2 attempts
            raise urllib.error.HTTPError(
                url="https://generativelanguage.googleapis.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "0.02"},
                fp=io.BytesIO(b'{"error": "rate limited"}'),
            )
        return MockHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    eval_input = {
        "golden_id": "GOLDEN_112",
        "customer_query": "Battery dies in 2 hours",
        "predicted_intent": "battery_power",
        "decision": "AUTO_HANDLE",
        "grounding_status": "grounded",
        "draft_reply": "Check battery health in Settings.",
    }

    result = judge.evaluate_example(eval_input)

    assert attempts == 3
    assert len(sleeps) == 2
    assert result["golden_id"] == "GOLDEN_112"
    assert result["scores"]["groundedness"] == 5
    assert result["rationale"] == "Passed after backoff."


def test_real_llm_judge_backoff_exhausts_and_never_fabricates(monkeypatch):
    judge = RealLLMReplyJudge(
        api_key="test-gemini-key",
        provider="google",
        model_name="gemini-3.5-flash",
        max_retries=3,
        initial_backoff=0.01,
        max_backoff=0.05,
    )

    attempts = 0
    monkeypatch.setattr("time.sleep", lambda s: None)

    def mock_urlopen(req, timeout=30):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error": "rate limited"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    eval_input = {
        "golden_id": "GOLDEN_112",
        "customer_query": "Battery dies in 2 hours",
        "predicted_intent": "battery_power",
        "decision": "AUTO_HANDLE",
        "grounding_status": "grounded",
        "draft_reply": "Check battery health in Settings.",
    }

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        judge.evaluate_example(eval_input)

    assert exc_info.value.code == 429
    assert attempts == 3  # Did not retry indefinitely, respected max_retries


def test_run_llm_judge_study_preserves_completed_records_on_failure(tmp_path):
    inputs_file = tmp_path / "study_inputs.jsonl"
    output_file = tmp_path / "llm_output.jsonl"

    study_inputs = [
        {"golden_id": "GOLDEN_001", "customer_query": "q1", "draft_reply": "r1"},
        {"golden_id": "GOLDEN_002", "customer_query": "q2", "draft_reply": "r2"},
        {"golden_id": "GOLDEN_003", "customer_query": "q3", "draft_reply": "r3"},
    ]
    with open(inputs_file, "w", encoding="utf-8") as f:
        for item in study_inputs:
            f.write(json.dumps(item) + "\n")

    # GOLDEN_001 already completed
    existing_record = {
        "golden_id": "GOLDEN_001",
        "provider": "google",
        "model_name": "gemini-3.5-flash",
        "scores": {"groundedness": 5, "correctness": 5, "helpfulness": 5, "safety": 5, "tone": 5},
        "rationale": "Initial completed.",
    }
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing_record) + "\n")

    mock_judge = MagicMock()
    mock_judge.is_available.return_value = True
    mock_judge.provider = "google"
    mock_judge.model_name = "gemini-3.5-flash"
    mock_judge.max_retries = 3

    def eval_with_failure(item):
        if item["golden_id"] == "GOLDEN_002":
            return {
                "golden_id": "GOLDEN_002",
                "provider": "google",
                "model_name": "gemini-3.5-flash",
                "scores": {"groundedness": 4, "correctness": 4, "helpfulness": 4, "safety": 4, "tone": 4},
                "rationale": "Newly evaluated.",
            }
        raise urllib.error.HTTPError("url", 429, "Rate limit", None, io.BytesIO(b""))

    mock_judge.evaluate_example.side_effect = eval_with_failure

    with pytest.raises(urllib.error.HTTPError):
        run_llm_judge_study(
            study_inputs_path=str(inputs_file),
            output_path=str(output_file),
            judge=mock_judge,
            request_delay=0.0,
        )

    # Verify that GOLDEN_001 and GOLDEN_002 are persisted safely on disk, with NO fabricated GOLDEN_003
    with open(output_file, "r", encoding="utf-8") as f:
        disk_records = [json.loads(line) for line in f if line.strip()]

    assert len(disk_records) == 2
    assert disk_records[0]["golden_id"] == "GOLDEN_001"
    assert disk_records[1]["golden_id"] == "GOLDEN_002"
    assert all(r["golden_id"] != "GOLDEN_003" for r in disk_records)

