"""
Unit and Integration Tests for Phase 3: Dense Semantic Retrieval and Upgraded Escalation Guardrails.
"""

import os
import json
import pytest
import numpy as np

from src.models.baseline_tfidf_lr import TfidfLogisticRegressionBaseline
from src.models.retriever import HistoricalSupportRetriever
from src.models.dense_retriever import DenseSupportRetriever
from src.models.agent import SupportPilotAgent


@pytest.fixture(scope="module")
def golden_data():
    path = "data/golden_eval/golden_eval_200.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def dense_retriever():
    path = "models/dense_retriever.joblib"
    assert os.path.exists(path), f"Dense retriever index missing: {path}"
    return DenseSupportRetriever.load(path)


@pytest.fixture(scope="module")
def tfidf_retriever():
    path = "models/historical_retriever.joblib"
    assert os.path.exists(path), f"TF-IDF retriever index missing: {path}"
    return HistoricalSupportRetriever.load(path)


@pytest.fixture(scope="module")
def clf_model():
    path = "models/tfidf_lr_intent_model.joblib"
    assert os.path.exists(path), f"Classifier model missing: {path}"
    return TfidfLogisticRegressionBaseline.load(path)


def test_dense_retriever_leakage_prevention(dense_retriever, golden_data):
    golden_ids = set(g["original_conversation_id"] for g in golden_data)
    indexed_ids = set(doc["conversation_id"] for doc in dense_retriever.corpus)
    overlap = indexed_ids.intersection(golden_ids)
    assert len(overlap) == 0, f"Leakage detected in dense retriever: {overlap}"


def test_dense_retriever_retrieval_output(dense_retriever):
    results = dense_retriever.retrieve("my iphone battery dies very quickly after updating to ios 11", top_k=3)
    assert len(results) == 3
    for r in results:
        assert 1 <= r["rank"] <= 3
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert len(r["historical_agent_reply"]) > 0
        assert len(r["matched_customer_query"]) > 0

    # Top similarity for battery drain query with dense retriever should be high (> 0.50)
    assert results[0]["similarity_score"] >= 0.50


def test_agent_with_dense_retriever(clf_model, dense_retriever):
    agent = SupportPilotAgent(classifier=clf_model, retriever=dense_retriever, confidence_threshold=0.12)
    output = agent.process_query("How can I adjust my screen brightness or reset auto-lock?")

    assert "customer_query" in output
    assert "predicted_intent" in output
    assert "confidence" in output
    assert "decision" in output
    assert "draft_reply" in output
    assert "retrieved_evidence" in output
    assert len(output["retrieved_evidence"]) == 3
    assert output["decision"] == "AUTO_HANDLE"


def test_phase3_escalation_guardrails(clf_model, dense_retriever):
    agent = SupportPilotAgent(classifier=clf_model, retriever=dense_retriever, confidence_threshold=0.12)

    # 1. Non-Latin script (Japanese)
    res_cjk = agent.process_query("11.1にあげたらTwitter少し見てるだけだし低電力モードなのに20パーへったぞ。")
    assert res_cjk["decision"] == "ESCALATE"
    assert res_cjk["escalation_reason"] == "non_english_query"

    # 2. Account Lockout (intent-independent)
    res_lock = agent.process_query("I am locked out of my iPad and the passcode doesn't work")
    assert res_lock["decision"] == "ESCALATE"
    assert res_lock["escalation_reason"] == "account_security_and_pii"

    # 3. Apple Pay / Payment failure (intent-independent)
    res_pay = agent.process_query("Apple Pay is saying Payment not completed when I try to check out")
    assert res_pay["decision"] == "ESCALATE"
    assert res_pay["escalation_reason"] == "financial_and_billing"


def test_retriever_baseline_preservation(clf_model, tfidf_retriever):
    # Verify TF-IDF retriever still works seamlessly as baseline
    agent = SupportPilotAgent(classifier=clf_model, retriever=tfidf_retriever, confidence_threshold=0.12)
    output = agent.process_query("my battery is draining")
    assert output["decision"] == "AUTO_HANDLE"
    assert len(output["retrieved_evidence"]) == 3
