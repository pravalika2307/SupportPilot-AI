"""
Unit and Integration Tests for SupportPilot AI Baselines, Retriever, and Agent Pipeline.
Verifies:
1. Majority Class Baseline correctness.
2. TF-IDF + Logistic Regression classification, probabilities, and serialization.
3. Retrieval engine correctness and strict leakage isolation.
4. Golden evaluation dataset schema and integrity.
5. End-to-end SupportPilotAgent decision policies and draft replies.
"""

import os
import json
import pytest
import numpy as np

from src.models.baseline_majority import MajorityClassBaseline
from src.models.baseline_tfidf_lr import TfidfLogisticRegressionBaseline
from src.models.retriever import HistoricalSupportRetriever
from src.models.agent import SupportPilotAgent


@pytest.fixture(scope="module")
def golden_data():
    path = "data/golden_eval/golden_eval_200.jsonl"
    assert os.path.exists(path), f"Golden evaluation file missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def train_data():
    path = "data/training/train_pool_800.jsonl"
    assert os.path.exists(path), f"Training pool file missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def trained_agent(train_data, golden_data):
    X_train = [t["clean_customer_query"] for t in train_data]
    y_train = [t["ground_truth_intent"] for t in train_data]
    golden_ids = set(g["original_conversation_id"] for g in golden_data)

    clf = TfidfLogisticRegressionBaseline(ngram_range=(1, 2), max_features=3000, C=1.0)
    clf.fit(X_train, y_train)

    retriever = HistoricalSupportRetriever(top_k=3, min_similarity_threshold=0.05)
    retriever.build_index(train_data, forbidden_ids=golden_ids)

    return SupportPilotAgent(classifier=clf, retriever=retriever, confidence_threshold=0.12)


# --- 1. Majority Baseline Tests ---

def test_majority_baseline():
    X = ["query 1", "query 2", "query 3", "query 4"]
    y = ["battery_power", "battery_power", "software_update", "battery_power"]
    base = MajorityClassBaseline().fit(X, y)

    preds = base.predict(["test query A", "test query B"])
    assert preds == ["battery_power", "battery_power"]
    
    confs = base.predict_with_confidence(["test query"])
    assert confs[0]["intent"] == "battery_power"
    assert round(confs[0]["confidence"], 2) == 0.75


# --- 2. TF-IDF + Logistic Regression Baseline Tests ---

def test_tfidf_lr_baseline(train_data):
    X = [t["clean_customer_query"] for t in train_data[:100]]
    y = [t["ground_truth_intent"] for t in train_data[:100]]
    
    model = TfidfLogisticRegressionBaseline(max_features=500, C=1.0)
    model.fit(X, y)
    
    preds = model.predict(["my screen is completely cracked and broken"])
    assert len(preds) == 1
    
    conf_res = model.predict_with_confidence(["battery draining fast"])
    assert "intent" in conf_res[0]
    assert 0.0 <= conf_res[0]["confidence"] <= 1.0
    assert len(conf_res[0]["probabilities"]) == len(model.classes_)
    
    top_feats = model.get_top_features_per_intent(top_n=3)
    assert len(top_feats) > 0

    # Test persistence
    tmp_path = "models/test_model.joblib"
    model.save(tmp_path)
    assert os.path.exists(tmp_path)
    loaded = TfidfLogisticRegressionBaseline.load(tmp_path)
    assert loaded.classes_ == model.classes_
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# --- 3. Retrieval Engine Tests ---

def test_retriever_leakage_prevention(train_data, golden_data):
    golden_ids = set(g["original_conversation_id"] for g in golden_data)
    retriever = HistoricalSupportRetriever(top_k=3)
    retriever.build_index(train_data, forbidden_ids=golden_ids)

    # Verify no golden ID made it into index corpus
    indexed_ids = set(doc["conversation_id"] for doc in retriever.corpus)
    assert len(indexed_ids.intersection(golden_ids)) == 0

    # Verify retrieval structure
    results = retriever.retrieve("how do i restart my frozen iphone?", top_k=3)
    assert len(results) == 3
    for r in results:
        assert "rank" in r
        assert "similarity_score" in r
        assert "historical_agent_reply" in r
        assert r["conversation_id"] not in golden_ids


# --- 4. Golden Set Integrity Tests ---

def test_golden_set_integrity(golden_data, train_data):
    assert len(golden_data) == 200, f"Expected 200 golden examples, got {len(golden_data)}"
    golden_ids = set(g["original_conversation_id"] for g in golden_data)
    train_ids = set(t["conversation_id"] for t in train_data)

    # Zero leakage
    assert len(golden_ids.intersection(train_ids)) == 0

    # Stratification: all 12 intents present
    intents = set(g["ground_truth_intent"] for g in golden_data)
    assert len(intents) == 12

    # Required fields
    required_keys = {
        "golden_id", "original_conversation_id", "clean_customer_query",
        "ground_truth_intent", "expected_action", "escalation_reason", "difficulty"
    }
    for g in golden_data:
        assert required_keys.issubset(g.keys())
        assert g["expected_action"] in ["AUTO_HANDLE", "ESCALATE"]
        assert g["difficulty"] in ["straightforward", "ambiguous", "edge_case"]


# --- 5. SupportPilotAgent End-to-End Decision Tests ---

def test_agent_hardware_escalation(trained_agent):
    result = trained_agent.process_query("I dropped my iPhone and the screen is completely cracked and broken")
    assert result["decision"] == "ESCALATE"
    assert result["escalation_reason"] == "hardware_physical_damage"
    assert result["grounding_status"] in {"grounded", "insufficient_evidence"}
    assert result["unsupported_policy_detected"] is False


def test_agent_billing_escalation(trained_agent):
    result = trained_agent.process_query("I was charged twice for my subscription and I demand a full refund")
    assert result["decision"] == "ESCALATE"
    assert result["escalation_reason"] == "financial_and_billing"
    assert result["grounding_status"] in {"grounded", "insufficient_evidence"}
    assert result["unsupported_policy_detected"] is False


def test_agent_non_english_escalation(trained_agent):
    result = trained_agent.process_query("Hola necesito ayuda urgente por favor con mi teléfono")
    assert result["decision"] == "ESCALATE"
    assert result["escalation_reason"] == "non_english_query"


def test_agent_abusive_escalation(trained_agent):
    result = trained_agent.process_query("Your support is pure bullshit I am calling my lawyer to sue you")
    assert result["decision"] == "ESCALATE"
    assert result["escalation_reason"] == "vague_or_abusive"


def test_agent_standard_autohandle(trained_agent):
    result = trained_agent.process_query("My battery is draining really quickly after updating, what settings can I change?")
    assert result["decision"] == "AUTO_HANDLE"
    assert result["escalation_reason"] == "none"
    assert len(result["retrieved_evidence"]) == 3
    assert result["grounding_status"] == "grounded"
    assert result["reply_evidence"] is not None
    assert result["unsupported_policy_detected"] is False
