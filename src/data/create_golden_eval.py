"""
Golden Evaluation Set Generator for SupportPilot AI.
Creates a reproducible, stratified 200-example benchmark set with ground-truth
intents, expected actions (AUTO_HANDLE vs ESCALATE), escalation reasons,
and difficulty tiers. Enforces strict train/evaluation data isolation.
"""

import json
import re
import os
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.analysis.eda_analysis import INTENT_PATTERNS, classify_by_rules

NON_ENGLISH_INDICATORS = [
    r"\bhola\b", r"\bnecesito\b", r"\bayuda\b", r"\bpor favor\b", r"\bactualizaci[oó]n\b",
    r"\bbater[ií]a\b", r"\btel[eé]fono\b", r"\bepıl\b", r"\byardımcı\b", r"\bnapcaz\b",
    r"\bkonusunda\b", r"\bmerci\b", r"\bbillete\b", r"\bbien\b", r"\bgracias\b"
]

HIGH_RISK_ESCALATION_KEYWORDS = {
    "hardware_repair_service": [r"\bcrack\b", r"\bcracked\b", r"\bbroken\b", r"\bwater damage\b", r"\bgenius bar\b", r"\brepair\b", r"\bshattered\b"],
    "app_store_billing": [r"\brefund\b", r"\bunauthorized\b", r"\bcharged twice\b", r"\bcredit card\b", r"\bdispute\b", r"\bcharge\b", r"\bbill\b"],
    "apple_id_icloud": [r"\blocked out\b", r"\bdisabled\b", r"\b2fa\b", r"\btwo-factor\b", r"\bhacked\b", r"\bpasscode\b", r"\bpassword\b", r"\bverification code\b"],
    "performance_freeze_crash": [r"\bbootloop\b", r"\bstuck on apple\b", r"\breboot loop\b", r"\bbrick\b", r"\bbricked\b", r"\bwon't turn on\b"],
    "abusive_or_legal": [r"\bsue\b", r"\blawyer\b", r"\blegal\b", r"\bworst company\b", r"\bfuck\b", r"\bshit\b", r"\bbullshit\b"]
}


def assess_expected_action_and_difficulty(record: Dict[str, Any], intent: str) -> Tuple[str, str, str]:
    """
    Determines expected action (AUTO_HANDLE vs ESCALATE), escalation reason,
    and difficulty level for a customer support interaction.
    """
    text = record["clean_customer_query"]
    text_lower = text.lower()
    
    # 1. Check for Non-English
    if any(re.search(p, text_lower) for p in NON_ENGLISH_INDICATORS):
        return "ESCALATE", "non_english_query", "edge_case"
        
    # 2. Check for abusive language or legal threats
    if any(re.search(p, text_lower) for p in HIGH_RISK_ESCALATION_KEYWORDS["abusive_or_legal"]):
        return "ESCALATE", "vague_or_abusive", "edge_case"
        
    # 3. Check for extremely vague / uninformative queries
    words = text.split()
    if len(words) < 5 or text_lower in ["help me", "i have a problem", "hey applesupport", "please help"]:
        return "ESCALATE", "vague_or_abusive", "edge_case"

    # 4. Intent-specific policy
    if intent == "hardware_repair_service":
        return "ESCALATE", "hardware_physical_damage", "straightforward" if any(re.search(p, text_lower) for p in [r"\bcrack\b", r"\bbroken\b"]) else "ambiguous"

    if intent == "app_store_billing":
        return "ESCALATE", "financial_and_billing", "straightforward"

    if intent == "apple_id_icloud":
        if any(re.search(p, text_lower) for p in HIGH_RISK_ESCALATION_KEYWORDS["apple_id_icloud"]):
            return "ESCALATE", "account_security_and_pii", "straightforward"
        # Standard iCloud sync / storage question can be auto-handled
        return "AUTO_HANDLE", "none", "ambiguous"

    if intent == "performance_freeze_crash":
        if any(re.search(p, text_lower) for p in HIGH_RISK_ESCALATION_KEYWORDS["performance_freeze_crash"]):
            return "ESCALATE", "unresolved_system_crash", "ambiguous"
        # Minor lag / app freezing can be auto-handled with force-restart instructions
        return "AUTO_HANDLE", "none", "straightforward"

    if intent == "general_inquiry_other":
        # Unclassified inquiries generally require human agent triage
        return "ESCALATE", "vague_or_abusive", "edge_case"

    # Standard technical intents default to AUTO_HANDLE if safe
    # Determine difficulty
    multi_intent_matches = 0
    for k, pats in INTENT_PATTERNS.items():
        if any(re.search(p, text_lower) for p in pats):
            multi_intent_matches += 1
            
    difficulty = "straightforward"
    if multi_intent_matches > 1:
        difficulty = "ambiguous"
    elif len(words) > 40 or "?" not in text:
        difficulty = "ambiguous"

    return "AUTO_HANDLE", "none", difficulty


def create_golden_evaluation_set(
    sample_file: str = "data/sample/applesupport_sample_1000.jsonl",
    output_dir: str = "data/golden_eval",
    train_dir: str = "data/training",
    target_golden_size: int = 200,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Extracts a stratified 200-sample golden set and isolates the remaining 800
    conversations into a clean training/retrieval pool.
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    with open(sample_file, "r", encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f]

    # Map each conversation to its intent
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for conv in conversations:
        intent = classify_by_rules(conv["clean_customer_query"])
        conv["ground_truth_intent"] = intent
        by_intent.setdefault(intent, []).append(conv)

    intents = sorted(by_intent.keys())
    # Determine target count per intent: ~16 or 17 per intent across 12 intents
    base_per_intent = target_golden_size // len(intents)
    remainder = target_golden_size % len(intents)

    golden_set: List[Dict[str, Any]] = []
    golden_ids = set()

    for idx, intent in enumerate(intents):
        pool = by_intent[intent]
        # Assign count
        k = base_per_intent + (1 if idx < remainder else 0)
        k = min(k, len(pool))
        
        # Sort or shuffle deterministically
        indices = np.random.choice(len(pool), size=k, replace=False)
        for i in indices:
            c = pool[i]
            action, reason, diff = assess_expected_action_and_difficulty(c, intent)
            golden_item = {
                "golden_id": f"GOLDEN_{len(golden_set)+1:03d}",
                "original_conversation_id": c["conversation_id"],
                "root_tweet_id": c["root_tweet_id"],
                "customer_initial_query": c["customer_initial_query"],
                "clean_customer_query": c["clean_customer_query"],
                "ground_truth_intent": intent,
                "expected_action": action,
                "escalation_reason": reason,
                "difficulty": diff,
                "historical_agent_reply": c["agent_first_reply"]
            }
            golden_set.append(golden_item)
            golden_ids.add(c["conversation_id"])

    # Build isolated training pool (conversations not in golden_ids)
    train_pool = [c for c in conversations if c["conversation_id"] not in golden_ids]

    # Save golden evaluation set
    golden_jsonl_path = os.path.join(output_dir, "golden_eval_200.jsonl")
    golden_csv_path = os.path.join(output_dir, "golden_eval_200.csv")

    with open(golden_jsonl_path, "w", encoding="utf-8") as f:
        for item in golden_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    pd.DataFrame(golden_set).to_csv(golden_csv_path, index=False, encoding="utf-8")

    # Save isolated training pool
    train_jsonl_path = os.path.join(train_dir, "train_pool_800.jsonl")
    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for item in train_pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Metrics on golden set
    intent_dist = Counter(item["ground_truth_intent"] for item in golden_set)
    action_dist = Counter(item["expected_action"] for item in golden_set)
    difficulty_dist = Counter(item["difficulty"] for item in golden_set)
    reason_dist = Counter(item["escalation_reason"] for item in golden_set)

    summary = {
        "total_golden_examples": len(golden_set),
        "total_training_examples": len(train_pool),
        "intent_distribution": dict(intent_dist),
        "action_distribution": dict(action_dist),
        "difficulty_distribution": dict(difficulty_dist),
        "escalation_reason_distribution": dict(reason_dist),
        "leak_check_passed": len(golden_ids.intersection(set(c["conversation_id"] for c in train_pool))) == 0
    }

    metadata_path = os.path.join(output_dir, "golden_eval_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = create_golden_evaluation_set()
    print("Golden Evaluation Set Created Successfully!")
    print(json.dumps(summary, indent=2))
