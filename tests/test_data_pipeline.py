"""
Unit tests for data extraction, thread reconstruction, and sample integrity.
"""

import json
from pathlib import Path
import pytest

from src.data.extract_applesupport import clean_tweet_text
from src.data.thread_reconstructor import ThreadReconstructor, clean_text_for_modeling
from src.analysis.eda_analysis import classify_by_rules


def test_clean_tweet_text():
    raw = "@AppleSupport my sound is &amp; was broken &gt; after update &lt;help&gt;   "
    cleaned = clean_tweet_text(raw)
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert ">" in cleaned
    assert "<help>" in cleaned
    assert cleaned.endswith("<help>")


def test_clean_text_for_modeling():
    text = "@AppleSupport @115854 My battery is dying https://t.co/xyz123   fast"
    cleaned = clean_text_for_modeling(text)
    assert not cleaned.startswith("@")
    assert "https://" not in cleaned
    assert cleaned == "My battery is dying fast"


def test_thread_reconstructor_single_turn_reply():
    reconstructor = ThreadReconstructor()

    # 1. Customer root tweet
    cust_tweet = {
        "tweet_id": "101",
        "author_id": "user_42",
        "inbound": True,
        "text": "@AppleSupport My iPhone 7 keeps freezing whenever I open Safari",
        "created_at": "Wed Nov 01 10:00:00 +0000 2017",
        "in_response_to_tweet_id": "",
        "response_tweet_id": "102"
    }
    # 2. AppleSupport reply
    apple_tweet = {
        "tweet_id": "102",
        "author_id": "AppleSupport",
        "inbound": False,
        "text": "@user_42 We can help with that. Which iOS version are you running?",
        "created_at": "Wed Nov 01 10:05:00 +0000 2017",
        "in_response_to_tweet_id": "101",
        "response_tweet_id": ""
    }

    reconstructor.add_tweet(cust_tweet)
    reconstructor.add_tweet(apple_tweet)

    convs = reconstructor.reconstruct_all_conversations(min_query_chars=10)
    assert len(convs) == 1
    conv = convs[0]
    assert conv["root_tweet_id"] == "101"
    assert conv["customer_id"] == "user_42"
    assert conv["num_turns"] == 2
    assert "freezing" in conv["customer_initial_query"]
    assert "We can help" in conv["agent_first_reply"]


def test_thread_reconstructor_multi_turn():
    reconstructor = ThreadReconstructor()

    t1 = {
        "tweet_id": "201",
        "author_id": "cust_1",
        "inbound": True,
        "text": "@AppleSupport Battery draining fast on iOS 11",
        "in_response_to_tweet_id": "",
        "response_tweet_id": "202"
    }
    t2 = {
        "tweet_id": "202",
        "author_id": "AppleSupport",
        "inbound": False,
        "text": "@cust_1 Let's take a look. Does it drain in Low Power Mode?",
        "in_response_to_tweet_id": "201",
        "response_tweet_id": "203"
    }
    t3 = {
        "tweet_id": "203",
        "author_id": "cust_1",
        "inbound": True,
        "text": "@AppleSupport Yes even in Low Power Mode!",
        "in_response_to_tweet_id": "202",
        "response_tweet_id": "204"
    }
    t4 = {
        "tweet_id": "204",
        "author_id": "AppleSupport",
        "inbound": False,
        "text": "@cust_1 Send us a DM so we can run a remote battery diagnostic.",
        "in_response_to_tweet_id": "203",
        "response_tweet_id": ""
    }

    reconstructor.load_tweets([t1, t2, t3, t4])
    convs = reconstructor.reconstruct_all_conversations(min_query_chars=10)
    assert len(convs) == 1
    assert convs[0]["num_turns"] == 4
    assert [t["speaker"] for t in convs[0]["turns"]] == ["customer", "agent", "customer", "agent"]


def test_classify_by_rules_taxonomy():
    assert classify_by_rules("My battery drops from 90% to 10% in an hour") == "battery_power"
    assert classify_by_rules("Unable to verify iOS 11 software update on iPhone") == "software_update"
    assert classify_by_rules("Locked out of my Apple ID and cannot reset password") == "apple_id_icloud"
    assert classify_by_rules("Cracked screen repair cost at Apple Store Genius Bar") == "hardware_repair_service"
    assert classify_by_rules("Why is the letter i changing to a strange symbol") == "display_touch_keyboard"
    assert classify_by_rules("Charged twice for my Apple Music subscription refund") == "app_store_billing"
    assert classify_by_rules("Wi-Fi keeps disconnecting and greyed out") == "network_connectivity"
    assert classify_by_rules("Microphone muffled no sound during phone calls") == "audio_sound"


def test_sample_dataset_file_integrity():
    sample_path = Path("data/sample/applesupport_sample_1000.jsonl")
    assert sample_path.exists(), "Sample dataset file must exist"

    lines = sample_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1000, f"Expected 1000 records, got {len(lines)}"

    seen_intents = set()
    for line in lines:
        record = json.loads(line)
        assert "conversation_id" in record
        assert "customer_initial_query" in record
        assert "agent_first_reply" in record
        assert "preliminary_intent" in record
        assert len(record["customer_initial_query"]) > 0
        assert len(record["agent_first_reply"]) > 0
        seen_intents.add(record["preliminary_intent"])

    # Ensure good diversity in sample
    assert len(seen_intents) >= 9, f"Expected broad intent coverage, found: {seen_intents}"
