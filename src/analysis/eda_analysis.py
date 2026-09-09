"""
Exploratory Data Analysis (EDA) module for AppleSupport customer support interactions.
Analyzes recurring issues, conversation topology, and intent distribution.
"""

import re
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np


# Validated 11 Data-Derived Technical Intents + 1 Fallback (general_inquiry_other)
INTENT_PATTERNS: Dict[str, List[str]] = {
    "hardware_repair_service": [
        r"\bcrack\b", r"\bcracked\b", r"\bbroken\b", r"\brepair\b", r"\bwarranty\b",
        r"\bapple care\b", r"\bapplecare\b", r"\bgenius bar\b", r"\bappointment\b",
        r"\bwater damage\b", r"\breplace\b", r"\breplacement\b",
        r"\bcost\b", r"\bphysical damage\b", r"\bfix my screen\b"
    ],
    "battery_power": [
        r"\bbattery\b", r"\bdrain\b", r"\bdraining\b", r"\bcharge\b", r"\bcharging\b",
        r"\bcharger\b", r"\boverheat\b", r"\bpercentage\b", r"\bpower\b", r"\bshut off\b",
        r"\bshutting down\b", r"\bdies\b", r"\bdying\b", r"\bbattery life\b"
    ],
    "software_update": [
        r"\bupdate\b", r"\bupdating\b", r"\bios\b", r"\bios 11\b", r"\bios 10\b",
        r"\binstall\b", r"\binstalled\b", r"\bfirmware\b", r"\bupgrade\b", r"\bdowngrade\b",
        r"\bverifying update\b", r"\bsoftware update\b"
    ],
    "apple_id_icloud": [
        r"\bapple id\b", r"\bicloud\b", r"\bpassword\b", r"\bpasscode\b", r"\baccount\b",
        r"\btwo-factor\b", r"\b2fa\b", r"\bverification code\b", r"\blocked out\b",
        r"\bsign in\b", r"\blogin\b", r"\bdisabled\b", r"\bsecurity question\b"
    ],
    "app_store_billing": [
        r"\bapp store\b", r"\bitunes\b", r"\bsubscription\b", r"\brefund\b",
        r"\bpurchase\b", r"\bcharged\b", r"\bbill\b", r"\bbilling\b", r"\breceipt\b",
        r"\bcredit card\b", r"\bpayment\b", r"\bin-app\b", r"\bcancel subscription\b"
    ],
    "performance_freeze_crash": [
        r"\bfreeze\b", r"\bfreezing\b", r"\bfrozen\b", r"\bcrash\b", r"\bcrashing\b",
        r"\blag\b", r"\blagging\b", r"\bslow\b", r"\brestart\b", r"\brestarting\b",
        r"\breboot\b", r"\bbootloop\b", r"\bapple logo\b", r"\bstuck on apple\b",
        r"\bforce restart\b", r"\bspinning wheel\b"
    ],
    "display_touch_keyboard": [
        r"\bkeyboard\b", r"\btyping\b", r"\bautocorrect\b", r"\bauto-correct\b",
        r"\bthe letter i\b", r"\bletter i\b", r"\bi️\b", r"\btype an i\b",
        r"\bcapital i\b", r"\ba and a question mark\b", r"\btouch\b", r"\btouchscreen\b",
        r"\bunresponsive\b", r"\bblack screen\b", r"\bflicker\b", r"\bglitch\b",
        r"\bhome button\b", r"\bdisplay\b", r"\bscreen\b"
    ],
    "network_connectivity": [
        r"\bwifi\b", r"\bwi-fi\b", r"\bbluetooth\b", r"\bcellular\b", r"\bno service\b",
        r"\bsearching\b", r"\blte\b", r"\b4g\b", r"\bsignal\b", r"\bhotspot\b",
        r"\bairdrop\b", r"\bconnect\b", r"\bdisconnecting\b"
    ],
    "audio_sound": [
        r"\bsound\b", r"\bspeaker\b", r"\bmic\b", r"\bmicrophone\b", r"\bvolume\b",
        r"\bearpiece\b", r"\bheadphone\b", r"\bairpod\b", r"\baudio\b", r"\bringtone\b",
        r"\balarm\b", r"\bmute\b", r"\bdistorted\b"
    ],
    "camera_photos": [
        r"\bcamera\b", r"\bphoto\b", r"\bphotos\b", r"\bpicture\b", r"\bpictures\b",
        r"\bscreenshot\b", r"\blive photo\b", r"\blens\b", r"\bportrait\b",
        r"\bvideo recording\b", r"\bcamera roll\b"
    ],
    "order_shipping": [
        r"\border\b", r"\bshipping\b", r"\bdelivery\b", r"\bdelivered\b",
        r"\bpre-order\b", r"\bpreorder\b", r"\bdispatch\b", r"\btrack\b",
        r"\breservation\b", r"\bshipment\b", r"\bstatus\b", r"\bstore pickup\b"
    ],
}



def classify_by_rules(text: str) -> str:
    """Deterministic rule-based intent mapper for initial clustering and analysis."""
    text_lower = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            return intent
    return "general_inquiry_other"


def compute_conversation_statistics(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute structural and textual metrics over reconstructed conversations."""
    if not conversations:
        return {}

    num_convs = len(conversations)
    turn_counts = [c["num_turns"] for c in conversations]
    query_lengths_char = [len(c["customer_initial_query"]) for c in conversations]
    query_lengths_word = [len(c["customer_initial_query"].split()) for c in conversations]
    reply_lengths_char = [len(c["agent_first_reply"]) for c in conversations]
    reply_lengths_word = [len(c["agent_first_reply"].split()) for c in conversations]

    # Intent breakdown
    intent_counts = Counter()
    for c in conversations:
        intent = classify_by_rules(c["customer_initial_query"])
        intent_counts[intent] += 1

    # Apple agent response patterns
    dm_direct_count = sum(1 for c in conversations if "dm" in c["agent_first_reply"].lower())
    ask_ios_version_count = sum(1 for c in conversations if "version" in c["agent_first_reply"].lower() or "ios" in c["agent_first_reply"].lower())
    ask_device_model_count = sum(1 for c in conversations if "model" in c["agent_first_reply"].lower() or "device" in c["agent_first_reply"].lower())

    # Top customer issue words (excluding stopwords)
    stopwords = {"the", "i", "a", "to", "and", "is", "my", "in", "it", "of", "for", "on", "you", "that", "this", "with", "me", "have", "so", "be", "not", "are", "applesupport", "@applesupport", "at", "but", "can", "why", "how", "what", "do", "get", "when", "just"}
    word_counts = Counter()
    for c in conversations:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", c["clean_customer_query"].lower())
        for w in words:
            if w not in stopwords:
                word_counts[w] += 1

    return {
        "total_conversations": num_convs,
        "turn_distribution": {
            "mean_turns": float(np.mean(turn_counts)),
            "median_turns": float(np.median(turn_counts)),
            "max_turns": int(np.max(turn_counts)),
            "min_turns": int(np.min(turn_counts)),
            "turns_histogram": dict(Counter(turn_counts).most_common())
        },
        "query_length_chars": {
            "mean": float(np.mean(query_lengths_char)),
            "median": float(np.median(query_lengths_char)),
            "p95": float(np.percentile(query_lengths_char, 95))
        },
        "query_length_words": {
            "mean": float(np.mean(query_lengths_word)),
            "median": float(np.median(query_lengths_word)),
        },
        "reply_length_words": {
            "mean": float(np.mean(reply_lengths_word)),
            "median": float(np.median(reply_lengths_word)),
        },
        "intent_distribution": dict(intent_counts.most_common()),
        "agent_response_signals": {
            "dm_redirection_pct": round(dm_direct_count / num_convs * 100, 2),
            "ask_ios_version_pct": round(ask_ios_version_count / num_convs * 100, 2),
            "ask_device_model_pct": round(ask_device_model_count / num_convs * 100, 2),
        },
        "top_keywords": word_counts.most_common(25)
    }
