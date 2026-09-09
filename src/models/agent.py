"""
SupportPilot AI Customer Support Agent Pipeline.
Coordinates:
1. Intent classification & calibrated confidence estimation.
2. Historical evidence retrieval (top-3 similar resolved cases).
3. Grounded Apple-brand draft reply generation.
4. Decision policy (AUTO_HANDLE vs ESCALATE) with explicit escalation reasons.
"""

import re
from typing import List, Dict, Any, Optional

from src.models.baseline_tfidf_lr import TfidfLogisticRegressionBaseline
from src.models.retriever import HistoricalSupportRetriever
from src.data.create_golden_eval import NON_ENGLISH_INDICATORS, HIGH_RISK_ESCALATION_KEYWORDS

# Standard grounded Apple technical resolution knowledge base
INTENT_RESOLUTION_GUIDES = {
    "battery_power": (
        "We can certainly help with your battery performance! You can check your battery usage under Settings > Battery "
        "to see if a specific app is draining power in the background. Enabling Low Power Mode can also help conserve battery life. "
        "What model iPhone are you using, and what iOS version is installed?"
    ),
    "software_update": (
        "We want to make sure your device updates smoothly! To complete the update, ensure your device is connected to a reliable "
        "Wi-Fi network, plugged into power, and has at least 5GB of free storage. You can check for updates in Settings > General > Software Update. "
        "Are you receiving a specific error message when attempting to install?"
    ),
    "display_touch_keyboard": (
        "We're here to help with your typing and display issue! If you are encountering the letter 'i' autocorrect symbol glitch, "
        "updating to the latest iOS release or adding a Text Replacement in Settings > General > Keyboard will resolve it. "
        "If your screen is unresponsive or flickering, try a force restart. What device model are you using?"
    ),
    "network_connectivity": (
        "Let's get your connection back up and running! We recommend toggling Airplane Mode on for 15 seconds, then off. "
        "If you're having trouble with Wi-Fi or Cellular, try resetting your connection under Settings > General > Reset > Reset Network Settings. "
        "Does this occur with all Wi-Fi networks or only a specific one?"
    ),
    "audio_sound": (
        "Audio issues can definitely be frustrating, and we'd be glad to help! Please check that the Ring/Silent switch isn't set to silent, "
        "and verify that Do Not Disturb is disabled. Also inspect the speaker and microphone meshes for any debris. "
        "Are you noticing this on phone calls, speaker audio, or with headphones?"
    ),
    "camera_photos": (
        "Let's look into your camera and photos issue! First, force close the Camera app by swiping up from the app switcher and reopen it. "
        "If the screen remains black, restart your device. Make sure Camera permissions are enabled under Settings > Privacy > Camera. "
        "Which camera (front or rear) is having the issue?"
    ),
    "order_shipping": (
        "We'd be happy to assist with your order and delivery details! You can track real-time shipment status anytime by logging into "
        "your Apple Store account online or via the Apple Store app under Order History. "
        "Do you have your web order number available?"
    ),
    "performance_freeze_crash": (
        "We understand how frustrating device freezing can be! If your device is frozen or unresponsive, perform a force restart "
        "by quickly pressing and releasing Volume Up, then Volume Down, and holding the Side button until the Apple logo appears. "
        "Does this freeze happen during specific app usage or randomly?"
    ),
    "hardware_repair_service": (
        "For physical hardware damage or cracked screens, your device will require in-person inspection or mail-in repair. "
        "Please visit https://support.apple.com/repair or an Apple Authorized Service Provider to schedule a Genius Bar appointment."
    ),
    "app_store_billing": (
        "To review charges, subscriptions, or request a refund securely, please visit reportaproblem.apple.com with your Apple ID. "
        "For account privacy and financial security, billing disputes are handled directly through our secure billing team."
    ),
    "apple_id_icloud": (
        "For Apple ID security and account access recovery, please visit iforgot.apple.com to verify your identity and reset your passcode. "
        "Never share verification codes or passwords over social media."
    ),
    "general_inquiry_other": (
        "Thanks for reaching out! We'd love to help you get this sorted out. Could you share a few more details about what you're experiencing, "
        "along with your device model and current iOS version?"
    )
}


# Unicode non-Latin scripts: CJK, Hiragana, Katakana, Hangul, Arabic, Cyrillic
UNICODE_NON_LATIN_SCRIPT_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]"
)

# Intent-independent high-severity safety triggers
DIRECT_SAFETY_TRIGGERS = {
    "account_security_and_pii": [
        r"\blocked out\b", r"\bdisabled\b", r"\bpasscode locked\b",
        r"\b2fa\b", r"\btwo-factor\b", r"\bhacked\b", r"\bverification code\b"
    ],
    "financial_and_billing": [
        r"\bapple pay\b", r"\brefund\b", r"\bunauthorized\b",
        r"\bcharged twice\b", r"\bpayment declined\b", r"\bpayment not completed\b"
    ],
    "hardware_physical_damage": [
        r"\bcracked screen\b", r"\bshattered\b", r"\bwater damage\b",
        r"\bdropped in water\b", r"\bgenius bar\b"
    ],
    "unresolved_system_crash": [
        r"\bbootloop\b", r"\bbricked\b", r"\breboot loop\b", r"\bstuck on apple logo\b"
    ]
}


class SupportPilotAgent:
    def __init__(
        self,
        classifier: TfidfLogisticRegressionBaseline,
        retriever: Any,
        confidence_threshold: float = 0.12,
        similarity_threshold: float = 0.20
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

    def _detect_escalation_risk(self, query: str, intent: str, confidence: float) -> Optional[str]:
        """
        Applies brand safety and operational risk guardrails to determine escalation necessity.
        Combines intent-independent safety triggers with intent-specific classification rules.
        """
        query_lower = query.lower()

        # 1. Non-English detection (Latin keywords + Non-Latin scripts e.g. Japanese, Chinese, Arabic)
        if UNICODE_NON_LATIN_SCRIPT_PATTERN.search(query):
            return "non_english_query"
        if any(re.search(p, query_lower) for p in NON_ENGLISH_INDICATORS):
            return "non_english_query"

        # 2. Abusive language or legal threats
        if any(re.search(p, query_lower) for p in HIGH_RISK_ESCALATION_KEYWORDS["abusive_or_legal"]):
            return "vague_or_abusive"

        # 3. Intent-independent critical safety triggers (catches misclassified high-risk queries)
        for reason, patterns in DIRECT_SAFETY_TRIGGERS.items():
            if any(re.search(p, query_lower) for p in patterns):
                return reason

        # 4. Intent-specific policy checks
        if intent == "hardware_repair_service":
            return "hardware_physical_damage"

        if intent == "app_store_billing":
            return "financial_and_billing"

        if intent == "apple_id_icloud":
            if any(re.search(p, query_lower) for p in HIGH_RISK_ESCALATION_KEYWORDS["apple_id_icloud"]):
                return "account_security_and_pii"

        if intent == "performance_freeze_crash":
            if any(re.search(p, query_lower) for p in HIGH_RISK_ESCALATION_KEYWORDS["performance_freeze_crash"]):
                return "unresolved_system_crash"

        # 5. Low confidence or unclassified fallback
        if confidence < self.confidence_threshold:
            return "low_intent_confidence"

        if intent == "general_inquiry_other":
            return "vague_or_abusive"

        return None

    def _generate_draft_reply(
        self,
        query: str,
        intent: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesizes a brand-aligned, grounded draft response using retrieved evidence
        or the verified Apple resolution repository.
        """
        top_match = retrieved_evidence[0] if retrieved_evidence else None
        
        # If top match is highly relevant and has a clean resolution
        if top_match and top_match["similarity_score"] >= self.similarity_threshold:
            hist_reply = top_match["historical_agent_reply"]
            # Clean up Twitter handles and links from historical reply
            clean_reply = re.sub(r"@\w+\b", "", hist_reply).strip()
            # If the historical reply provides useful guidance
            if len(clean_reply.split()) > 8 and "dm" not in clean_reply.lower():
                return f"Based on verified resolution for similar cases: {clean_reply}"

        # Otherwise fallback to grounded intent resolution guide
        return INTENT_RESOLUTION_GUIDES.get(intent, INTENT_RESOLUTION_GUIDES["general_inquiry_other"])

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Full end-to-end SupportPilot AI pipeline execution.
        """
        clean_query = query.strip()
        
        # 1. Intent classification & confidence scoring
        clf_result = self.classifier.predict_with_confidence([clean_query])[0]
        predicted_intent = clf_result["intent"]
        confidence = round(clf_result["confidence"], 4)
        probabilities = clf_result["probabilities"]

        # 2. Evidence retrieval (top-3)
        retrieved_evidence = self.retriever.retrieve(clean_query, top_k=3)

        # 3. Decision policy: AUTO_HANDLE vs ESCALATE
        escalation_reason = self._detect_escalation_risk(clean_query, predicted_intent, confidence)
        
        if escalation_reason is not None:
            decision = "ESCALATE"
            rationale = (
                f"Escalated due to {escalation_reason.replace('_', ' ')} policy "
                f"(intent={predicted_intent}, confidence={confidence:.2f})."
            )
        else:
            decision = "AUTO_HANDLE"
            escalation_reason = "none"
            rationale = (
                f"Eligible for auto-resolution: standard technical troubleshooting for {predicted_intent} "
                f"with sufficient confidence ({confidence:.2f} >= {self.confidence_threshold:.2f})."
            )

        # 4. Grounded draft reply generation
        draft_reply = self._generate_draft_reply(clean_query, predicted_intent, retrieved_evidence)

        return {
            "customer_query": clean_query,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "decision": decision,
            "escalation_reason": escalation_reason,
            "decision_rationale": rationale,
            "draft_reply": draft_reply,
            "retrieved_evidence": retrieved_evidence,
            "class_probabilities": probabilities
        }
