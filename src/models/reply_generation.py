"""Evidence-bound reply generation for SupportPilot AI.

The module deliberately does not use a free-form language model.  A customer-facing
reply is a sanitized historical AppleSupport reply, with its source retained in the
output.  This makes every operational statement attributable to a retrieved example
and prevents the agent from inventing support policy.
"""

import re
from typing import Any, Dict, List


class GroundedReplyGenerator:
    """Selects one cited historical reply or returns an insufficient-evidence result."""

    def __init__(self, min_similarity_threshold: float = 0.20) -> None:
        self.min_similarity_threshold = min_similarity_threshold

    CONTEXT_BOUND_PATTERNS = (
        r"\b(received|got) your dm\b", r"\brespond(?:ing|ed)? (?:there |to you )?(?:momentarily|shortly|soon)\b",
        r"\bwe(?:'| a)?ll (?:be )?respond(?:ing)? (?:there |to you )?(?:momentarily|shortly|soon)\b",
        r"\byour (?:order|case|repair|request) (?:is|has been|was)\b", r"\bwe have (?:already )?(?:processed|approved|completed)\b",
    )

    @staticmethod
    def _sanitize_historical_reply(reply: str) -> str:
        """Remove a public Twitter handle without adding any new support guidance."""
        return re.sub(r"@\w+\b", "", reply).strip()

    @classmethod
    def is_context_bound(cls, reply: str) -> bool:
        return any(re.search(pattern, reply, flags=re.IGNORECASE) for pattern in cls.CONTEXT_BOUND_PATTERNS)

    def generate(self, retrieved_evidence: List[Dict[str, Any]], predicted_intent: str) -> Dict[str, Any]:
        """Return a reply that is fully attributable to one retrieved record.

        The copied reply is intentionally not paraphrased.  Paraphrasing would make
        a lexical policy check weaker and can silently introduce unsupported steps.
        """
        rejected = []
        for candidate in retrieved_evidence:
            if candidate["similarity_score"] < self.min_similarity_threshold:
                rejected.append("low_similarity")
                continue
            if candidate["historical_intent"] != predicted_intent:
                rejected.append("intent_mismatch")
                continue
            source_reply = self._sanitize_historical_reply(candidate["historical_agent_reply"])
            if not source_reply or self.is_context_bound(source_reply):
                rejected.append("context_bound")
                continue
            citation = {key: candidate[key] for key in ("conversation_id", "rank", "similarity_score", "matched_customer_query", "historical_agent_reply", "historical_intent")}
            return {"text": source_reply, "grounding_status": "grounded", "citation": citation, "unsupported_policy_detected": False, "rejection_reasons": []}
        return {
            "text": "I don’t have a sufficiently relevant verified support example to provide a safe reply.",
            "grounding_status": "insufficient_evidence", "citation": None, "unsupported_policy_detected": False,
            "rejection_reasons": sorted(set(rejected or ["no_retrieved_evidence"])),
        }
