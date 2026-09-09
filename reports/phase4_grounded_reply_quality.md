# SupportPilot AI — Phase 4 Grounded Reply Quality Benchmark

## 1. Actual Measured Metrics (Real 200 Golden Examples)

All metrics below were computed across the exact same 200 zero-leakage golden AppleSupport examples against the isolated 800-thread historical corpus:

- **Top-1 Retrieval Relevance Rate**: **50.00%** (100 / 200)
  - *Definition*: Percentage of queries where the single highest-ranked candidate (`rank == 1`) met both the dense semantic similarity threshold ($\ge 0.45$) AND had concordant intent with the customer's query (`historical_intent == predicted_intent`).
- **Provenance-Backed / Grounded Response Rate**: **70.50%** (141 / 200)
  - *Definition*: Percentage of queries where the agent successfully found and cited an intent-aligned, non-context-bound verified historical reply from among the **Top-3 retrieved candidates**.
  - *Why this differs from Top-1 Relevance (50.00%)*: In 41 cases (20.50%), the Rank-1 candidate was rejected (due to intent divergence or context-specific DM language), but Candidate #2 or Candidate #3 provided a valid, intent-aligned historical resolution. This metric measures attribution coverage, not verified customer satisfaction or ticket resolution: the 141 responses are provenance-backed responses directly attributable to isolated Apple historical records, subject to the retrieval and safety gates.
- **Appropriate Abstention Rate (Safe Escalation)**: **29.50%** (59 / 200)
  - *Definition*: Percentage of queries where all retrieved candidates failed the similarity threshold, had intent mismatches, or were context-bound. Rather than guessing or inventing policy, the agent safely withheld a reply and escalated with `insufficient_grounding`.
- **Unsafe / Context-Bound Reply Rate**: **0.00%** (0 / 200)
  - *Definition*: Percentage of emitted replies containing interaction-specific private claims (e.g. *"received your DM"*, *"responding momentarily"*), unauthorized refund promises, fake turnaround times, or unverified links.
- **Context-Bound Candidates Rejected**: **26 candidates**
  - Explicitly filtered out to prevent private DM / interaction-specific claims from leaking into responses.

---

## 2. Design & Anti-Hallucination Guardrails

Replies are sanitized, verbatim segments of an intent-aligned, cited dense-retrieval match when its score is at least 0.45 and it contains no source-specific status or timing claim. When no candidate satisfies these constraints, the system declines to generate a policy-bearing reply and safely escalates for insufficient grounding. Every emitted reply includes complete citation provenance:
- Cited conversation ID
- Retrieval rank (1, 2, or 3)
- Cosine similarity score
- Matched historical query
- Verbatim historical agent reply

---

## 3. LLM-as-a-Judge & Human Agreement Plumbing

- **Rubric**: Explicit five-dimension 1–5 rubric covering **Groundedness**, **Relevance / Correctness**, **Helpfulness**, **Tone**, and **Safety / Hallucination**.
- **Zero Fabrication**: No synthetic judge ratings are reported (`llm_judge.status = "not_run"`). All evaluation scores remain empty until real LLM API calls or human annotations are provided.
- **Agreement Metric**: The evaluation harness includes `human_llm_agreement()`, which calculates quadratic-weighted Cohen's kappa ($\kappa$) across all five dimensions once matching human and LLM labels are indexed by `golden_id`.