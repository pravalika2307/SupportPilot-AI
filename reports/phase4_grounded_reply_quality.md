# SupportPilot AI — Phase 4 Grounded Reply Quality

## Actual evaluation
- Evaluated 200 real golden AppleSupport examples against an 800-thread zero-leakage retrieval pool.
- Top-1 retrieval relevance rate (dense score ≥ 0.45 and predicted/evidence intent agreement): **50.00%**.
- Grounded reply rate: **70.50%**.
- Appropriate abstention rate (no safe reply, escalated): **29.50%**.
- Unsafe/context-bound reply rate: **0.00%**.

## Design
Replies are sanitized copies of an intent-aligned, cited dense-retrieval match when its score is at least 0.45 and it contains no source-specific status or timing claim. Otherwise the system declines to create a policy-bearing reply and escalates for insufficient grounding. The agent output includes the cited conversation ID, matched query, original reply, rank, and similarity score.

## LLM and human judging
The repository includes a dependency-injected LLM-as-judge adapter with an explicit five-dimension 1–5 rubric: groundedness, relevance, helpfulness, tone, and safety. No LLM score is reported in this run because no real LLM endpoint or human annotations were supplied. `human_llm_agreement` computes per-dimension Cohen’s kappa when matching real labels are provided.