# SupportPilot AI (AppleSupport)

**SupportPilot AI** is a brand-specific AI customer-support system built for the Hiver SDE Intern assignment, centered around **`@AppleSupport`** from the *Customer Support on Twitter* (`twcs.csv`) dataset.

The system is designed to:
1. **Classify** inbound customer messages into a data-derived intent taxonomy.
2. **Retrieve** historically similar, resolved support conversations.
3. **Generate** evidence-bound support replies from cited historical AppleSupport examples.
4. **Decide** whether to **AUTO-HANDLE** or **ESCALATE** with explicit rationales (e.g. security risks, account lockouts, or hardware damage).
5. **Evaluate** rigor using reproducible baselines, a golden evaluation set (150–250 real examples), deterministic citation checks, and an LLM-as-a-judge harness.

---

## Phase 1 Deliverables & Discoveries

### 1. Dataset Scale & Analysis
- **Full Dataset**: 2,811,774 tweets across multiple global brands.
- **AppleSupport Volume**: **204,013 tweets** mention or are authored by `@AppleSupport` (~7.25% of the total dataset), making it the single largest support organization in the dataset.
- **Conversation Topology**:
  - Customer issues initiate as public root inbound tweets (`inbound=True`, `in_response_to_tweet_id=None`).
  - Average conversation length is **3.26 turns**.
  - **42.86%** of agent replies direct customers to a private Direct Message (DM) for account security or diagnostics.
  - **23.86%** of agent replies ask for the exact iOS version (`Settings > General > About`), while **14.22%** prompt for the hardware model.

### 2. Data-Derived Intent Taxonomy (10 Intents)

The taxonomy was derived directly from empirical clustering of customer problem statements:

| Intent Key | Canonical Name | Prevalence | Core Issues & Signatures |
|---|---|---|---|
| `battery_power` | Battery & Power | ~12.8% | Rapid battery drain post-update, device overheating, charging cable failure, shutdowns at 20-30%. |
| `software_update` | Software Update | ~26.7% | iOS installation errors ("Unable to Verify Update"), firmware restore, rollback requests. |
| `device_performance` | Device Performance | ~5.6% | Device freezing, app crashes, UI stutter/lag, spinning wheel, storage full warnings. |
| `apple_id_icloud` | Apple ID & iCloud | ~4.7% | Account disabled/locked, password reset, 2FA verification codes, iCloud sync failures. |
| `app_store_billing` | App Store & Billing | ~3.0% | Unrecognized charges, subscription cancellations, refund disputes, failed payment methods. |
| `display_touch_keyboard` | Display & Keyboard | ~5.2% | Unresponsive touchscreen, ghost touches, iOS 11 predictive text bug (letter "I" glitch), black screen. |
| `network_connectivity` | Connectivity & Network | ~2.2% | Wi-Fi toggle greyed out, Bluetooth drops with AirPods/car, cellular "No Service". |
| `audio_sound` | Audio & Sound | ~1.3% | Muffled earpiece, speaker crackling, microphone failure on calls, alarm volume issues. |
| `hardware_repair_service` | Hardware & Repair | ~1.8% | Cracked glass, water damage, repair pricing, AppleCare claims, Genius Bar appointments. |
| `order_shipping` | Orders & Shipping | ~0.6% | iPhone pre-order shipping dates, delivery tracking, store pickup reservations. |
| `general_inquiry_other` | General Inquiries | ~33.8% | Store hours, polite greetings, trade-in questions, non-technical queries. |

Detailed statistical distributions, token frequency tables, and problem signatures are documented in [reports/phase1_applesupport_eda.md](reports/phase1_applesupport_eda.md).

---

## Phase 2 Deliverables & Benchmark Results

### 1. Intent Taxonomy Validation & Refinement
In Phase 2, we empirically validated `general_inquiry_other` and found:
- Device freezing, crashing, and bootloops accounted for 17.2% of "other" queries &rarr; established `performance_freeze_crash` as a distinct intent.
- iOS 11 predictive text bug (letter 'i') accounted for 8.9% &rarr; re-routed into `display_touch_keyboard`.
- Taxonomy stabilized at **11 core technical intents + 1 calibrated fallback** (`general_inquiry_other`).

### 2. Golden Evaluation Set (200 Examples)
- Stratified 200 real examples across all 12 intents with difficulty ratings (Straightforward: 44%, Ambiguous: 44.5%, Edge Case: 11.5%).
- Ground-truth expected action: 136 `AUTO_HANDLE` (68.0%) vs 64 `ESCALATE` (32.0%).
- Isolated strictly into `data/golden_eval/golden_eval_200.jsonl` with 0 ID leakage into the 800-thread training corpus (`data/training/train_pool_800.jsonl`).

### 3. Real Benchmark Results (Golden 200)

| System Component | Metric | Baseline / Score |
| :--- | :--- | :--- |
| **Majority-Class Baseline** | Accuracy / Macro F1 | 8.50% / 1.31% |
| **TF-IDF + Logistic Regression** | Accuracy / Macro F1 | **64.00%** / **63.97%** |
| **TF-IDF + Logistic Regression** | Weighted F1 / Latency | **64.09%** / 0.06 ms/query |
| **Historical Support Retriever (TF-IDF)** | Mean Top-1 Cosine Sim | 0.236 (800 indexed cases) |
| **SupportPilot Agent Escalation (Phase 2)**| Accuracy / F1 | **76.50%** / **68.03%** |
| **SupportPilot Agent Escalation (Phase 2)**| Precision / Recall | **60.24%** / **78.12%** |

---

## Phase 3: Dense Semantic Retrieval & Escalation Quality Upgrades

### 1. Dense Semantic Retriever (`all-MiniLM-L6-v2`)
In Phase 3, we built a dense semantic retriever mapping customer queries into 384-dimensional embeddings and compared it side-by-side against the baseline TF-IDF retriever under identical zero-leakage conditions (800 training threads indexed, 200 golden evaluation queries):

| Retrieval Engine | Top-1 Cosine Sim | Top-3 Avg Cosine Sim | Intent Concordance | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF Sparse Retriever (Baseline)** | 0.2364 | 0.1964 | 39.00% | **6.51 ms/query** |
| **Dense Semantic Retriever (Phase 3)** | **0.6190** (+161.8%) | **0.5834** (+197.0%) | **55.00%** (+41.0%) | **10.99 ms/query** |

### 2. Escalation Quality Upgrades (Data-Driven FN Reduction)
Investigation of Phase 2 false negatives revealed missed escalations due to non-Latin unicode scripts (Japanese/CJK), account lockouts misclassified by bag-of-words, and Apple Pay errors. We implemented intent-independent safety triggers:
- Non-Latin script detection (`[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af...]`).
- Direct credential/lockout safety triggers (`locked out`, `passcode locked`, `account disabled`).
- Direct payment/Apple Pay triggers (`apple pay`, `payment not completed`, `refund`).

| Agent Metric | Phase 2 Baseline | Phase 3 Upgraded | Absolute Delta |
| :--- | :--- | :--- | :--- |
| **Overall Decision Accuracy** | 76.50% | **78.00%** | **+1.50%** |
| **Escalation Precision** | 60.24% | **61.63%** | **+1.39%** |
| **Escalation Recall** | 78.12% | **82.81%** | **+4.69%** |
| **Escalation F1-Score** | 68.03% | **70.67%** | **+2.64%** |
| **Missed Escalations (False Negatives)** | 14 | **11** | **-3 (-21.4%)** |

---

## Phase 4: Grounded Reply Generation & Quality Evaluation

Phase 4 makes a customer-facing reply evidence-bound instead of falling back to a hand-written intent guide:

- A reply is a sanitized copy of the top retrieved historical AppleSupport reply only when its similarity is at least `0.20`.
- `reply_evidence` exposes the source conversation ID, rank, similarity, matched query, and original historical reply.
- If there is insufficient evidence, the agent emits no policy-bearing troubleshooting and escalates with `insufficient_grounding`.
- The deterministic evaluation verifies citation integrity and unsupported-policy output. An injected `LLMReplyJudge` uses a five-dimension 1–5 rubric (groundedness, relevance, helpfulness, tone, safety); it intentionally records no score without a real judge call.
- `human_llm_agreement` computes per-dimension Cohen's kappa after matching real human and LLM annotations by `golden_idActual run on the real zero-leakage 200-example golden set: **70.50% provenance-backed / grounded response rate** (across top-3 candidates), **50.00% top-1 retrieval relevance rate**, **29.50% appropriate abstention rate** (safely escalated), and **0.00% detected unsafe or context-bound output**.

---

## Phase 5: Final Evaluation & Submission Layer

Phase 5 delivers the unified, reproducible evaluation harness and complete submission documentation for SupportPilot AI, running strictly against the 200 Golden Evaluation examples under zero leakage from the 800 training conversations.

### 1. Single Unified Reproduction Command

```bash
# Run the complete end-to-end benchmark suite across all 200 golden examples
python -m src.evaluation.run_final_eval
```

Outputs generated:
- Markdown Report: [reports/final_submission_eval_report.md](reports/final_submission_eval_report.md)
- Machine-Readable JSON: `reports/final_submission_metrics.json`

### 2. Final Tripartite Benchmark Results (REAL Data Only)

| Evaluation Dimension | Metric | Heuristic Baseline | Machine-Assisted Rec | Human-Verified Benchmark | Operational Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Classification** | Multi-Class Accuracy | 64.00% | 65.50% | **65.50%** (+1.50%) | Accuracy across 12 balanced intents |
| **Intent Classification** | Macro F1-Score | 63.97% | 65.47% | **65.47%** (+1.50%) | Unweighted average across 12 classes |
| **Intent Classification** | Weighted F1-Score | 64.09% | 65.49% | **65.49%** (+1.40%) | Class-weighted average |
| **Escalation Policy** | Decision Accuracy | 72.00% | 70.00% | **70.00%** (-2.00%) | Overall AUTO_HANDLE vs ESCALATE correctness |
| **Escalation Policy** | Escalation Recall (Safety) | 89.06% | 88.33% | **88.33%** | **Caught 53 of 60** safety-critical escalation cases |
| **Escalation Policy** | Escalation Precision | 53.77% | 50.00% | **50.00%** | Precision of human routing queue |
| **Escalation Policy** | Escalation F1-Score | 67.06% | 63.86% | **63.86%** | Safety vs automation containment balance |
| **Retrieval Relevance** | Top-1 Retrieval Relevance | 50.00% | 50.00% | **50.00%** | Candidate #1 similarity $\ge 0.45$ AND intent agreement |
| **Reply Grounding** | Grounded Response Rate | 70.50% | 70.50% | **70.50%** | Attribution coverage across Top-3 candidates |
| **Reply Grounding** | Appropriate Abstention Rate| 29.50% | 29.50% | **29.50%** | Safely escalated due to insufficient evidence |
| **Safety Compliance** | Unsafe / Context-Bound Rate| **0.00%** | **0.00%** | **0.00%** | Zero leaked private DMs, fake timelines, or ungrounded claims |
| **Dataset Isolation** | Leakage Check | **PASSED** | **PASSED** | **PASSED** | 0 shared IDs between train (800) and golden eval (200) |
| **LLM-as-a-Judge** | Judge Status | `not_run` | `not_run` | `not_run` | Zero synthetic scores reported; rubric & Cohen's $\kappa$ harness ready |

### 3. What is Misleading About My Headline Number?

> [!IMPORTANT]
> **Audited Headline Metric**: *Provenance-Backed / Grounded Response Rate = 70.50%*.
>
> It is tempting to present **70.50%** as the percentage of customer support inquiries successfully *resolved* by the automated system. **This interpretation is fundamentally misleading and must NOT be made.**
>
> 1. **Attribution Coverage vs. Ticket Resolution**: The **70.50%** figure measures **attribution coverage** &mdash; that is, the system found a verified, non-context-bound historical Apple Support interaction within the Top-3 dense semantic candidates that passed the $\ge 0.45$ similarity threshold and matched the customer's intent. It proves that the model's reply is 100% grounded in real historical Apple guidance without hallucinating unsupported policies.
> 2. **Lack of End-User Outcome Verification**: In customer support operations, true **First Contact Resolution (FCR)** requires observing that the customer's technical fault was permanently resolved (e.g. device restarted, update completed, battery drain ceased) and that no repeat contact occurred within 48–72 hours. A Twitter dataset of initial agent responses contains no telemetric confirmation that the suggested steps succeeded on the customer's specific physical hardware.
> 3. **Top-1 Precision vs. Top-3 Fallback Search**: While **70.50%** of queries found an attributable reply across the Top-3 candidates, the **Top-1 Retrieval Relevance Rate was 50.00%**. In 20.50% of cases, Candidate #1 had an intent mismatch or context-bound phrasing, requiring fallback to Candidate #2 or #3.

### 4. Golden Set Labeling Audit & Human Review Confirmation

- **Human Review Status**: **YES &mdash; Genuinely human reviewed and verified.**
- **Reviewer Identifier**: `Pravalika`
- **Total Records Verified**: **200 of 200** golden evaluation examples.
- **Records Modified**: **7 of 200** (3.5% correction rate correcting heuristic keyword noise).
- **Records Preserved Unchanged**: **193 of 200** (96.5%).
- **Audit Manifest**: [data/golden_eval/human_review_audit.json](data/golden_eval/human_review_audit.json) (`actually_human_reviewed = true`).
- **Separately Versioned Datasets**:
  - Original Heuristic Set: [data/golden_eval/golden_eval_200.jsonl](data/golden_eval/golden_eval_200.jsonl)
  - Machine-Assisted Recommendations: [data/golden_eval/machine_recommended_golden_200.jsonl](data/golden_eval/machine_recommended_golden_200.jsonl)
  - Final Human-Verified Benchmark: [data/golden_eval/human_verified_golden_200.jsonl](data/golden_eval/human_verified_golden_200.jsonl)
- **Review Infrastructure**:
  - Annotation Guidelines: [data/golden_eval/ANNOTATION_GUIDELINES.md](data/golden_eval/ANNOTATION_GUIDELINES.md)
  - Interactive Annotation Studio (Web UI): `src/data/annotation_studio.html` (served via `python -m src.data.human_review_tool --serve`)
  - CLI Review Tool: `src/data/human_review_tool.py`

### 5. Architectural Deliverables & Documentation

- **14 Non-Obvious Engineering Decisions**: [reports/decision_log.md](reports/decision_log.md)
- **Top 5 Real Failure Mode Root-Cause Analyses**: [reports/final_submission_eval_report.md](reports/final_submission_eval_report.md#4-top-5-meaningful-real-failure-modes)
- **Machine-Readable Metrics**: `reports/final_submission_metrics.json`


---

## Project Structure

```
SupportPilot AI/
├── .gitignore                      # Strictly excludes raw 500MB+ dataset & caches
├── requirements.txt                # Core dependencies: scikit-learn, sentence-transformers, torch, pandas
├── README.md                       # Comprehensive system documentation
├── data/
│   ├── sample/
│   │   ├── applesupport_sample_1000.jsonl  # 1,000 reproducible conversation threads
│   │   └── applesupport_sample_1000.csv    # Tabular sample export
│   ├── golden_eval/
│   │   ├── golden_eval_200.jsonl           # 200 stratified golden test examples
│   │   ├── golden_eval_200.csv             # Tabular golden evaluation set
│   │   ├── golden_eval_metadata.json       # Audit metadata & distribution stats
│   │   └── ANNOTATION_GUIDELINES.md        # Human annotation protocol & disclosure
│   └── training/
│       └── train_pool_800.jsonl            # 800 isolated training/retrieval threads
├── models/
│   ├── tfidf_lr_intent_model.joblib        # Trained intent classifier
│   ├── historical_retriever.joblib         # TF-IDF retrieval index (baseline)
│   └── dense_retriever.joblib              # Dense semantic retrieval index (Phase 3)
├── src/
│   ├── data/
│   │   ├── extract_applesupport.py         # Streaming raw ZIP extractor
│   │   ├── thread_reconstructor.py         # Multi-turn conversation reconstructor
│   │   ├── create_golden_eval.py           # Stratified evaluation set builder
│   │   └── human_review_tool.py            # Interactive CLI golden set review tool
│   ├── analysis/
│   │   └── eda_analysis.py                 # Intent taxonomy rules and statistical metrics
│   ├── models/
│   │   ├── baseline_majority.py            # Majority-class benchmark
│   │   ├── baseline_tfidf_lr.py            # TF-IDF + Logistic Regression baseline
│   │   ├── retriever.py                    # Historical support retrieval engine (TF-IDF)
│   │   ├── dense_retriever.py              # Dense semantic retrieval engine (SentenceTransformers)
│   │   ├── agent.py                        # SupportPilotAgent decision & drafting pipeline
│   │   └── reply_generation.py             # Evidence-bound historical reply selector
│   └── evaluation/
│       ├── run_final_eval.py               # Phase 5 unified evaluation CLI
│       ├── evaluate_pipeline.py            # Phase 2 evaluation harness
│       ├── evaluate_phase3_retrievers.py   # Phase 3 side-by-side evaluation harness
│       ├── evaluate_phase4_replies.py      # Phase 4 reply-quality evaluation
│       └── reply_quality.py                # Rubric, deterministic checks, agreement helper
├── reports/
│   ├── decision_log.md                     # 14 non-obvious engineering decisions & tradeoffs
│   ├── final_submission_eval_report.md     # Final Phase 5 benchmark & failure report
│   ├── final_submission_metrics.json       # Machine-readable final benchmark metrics
│   ├── phase1_applesupport_eda.md          # Comprehensive Phase 1 EDA report
│   ├── phase2_baseline_and_golden_eval.md  # Complete Phase 2 benchmark report
│   ├── phase2_metrics.json                 # Machine-readable Phase 2 metrics
│   ├── phase3_dense_retrieval_and_escalation.md # Phase 3 dense retrieval report
│   ├── phase3_metrics.json                 # Machine-readable Phase 3 metrics
│   ├── phase4_grounded_reply_quality.md    # Phase 4 grounded reply report
│   └── phase4_reply_quality_metrics.json   # Machine-readable Phase 4 metrics
└── tests/
    ├── test_data_pipeline.py               # Data pipeline & thread reconstruction tests
    ├── test_baselines_and_agent.py         # Baselines, TF-IDF retriever, and agent tests
    ├── test_dense_retriever_and_phase3.py  # Dense retrieval & upgraded guardrails tests
    └── test_phase4_grounded_replies.py     # Grounded reply & safety gate tests
```

---

## Running Tests

```bash
# Run all automated unit and integration tests (26 passing tests)
python -m pytest tests/ -v
```

---

## Quickstart & Environment Setup

### 1. Requirements
- Python 3.10+ (tested on Python 3.13)
- Dependencies in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Reproducible Development Sample
The repository includes a balanced, stratified 1,000-thread development sample (`data/sample/applesupport_sample_1000.jsonl` and `.csv`) with zero downloads required to run tests or evaluations.
