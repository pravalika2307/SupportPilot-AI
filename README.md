# SupportPilot AI (AppleSupport)

**SupportPilot AI** is a brand-specific AI customer-support system built for the Hiver SDE Intern assignment, centered around **`@AppleSupport`** from the *Customer Support on Twitter* (`twcs.csv`) dataset.

The system is designed to:
1. **Classify** inbound customer messages into a data-derived intent taxonomy.
2. **Retrieve** historically similar, resolved support conversations.
3. **Generate** grounded, brand-aligned support replies with step-by-step guidance.
4. **Decide** whether to **AUTO-HANDLE** or **ESCALATE** with explicit rationales (e.g. security risks, account lockouts, or hardware damage).
5. **Evaluate** rigor using reproducible baselines, a golden evaluation set (150–250 real examples), and an LLM-as-a-judge harness.

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
| **Historical Support Retriever** | Mean Top-1 Cosine Sim | 0.236 (800 indexed cases) |
| **SupportPilot Agent Escalation** | Accuracy / F1 | **76.50%** / **68.03%** |
| **SupportPilot Agent Escalation** | Precision / Recall | **60.24%** / **78.12%** |

---

## Project Structure

```
SupportPilot AI/
├── .gitignore                      # Strictly excludes raw 500MB+ dataset & caches
├── requirements.txt                # Core dependencies: scikit-learn, pandas, torch, etc.
├── README.md                       # Comprehensive system documentation
├── data/
│   ├── sample/
│   │   ├── applesupport_sample_1000.jsonl  # 1,000 reproducible conversation threads
│   │   └── applesupport_sample_1000.csv    # Tabular sample export
│   ├── golden_eval/
│   │   ├── golden_eval_200.jsonl           # 200 stratified golden test examples
│   │   └── golden_eval_200.csv             # Tabular golden evaluation set
│   └── training/
│       └── train_pool_800.jsonl            # 800 isolated training/retrieval threads
├── models/
│   ├── tfidf_lr_intent_model.joblib        # Trained intent classifier
│   └── historical_retriever.joblib         # Historical support retrieval index
├── src/
│   ├── data/
│   │   ├── extract_applesupport.py         # Streaming raw ZIP extractor
│   │   ├── thread_reconstructor.py         # Multi-turn conversation reconstructor
│   │   └── create_golden_eval.py           # Stratified evaluation set builder
│   ├── analysis/
│   │   └── eda_analysis.py                 # Intent taxonomy rules and statistical metrics
│   ├── models/
│   │   ├── baseline_majority.py            # Majority-class benchmark
│   │   ├── baseline_tfidf_lr.py            # TF-IDF + Logistic Regression baseline
│   │   ├── retriever.py                    # Historical support retrieval engine
│   │   └── agent.py                        # SupportPilotAgent decision & drafting pipeline
│   └── evaluation/
│       └── evaluate_pipeline.py            # Automated evaluation harness
├── reports/
│   ├── phase1_applesupport_eda.md          # Comprehensive Phase 1 EDA report
│   ├── phase2_baseline_and_golden_eval.md  # Complete Phase 2 benchmark report
│   └── phase2_metrics.json                 # Machine-readable evaluation metrics
└── tests/
    ├── test_data_pipeline.py               # Data pipeline & thread reconstruction tests
    └── test_baselines_and_agent.py         # Baselines, retriever, and agent tests
```

---

## Running Evaluation & Tests

```bash
# Run all automated unit and integration tests
python -m pytest tests/ -v

# Run the full Phase 2 evaluation harness
python -m src.evaluation.evaluate_pipeline
```

---

## Quickstart & Environment Setup

### 1. Requirements
- Python 3.10+ (tested on Python 3.13)
- Lightweight packages: `pandas`, `numpy`, `scikit-learn`, `pydantic`, `rich`, `pytest`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m pytest tests/test_data_pipeline.py -v
```

### 3. Reproducible Development Sample
No 500+ MB download is needed to develop or test! The repository includes a balanced, stratified 1,000-thread development sample:
- `data/sample/applesupport_sample_1000.jsonl`
- `data/sample/applesupport_sample_1000.csv`

Each record contains the full conversation metadata:
```json
{
  "conversation_id": "conv_116345",
  "root_tweet_id": "116345",
  "customer_id": "105838",
  "customer_initial_query": "@AppleSupport newest version of iOS and iPhone 7",
  "clean_customer_query": "newest version of iOS and iPhone 7",
  "agent_first_reply": "@105838 Since we just released iOS 11.1 today, can you clarify which exact version you have installed?",
  "num_turns": 4,
  "preliminary_intent": "software_update",
  "has_link_or_dm": true
}
```

### 4. Regenerating Data & Analysis
If the raw `archive.zip` or `twcs.csv` is available on the machine:
```bash
python -m src.data.build_sample_and_report
```

---

## Planned Evaluation Harness (Phase 2)

The evaluation suite will benchmark the following components without simulated or fake results:
1. **Baselines**:
   - **Majority-Class Baseline**: Predicts the most frequent class.
   - **TF-IDF + Logistic Regression Baseline**: Classical n-gram feature baseline.
   - **Main AI Agent**: Context-aware classifier + grounded RAG agent.
2. **Evaluation Metrics**:
   - Multi-class Accuracy & Macro F1.
   - Per-intent Precision, Recall, and Confusion Matrix.
   - Groundedness, Helpfulness, Tone, and Hallucination rate via LLM-as-a-judge.
   - Cohen's Kappa for Human-vs-LLM judge agreement.
   - Escalation Precision / Recall (`AUTO-HANDLE` vs `ESCALATE`).
   - Top 5 failure mode taxonomy.
