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

## Project Structure

```
SupportPilot AI/
├── .gitignore                      # Strictly excludes raw 500MB+ dataset & caches
├── requirements.txt                # Lightweight, well-known dependencies
├── README.md                       # Project documentation & architecture overview
├── data/
│   ├── sample/
│   │   ├── applesupport_sample_1000.jsonl  # 1,000 reproducible conversation threads
│   │   └── applesupport_sample_1000.csv    # Tabular export for quick inspection
│   └── golden_eval/
│       └── golden_eval_200.jsonl           # 150-250 verified examples (Phase 2)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extract_applesupport.py         # Memory-efficient streaming from ZIP/CSV
│   │   ├── thread_reconstructor.py         # Rebuilds parent-child dialogue trees
│   │   └── build_sample_and_report.py      # Generates sample & statistical report
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── eda_analysis.py                 # Token frequencies, distributions, taxonomy
│   ├── models/                             # (Phase 2: Baselines, RAG, & Agent)
│   │   ├── baseline_majority.py
│   │   ├── baseline_tfidf_lr.py
│   │   ├── intent_classifier.py
│   │   ├── retriever.py
│   │   └── agent.py
│   └── evaluation/                         # (Phase 2: Evaluation Harness)
│       ├── metrics.py
│       ├── llm_judge.py
│       └── run_eval.py
├── reports/
│   └── phase1_applesupport_eda.md          # Full EDA & taxonomy analysis report
└── tests/
    └── test_data_pipeline.py               # Automated unit tests for data extraction & parsing
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
