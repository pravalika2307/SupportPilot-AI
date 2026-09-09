# SupportPilot AI - Phase 3 Retrieval Improvement & Escalation Quality Report

## Executive Summary
- **Dense Semantic Retriever (`all-MiniLM-L6-v2`)**: Outperformed sparse TF-IDF across all semantic metrics on the Golden 200 set.
  - **Mean Top-1 Cosine Similarity**: Increased from **`0.2364`** &rarr; **`0.6190`** (**+161.8%** relative gain).
  - **Top-1 Intent Concordance**: Increased from **`39.0%`** &rarr; **`55.0%`** (**+41.0%** relative gain).
  - **Dense Inference Latency**: **`10.99 ms/query`** on CPU.
- **Agent Escalation Policy Upgrade**: Data-driven intent-independent guardrails (catching non-Latin CJK scripts, account lockouts, and Apple Pay payment errors).
  - **Escalation Recall**: Increased from **`78.12%`** &rarr; **`82.81%`** (**+4.69%** absolute gain).
  - **Missed Escalations (False Negatives)**: Reduced from **`14`** down to **`11`**.
  - **Overall Escalation F1-Score**: Improved from **`68.03%`** &rarr; **`70.67%`**.

---

## 1. Retrieval Engine Comparison: Sparse TF-IDF vs Dense Semantic

Both retrievers were evaluated over the same 800 training conversations and queried with the 200 Golden set examples under strict zero-leakage isolation.

| Metric | TF-IDF Retriever (Phase 2) | Dense Semantic Retriever (Phase 3) | Absolute Improvement | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Top-1 Cosine Similarity** | `0.2364` | **`0.6190`** | **+0.3825** | **+161.8%** |
| **Mean Top-3 Avg Cosine Similarity** | `0.1964` | **`0.5834`** | **+0.3870** | **+197.1%** |
| **Intent Concordance Ratio (Top-1)** | `39.00%` | **`55.00%`** | **+16.00%** | **+41.0%** |
| **Inference Latency (CPU)** | **`6.51 ms/query`** | **`10.99 ms/query`** | +4.48 ms | Sub-30ms production ready |

### Key Semantic Strengths of Dense Retrieval
1. **Synonym and Paraphrase Handling**: Queries using phrasing like *"earphones cutting out"* successfully retrieve past cases with *"AirPods audio disconnecting"*, whereas TF-IDF failed due to zero n-gram overlap.
2. **Natural Query Variations**: Queries like *"can't get in my iPad"* match *"disabled device passcode reset"* with high semantic confidence.

---

## 2. Agent Escalation Quality Benchmark: Before vs After

| Metric | Phase 2 Baseline Policy | Phase 3 Upgraded Policy | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Decision Accuracy** | 76.50% | **78.00%** | **+1.50%** |
| **Escalation Precision** | 60.24% | **61.63%** | **+1.39%** |
| **Escalation Recall** | 78.12% | **82.81%** | **+4.69%** |
| **Escalation F1-Score** | 68.03% | **70.67%** | **+2.64%** |

### Confusion Matrix Comparison

| Outcome | Phase 2 Baseline | Phase 3 Upgraded | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **True Negatives (Safe Auto-Handle)** | 103 | **103** | Standard technical issues safely resolved without human intervention |
| **False Positives (Over-escalated)** | 33 | **33** | Safe queries escalated as precaution |
| **False Negatives (Missed Escalations)** | 14 | **11** | **Dangerous misses reduced by 3** |
| **True Positives (Proper Escalations)** | 50 | **53** | Critical security, billing, and repair cases properly handed off |

---

## 3. Qualitative Demonstration: Dense Grounding in Action

### Example: GOLDEN_001 (app_store_billing)
- **Customer Query**: *"why are my App Store categories in Spanish but everything else in English??"*
- **Decision**: `ESCALATE` (Reason: `financial_and_billing`)
- **Dense Top-1 Match (Cosine: 0.407)**: *"I️ can’t believe that @115858 & @AppleSupport has not fixed the vowel text issue. It’s very annoying and takes extra time to write something👎"*
- **Historical Verified Reply**: *"@117460 We'd like to look into the trouble you're having. Does this only occur in the Facebook app?..."*
- **Generated Draft Reply**: *"Based on verified resolution for similar cases: We'd like to look into the trouble you're having. Does this only occur in the Facebook app?..."*

### Example: GOLDEN_005 (app_store_billing)
- **Customer Query**: *"I can’t download apps from the app store."*
- **Decision**: `ESCALATE` (Reason: `financial_and_billing`)
- **Dense Top-1 Match (Cosine: 0.784)**: *"Since I updated my iPhone I can’t download apps now. Thanks @AppleSupport @115858 🙄🙄🙄"*
- **Historical Verified Reply**: *"@118243 Are you on the latest iOS 11.1 version?..."*
- **Generated Draft Reply**: *"To review charges, subscriptions, or request a refund securely, please visit reportaproblem.apple.com with your Apple ID. For account privacy and financial security, billing disput..."*
