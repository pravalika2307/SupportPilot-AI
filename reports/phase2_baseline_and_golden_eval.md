# SupportPilot AI - Phase 2 Baseline & Golden Evaluation Benchmark Report

## Executive Summary
- **Golden Evaluation Set**: 200 real AppleSupport conversations, perfectly stratified across 12 intents.
- **Train / Retrieval Pool**: 800 conversations strictly isolated with zero ID overlap.
- **Intent Classification Accuracy**: **64.00%** (TF-IDF + LR) vs **8.50%** (Majority Baseline) &mdash; an absolute improvement of **+55.50%**.
- **Intent Classification Macro F1**: **63.97%** vs **1.31%**.
- **Escalation Policy F1**: **68.03%** (Recall: **78.12%**, Precision: **60.24%**).

---

## 1. Intent Classification Baseline Comparison

| Metric | Majority-Class Baseline | TF-IDF + Logistic Regression | Absolute Gain |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 8.50% | **64.00%** | **+55.50%** |
| **Macro Precision** | 8.33% | **66.34%** | **+58.01%** |
| **Macro Recall** | 8.33% | **63.76%** | **+55.42%** |
| **Macro F1-Score** | 1.31% | **63.97%** | **+62.67%** |
| **Weighted F1-Score** | 1.33% | **64.09%** | **+62.76%** |
| **Inference Latency** | <0.01 ms | **0.03 ms/query** | &mdash; |

---

## 2. Per-Intent Performance (TF-IDF + Logistic Regression)

| Intent Class | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| `app_store_billing` | 17 | 52.6% | 58.8% | **55.6%** |
| `apple_id_icloud` | 17 | 76.5% | 76.5% | **76.5%** |
| `audio_sound` | 17 | 72.2% | 76.5% | **74.3%** |
| `battery_power` | 17 | 93.8% | 88.2% | **90.9%** |
| `camera_photos` | 17 | 71.4% | 88.2% | **79.0%** |
| `display_touch_keyboard` | 17 | 52.4% | 64.7% | **57.9%** |
| `general_inquiry_other` | 17 | 30.4% | 41.2% | **35.0%** |
| `hardware_repair_service` | 17 | 68.8% | 64.7% | **66.7%** |
| `network_connectivity` | 16 | 64.7% | 68.8% | **66.7%** |
| `order_shipping` | 16 | 100.0% | 50.0% | **66.7%** |
| `performance_freeze_crash` | 16 | 36.4% | 25.0% | **29.6%** |
| `software_update` | 16 | 76.9% | 62.5% | **69.0%** |

### Top Predictive Features per Intent Class

- **`app_store_billing`**: `itunes` (3.36), `app store` (1.54), `purchase` (1.46), `store` (1.36), `refund` (1.24), `app` (0.97)
- **`apple_id_icloud`**: `icloud` (3.11), `password` (1.60), `account` (1.57), `apple id` (1.45), `id` (1.30), `my apple` (1.20)
- **`audio_sound`**: `sound` (2.86), `volume` (2.05), `alarm` (1.74), `audio` (1.30), `speaker` (1.26), `calls` (1.10)
- **`battery_power`**: `battery` (4.71), `ios` (1.65), `battery life` (1.41), `my battery` (1.39), `life` (1.35), `11` (1.09)
- **`camera_photos`**: `photos` (2.75), `pictures` (2.22), `camera` (1.79), `my pictures` (1.42), `screenshot` (1.35), `photo` (1.25)
- **`display_touch_keyboard`**: `screen` (2.13), `keyboard` (2.11), `letter` (1.77), `the letter` (1.51), `touch` (1.33), `type` (1.19)
- **`general_inquiry_other`**: `store` (0.75), `why` (0.65), `apple store` (0.59), `what` (0.53), `doing` (0.51), `keep` (0.51)
- **`hardware_repair_service`**: `repair` (1.71), `broken` (1.67), `replacement` (1.42), `applecare` (1.18), `an` (1.14), `apple care` (1.11)
- **`network_connectivity`**: `wifi` (3.40), `bluetooth` (1.98), `connect` (1.43), `off` (1.07), `turn` (1.02), `signal` (0.99)
- **`order_shipping`**: `order` (4.25), `shipping` (1.76), `delivery` (1.39), `confirmation` (0.91), `that` (0.89), `my order` (0.83)
- **`performance_freeze_crash`**: `freezing` (2.62), `freeze` (1.28), `restart` (1.26), `frozen` (1.24), `phone` (1.09), `restarting` (1.02)
- **`software_update`**: `ios` (3.01), `update` (2.95), `ios 11` (2.30), `11` (1.93), `after` (1.09), `updating` (1.08)

---

## 3. Historical Support Retrieval Performance

- **Indexed Historical Cases**: 800 resolved interactions.
- **Evaluation Isolation**: Complete. Zero golden set conversations indexed.
- **Mean Top-1 Cosine Similarity**: `0.2364`
- **Mean Top-3 Average Cosine Similarity**: `0.1964`

---

## 4. SupportPilot Agent Escalation Policy Benchmark

The agent policy balances operational containment (`AUTO_HANDLE`) with brand safety and security (`ESCALATE`).

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Decision Accuracy** | **76.50%** | Correctly assigned action vs ground truth |
| **Escalation Precision** | **60.24%** | Of cases escalated, how many truly required human escalation |
| **Escalation Recall** | **78.12%** | Of cases requiring escalation, how many were caught |
| **Escalation F1** | **68.03%** | Harmonic balance between safety and automation |

### Escalation Confusion Matrix

| | Predicted AUTO_HANDLE | Predicted ESCALATE | Total |
| :--- | :--- | :--- | :--- |
| **Actual AUTO_HANDLE** | **103** (True Negative) | 33 (Over-escalated) | 136 |
| **Actual ESCALATE** | 14 (Under-escalated) | **50** (True Positive) | 64 |
| **Total** | 117 | 83 | **200** |

### Performance by Difficulty Tier

| Difficulty Tier | Examples | Decision Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| `straightforward` | 88 | **68.18%** | High confidence single-intent queries |
| `ambiguous` | 89 | **85.39%** | Multi-symptom or underspecified queries |
| `edge_case` | 23 | **73.91%** | Non-English, slang, or hostile language |

---

## 5. Sample Agent Pipeline Executions

### Example GOLDEN_001: app_store_billing
- **Customer Query**: *"why are my App Store categories in Spanish but everything else in English??"*
- **Predicted Intent**: `app_store_billing` (Confidence: `0.20`)
- **Decision**: `ESCALATE` (Reason: `financial_and_billing`)
- **Rationale**: Escalated due to financial and billing policy (intent=app_store_billing, confidence=0.20).
- **Draft Reply**: *"Based on verified resolution for similar cases: Interesting. Let's look into this. Which iOS version is showing in Settings > General > About?..."*
- **Top Retrieved Match (Score 0.36)**: *"@117655 Interesting. Let's look into this. Which iOS version is showing in Settings > General > About?..."*

### Example GOLDEN_041: audio_sound
- **Customer Query**: *"Why does my Siri sound fucked up!!! Anyone else having this problem?? @AppleSupport"*
- **Predicted Intent**: `audio_sound` (Confidence: `0.14`)
- **Decision**: `AUTO_HANDLE` (Reason: `none`)
- **Rationale**: Eligible for auto-resolution: standard technical troubleshooting for audio_sound with sufficient confidence (0.14 >= 0.12).
- **Draft Reply**: *"Audio issues can definitely be frustrating, and we'd be glad to help! Please check that the Ring/Silent switch isn't set to silent, and verify that Do Not Disturb is disabled. Also..."*
- **Top Retrieved Match (Score 0.26)**: *"@119400 Having a reliable iPhone is important, and we want you to enjoy using it without worrying about battery life. Wh..."*

### Example GOLDEN_101: display_touch_keyboard
- **Customer Query**: *"fix your shit. See this I️ get to it girl it’s a glitch"*
- **Predicted Intent**: `display_touch_keyboard` (Confidence: `0.15`)
- **Decision**: `ESCALATE` (Reason: `vague_or_abusive`)
- **Rationale**: Escalated due to vague or abusive policy (intent=display_touch_keyboard, confidence=0.15).
- **Draft Reply**: *"We're here to help with your typing and display issue! If you are encountering the letter 'i' autocorrect symbol glitch, updating to the latest iOS release or adding a Text Replace..."*
- **Top Retrieved Match (Score 0.15)**: *"@118249 We're always ready to help, but we'll need more details to start. Reply by DM with which model is it and what's ..."*
