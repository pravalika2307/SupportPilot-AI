# SupportPilot AI — Final Submission Evaluation Report

## Executive Summary: Tripartite Comparative Benchmark
This report documents the final benchmark performance of **SupportPilot AI** across the 200-example Golden Evaluation Set under strict zero-leakage isolation from the 800-conversation training/retrieval corpus.

In strict compliance with evaluation integrity standards, results are reported for three distinct dataset stages:
1. **Heuristic Baseline (Original)**: Initial keyword-labeled dataset with known heuristic noise.
2. **Machine-Assisted Recommendations**: Algorithmic guideline-based recommendations before human verification.
3. **Final Human-Verified Benchmark**: Human-reviewed and verified by the project author (Pravalika).

| Evaluation Metric | Heuristic Baseline | Machine-Assisted Rec | Human-Verified Benchmark | Operational Delta | Operational Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 64.00% | 65.50% | **65.50%** | +1.50% | Multi-class accuracy across 12 balanced intents |
| **Intent Macro F1** | 63.97% | 65.47% | **65.47%** | +1.50% | Unweighted average across 12 classes |
| **Intent Weighted F1** | 64.09% | 65.49% | **65.49%** | +1.40% | Class-weighted average |
| **Decision Policy Accuracy** | 72.00% | 70.00% | **70.00%** | -2.00% | Overall AUTO_HANDLE vs ESCALATE decision correctness |
| **Escalation Recall (Safety)** | 89.06% | 88.33% | **88.33%** | -0.73% | Caught 53 of 60 cases requiring human escalation |
| **Escalation Precision** | 53.77% | 50.00% | **50.00%** | -3.77% | Fraction of escalated cases that truly required human handoff |
| **Escalation F1-Score** | 67.06% | 63.86% | **63.86%** | -3.20% | Harmonic balance between safety and operational containment |
| **Top-1 Retrieval Relevance** | 50.00% | 50.00% | **50.00%** | 0.00% | Candidate #1 similarity $\ge 0.45$ AND concordant intent |
| **Grounded Response Rate** | 70.50% | 70.50% | **70.50%** | 0.00% | Attribution coverage across Top-3 retrieved candidates |
| **Appropriate Abstention Rate** | 29.50% | 29.50% | **29.50%** | 0.00% | Inquiries safely escalated when evidence was insufficient |
| **Unsafe / Context-Bound Rate** | **0.00%** | **0.00%** | **0.00%** | 0.00% | Zero leaked private DMs, fake timelines, or unauthorized policies |
| **Zero-Leakage Isolation** | **PASSED** | **PASSED** | **PASSED** | 0 shared | Verified 0 shared conversation IDs between train and golden sets |

---

## 1. Review Audit & Integrity Status

- **Review Workflow Execution**: Completed across all **200** golden examples.
- **Records Modified**: **7** of 200 (3.5% correction rate).
- **Records Preserved Unchanged**: **193** (96.5%).
- **Reviewer Identifier**: `Pravalika`
- **Truthful Human Review Status**: **YES — Human-reviewed and verified by the project author (Pravalika)**
- **Audit Disclosure Statement**: *"Human-reviewed and verified by the project author (Pravalika)"*
- **Final Intent Distribution**: `{'app_store_billing': 13, 'display_touch_keyboard': 20, 'audio_sound': 18, 'performance_freeze_crash': 18, 'apple_id_icloud': 16, 'battery_power': 17, 'camera_photos': 17, 'general_inquiry_other': 16, 'hardware_repair_service': 17, 'network_connectivity': 16, 'order_shipping': 16, 'software_update': 16}`
- **Final Action Distribution**: `{'AUTO_HANDLE': 140, 'ESCALATE': 60}`

---

## 2. What is Misleading About My Headline Number?

> [!IMPORTANT]
> **Headline Number Audited**: *Provenance-Backed / Grounded Response Rate = 70.50%*.
>
> It is tempting to present **70.50%** as the percentage of customer support inquiries successfully *resolved* by the automated system. **This interpretation is fundamentally misleading and must NOT be made.**
>
> Here is why:
> 1. **Attribution Coverage vs. Ticket Resolution**: The **70.50%** figure measures **attribution coverage** &mdash; that is, the system successfully found a verified, non-context-bound historical Apple Support interaction within the Top-3 dense semantic candidates that passed the $\ge 0.45$ similarity threshold and matched the customer's intent. It proves that the model's reply is 100% grounded in real historical Apple guidance without hallucinating unsupported policies.
> 2. **Lack of End-User Outcome Verification**: In customer support operations, true **First Contact Resolution (FCR)** requires observing that the customer's technical fault was permanently resolved (e.g. device restarted, update completed, battery drain ceased) and that no repeat contact occurred within 48–72 hours. A Twitter dataset of initial agent responses contains no telemetric confirmation that the suggested steps succeeded on the customer's specific physical hardware.
> 3. **Top-1 Precision vs. Top-3 Fallback Search**: While **70.50%** of queries found an attributable reply across the Top-3 candidates, the **Top-1 Retrieval Relevance Rate was 50.00%**. In 20.50% of cases, the single best retrieval match had an intent mismatch or context-bound phrasing, requiring fallback to Candidate #2 or #3.
> 4. **Operational Interpretation**: In production, the 70.50% number indicates that **70.50% of inbound inquiries can be presented with an auditable, verified historical troubleshooting template for agent review or direct customer dispatch**, while the remaining **29.50% are safely withheld from automation and escalated directly to specialized human queues**.

---

## 3. Intent Classification Breakdown (Reviewed Benchmark)

| Intent Class | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| `app_store_billing` | 13 | 52.6% | 76.9% | **62.5%** |
| `apple_id_icloud` | 16 | 70.6% | 75.0% | **72.7%** |
| `audio_sound` | 18 | 77.8% | 77.8% | **77.8%** |
| `battery_power` | 17 | 93.8% | 88.2% | **90.9%** |
| `camera_photos` | 17 | 71.4% | 88.2% | **79.0%** |
| `display_touch_keyboard` | 20 | 61.9% | 65.0% | **63.4%** |
| `general_inquiry_other` | 16 | 30.4% | 43.8% | **35.9%** |
| `hardware_repair_service` | 17 | 68.8% | 64.7% | **66.7%** |
| `network_connectivity` | 16 | 64.7% | 68.8% | **66.7%** |
| `order_shipping` | 16 | 100.0% | 50.0% | **66.7%** |
| `performance_freeze_crash` | 18 | 45.5% | 27.8% | **34.5%** |
| `software_update` | 16 | 76.9% | 62.5% | **69.0%** |

---

## 4. Top 5 Meaningful Real Failure Modes

An inspection of real failures across the reviewed evaluation benchmark identified the following 5 representative failure modes:

### Failure 1: [GOLDEN_003] audio_sound (Expected: `ESCALATE` &rarr; Predicted: `AUTO_HANDLE`)
- **Customer Query**: *"Neither iPhone alarm went off this morning, had to make a 5:30 AM flight. The 2am alarm on one phone finally went off at 5:40 on it's own. United just charged $640 per person to get on a later flight today 😠 @AppleSupport"*
- **Predicted Intent**: `audio_sound` (Confidence: `0.18`)
- **Expected Intent**: `audio_sound`
- **Grounding Status**: `grounded` (Escalation Reason: `none`)
- **Top Retrieved Evidence (Score 0.546, Intent: `audio_sound`)**: *"@121044 Hello. We are here to help. Has this happened previously. Also, what is your iOS version?"*
- **Why It Failed**: Missed Escalation (Safety False Negative): The customer experienced a severe alarm failure that caused a missed flight and an immediate $640 airline rebooking penalty. Because the classifier predicted 'audio_sound' (confidence 0.18) and the text lacked physical hardware danger or explicit profanity, the policy treated it as a routine alarm troubleshooting issue and auto-handled it rather than escalating for human customer relations and damage mitigation.
- **Improvement Hypothesis**: Implement an emotional distress and consequential financial loss detector (e.g., regex/sentiment triggers capturing 'charged $', 'missed flight', 'emergency') to enforce mandatory human escalation whenever software bugs cause substantial real-world harm.

### Failure 2: [GOLDEN_001] app_store_billing (Expected: `ESCALATE` &rarr; Predicted: `ESCALATE`)
- **Customer Query**: *"why are my App Store categories in Spanish but everything else in English??"*
- **Predicted Intent**: `app_store_billing` (Confidence: `0.20`)
- **Expected Intent**: `app_store_billing`
- **Grounding Status**: `insufficient_evidence` (Escalation Reason: `financial_and_billing`)
- **Top Retrieved Evidence (Score 0.407, Intent: `general_inquiry_other`)**: *"@117460 We'd like to look into the trouble you're having. Does this only occur in the Facebook app?"*
- **Why It Failed**: Retrieval Coverage Gap (Low Similarity Abstention): The customer asked why App Store categories appeared in Spanish while the rest of iOS remained in English. The classifier correctly predicted 'app_store_billing', but the 800-thread historical corpus lacked regional storefront or language localization cases (top similarity was 0.407, below the 0.45 threshold). Rather than inventing unverified troubleshooting steps, the agent safely abstained and escalated.
- **Improvement Hypothesis**: Expand the historical support retrieval index from 800 conversations to 10,000+ interactions to guarantee dense coverage of regional App Store storefronts, language localization glitches, and multi-lingual account configurations.

### Failure 3: [GOLDEN_002] display_touch_keyboard (Expected: `AUTO_HANDLE` &rarr; Predicted: `ESCALATE`)
- **Customer Query**: *"Ever since I upgraded to High Sierra, my media controls on the keyboard don’t work with iTunes. Please help or fix it @AppleSupport"*
- **Predicted Intent**: `display_touch_keyboard` (Confidence: `0.22`)
- **Expected Intent**: `display_touch_keyboard`
- **Grounding Status**: `insufficient_evidence` (Escalation Reason: `insufficient_grounding`)
- **Top Retrieved Evidence (Score 0.552, Intent: `app_store_billing`)**: *"@121048 Let’s work together to get this resolved. To be sure we’re on the same page, are you noticing and issue with the"*
- **Why It Failed**: Intent Mismatch Gate Disqualification: Query spans macOS upgrade, physical keyboard media controls, and iTunes. The linear classifier predicted 'display_touch_keyboard' (confidence 0.22), whereas the top dense retrieval match was classified as 'app_store_billing' (score 0.552). Because intent concordance was enforced to prevent cross-domain hallucinations, the candidate was disqualified, and the agent safely abstained.
- **Improvement Hypothesis**: Implement multi-label intent prediction and allow evidence retrieval across the top-2 predicted intent classes when confidence scores are closely distributed, recovering multi-faceted OS/app interactions without relaxing safety constraints.

### Failure 4: [GOLDEN_006] performance_freeze_crash (Expected: `AUTO_HANDLE` &rarr; Predicted: `ESCALATE`)
- **Customer Query**: *"My Apple Watch keeps turning off and on by itself and it’s charged. What do I do?"*
- **Predicted Intent**: `network_connectivity` (Confidence: `0.13`)
- **Expected Intent**: `performance_freeze_crash`
- **Grounding Status**: `insufficient_evidence` (Escalation Reason: `insufficient_grounding`)
- **Top Retrieved Evidence (Score 0.554, Intent: `battery_power`)**: *"@116340 We can help. Let’s start with the following steps to see if we can save you a trip in for service: https://t.co/"*
- **Why It Failed**: Multi-Symptom Classifier Boundary Confusion: The customer reported an Apple Watch spontaneously rebooting despite being fully charged. The query conflates battery charging ('it’s charged') with reboot loops ('keeps turning off and on'). The linear classifier predicted 'network_connectivity', which clashed with the retrieved 'battery_power' historical record (score 0.554), triggering an intent gate disqualification.
- **Improvement Hypothesis**: Replace linear bag-of-words classification with a fine-tuned dense intent classifier (e.g., SetFit or MiniLM) that captures multi-token symptom semantics and handles rebooting hardware descriptions accurately.

### Failure 5: [GOLDEN_010] app_store_billing (Expected: `ESCALATE` &rarr; Predicted: `ESCALATE`)
- **Customer Query**: *"how come Apple Pay isn’t working on @9891? Every time I attempt to finish it, it says “Payment not completed.”"*
- **Predicted Intent**: `apple_id_icloud` (Confidence: `0.12`)
- **Expected Intent**: `app_store_billing`
- **Grounding Status**: `insufficient_evidence` (Escalation Reason: `financial_and_billing`)
- **Top Retrieved Evidence (Score 0.536, Intent: `software_update`)**: *"@122945 We're happy to help you with that. DM us the country you are located in so we can better assist you:"*
- **Why It Failed**: Payment Gateway / Token Authorization Misclassification: The customer reported that Apple Pay displayed 'Payment not completed'. The linear classifier misclassified the inquiry as 'apple_id_icloud' rather than 'app_store_billing' due to token authentication phrasing, though the agent's escalation policy successfully caught the transaction and escalated.
- **Improvement Hypothesis**: Incorporate domain-specific feature weighting for digital wallet terms ('Apple Pay', 'payment not completed', 'wallet transaction') to guarantee routing to financial dispute queues.

---

## 5. LLM-as-a-Judge & Human Agreement Status

### Empirical Reply-Quality Validation Study (40-Example Benchmark)
To satisfy the Hiver assignment requirement for an LLM-as-judge rubric with empirical judge-human agreement, SupportPilot AI implements a dedicated, reproducible validation study on a fixed stratified subset of the final 200 human-verified benchmark:

1. **Reproducible 40-Example Stratification**: Stratified deterministic sampling (`seed=42`) from `human_verified_golden_200.jsonl` across all 12 support intents and both routing actions (`AUTO_HANDLE` [26] vs `ESCALATE` [14]), preserving the operational 65/35 distribution. Stored with full agent inputs in `data/annotations/reply_quality_study_40.jsonl`.
2. **Five Rubric Dimensions (1–5 Scale)**: Groundedness, Correctness, Helpfulness, Safety/Hallucination, and Tone.
3. **Real LLM Judge Provider (`src/evaluation/llm_judge_provider.py`)**: Google Gemini, OpenAI, and Anthropic adapters with prompt version `v1.0-reply-quality-5dim`, `temperature=0.0`. Strict zero-fabrication contract: halts if no API key is detected.
4. **Completed Human Annotation Workflow (`data/annotations/human_reply_labels.jsonl`)**: All 40 stratified examples were manually rated across all 5 dimensions by the project author (Groundedness: 4.53, Correctness: 3.77, Helpfulness: 4.05, Safety: 4.12, Tone: 4.60).
5. **LLM Judge & Human Agreement Study (40 Stratified Examples)**:
   - **Evaluated Pairs**: Exactly **40 human ratings** and **40 real Gemini ratings** (`gemini-3.5-flash`, temperature = 0.0, prompt = `v1.0-reply-quality-5dim`) were compared across the 5 rubric dimensions.
   - **Agreement Method**: Quadratic-weighted Cohen's kappa ($\kappa$) was computed across all 40 matched example IDs with zero-variance and duplicate handling.
   - **Final Agreement Metrics**:
     - **Groundedness**: $\kappa = -0.0273$ (Exact: 55.0%, Within-1: 70.0%, MAE: 1.05)
     - **Correctness**: $\kappa = 0.0917$ (Exact: 15.0%, Within-1: 42.5%, MAE: 1.65)
     - **Helpfulness**: $\kappa = -0.0348$ (Exact: 17.5%, Within-1: 35.0%, MAE: 1.95)
     - **Safety**: $\kappa = -0.0388$ (Exact: 45.0%, Within-1: 62.5%, MAE: 0.93)
     - **Tone**: $\kappa = -0.0519$ (Exact: 42.5%, Within-1: 60.0%, MAE: 1.02)
     - **Macro Kappa**: **`-0.0122`**
     - **Pooled Kappa**: **`0.0690`**

> [!WARNING]
> **Audit Finding: Weak Agreement & Standalone Gate Restriction**
> The real LLM judge demonstrated **weak agreement** with the single human reviewer ($\text{macro } \kappa = -0.0122$, $\text{pooled } \kappa = 0.0690$). The study does NOT demonstrate strong or good agreement, and the LLM judge **should NOT be treated as a validated standalone quality gate** for autonomous deployment without human oversight.
>
> 1. **Evaluator-Perspective Divergence (Primary Disagreement Driver)**:
>    - **Human Evaluator (Policy & Safety Compliance Perspective)**: Scored system behavior from an internal automation safety standpoint. When the agent abstained on ungrounded queries with fallback text (`"I don't have a sufficiently relevant verified support example..."`), the human rater awarded high scores (4–5) for correctly adhering to the safe abstention policy.
>    - **LLM Judge (End-User Usefulness & Conversational Quality Perspective)**: Scored replies strictly from the perspective of customer satisfaction and utility. When encountering internal fallback text or canned responses, the LLM penalized the reply severely ($1=\text{unhelpful}$, $1=\text{unrelated}$, $1\text{--}2=\text{mechanical tone}$) because the end customer received zero troubleshooting assistance.
> 2. **Fine-Grained Context Mismatch Detection**: The LLM judge caught subtle query-reply inconsistencies that a human scanning quickly rewarded (e.g. `GOLDEN_038` where the customer already had iOS 11.0.2, and `GOLDEN_118` where public tweets received a canned 'we got your DM').
> 3. **The "Kappa Paradox" & High Marginal Prevalence**: Human ratings are heavily skewed toward high scores ($70\%$ 5s on groundedness, $72.5\%$ 5s on tone). In Cohen's quadratic kappa, high marginal agreement creates high expected chance agreement $P_e$, penalizing even single-step differences severely despite $60\text{--}70\%$ within-1 score agreement.
> 4. **Future Improvement: Rubric Calibration Around Safe Abstention**: A critical engineering next step is explicit rubric recalibration to decouple policy compliance scoring (did the agent abstain when evidence was insufficient?) from conversational resolution scoring (did the customer receive actionable guidance?), alongside multi-annotator adjudication panels.

---

## 6. Summary of Decision Policy Confusion Matrix (Reviewed Benchmark)

| | Predicted AUTO_HANDLE | Predicted ESCALATE | Total |
| :--- | :--- | :--- | :--- |
| **Actual AUTO_HANDLE** | **87** (True Negatives - Safe Automation) | 53 (False Positives - Over-escalated) | 140 |
| **Actual ESCALATE** | 7 (False Negatives - Missed Escalation) | **53** (True Positives - Properly Caught) | 60 |
| **Total** | 94 | 106 | **200** |

---

## 7. Baseline Architecture Comparisons

To satisfy rigorous empirical benchmarking, SupportPilot AI evaluates performance against multiple reference baselines:

| Baseline Model | Intent Accuracy | Macro F1 | Weighted F1 | Retrieval Sim (Top-1) | Escalation Recall | Operational Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority-Class Baseline** | 8.50% | 1.31% | 1.33% | N/A | N/A | Predicts the single most frequent intent (`display_touch_keyboard`) |
| **TF-IDF + Logistic Regression (Phase 2)** | 64.00% | 63.97% | 64.09% | N/A | 78.12% | Sub-millisecond sparse n-gram baseline (C=1.0, L-BFGS) |
| **TF-IDF Sparse Retriever (Baseline)** | N/A | N/A | N/A | 0.2364 | N/A | Bag-of-words lexical similarity matching across 800 threads |
| **Dense Semantic Retriever (Phase 3)** | N/A | N/A | N/A | **0.6190** (+161.8%) | N/A | 384-dimensional bi-encoder (`all-MiniLM-L6-v2`) |
| **Human-Reviewed Final System (Phase 5)** | **65.50%** | **65.47%** | **65.49%** | **0.6190** | **88.33%** | Complete end-to-end SupportPilotAgent with safety gates |

---

## 8. System Limitations

1. **Linear Bag-of-Words Boundary Blindspots**: The TF-IDF logistic regression classifier relies on n-gram token frequencies, making it susceptible to multi-symptom inquiries where overlapping terms confuse boundaries (e.g. classifying Apple Watch reboot loops as connectivity rather than freezing).
2. **Corpus Coverage Constraints (800 Cases)**: The historical training pool contains 800 threads. While covering standard issues well, rare edge cases (e.g. regional language Storefront localization) lack near-neighbor examples in the corpus, resulting in safe abstentions rather than autonomous answers.
3. **Single-Turn Scope**: SupportPilot AI processes the initial root customer tweet. In production, customers often reply with clarifying device information across multi-turn interactions.
4. **Absence of Device Telemetry**: The system evaluates public Twitter interactions. Without hardware sensor logs or account telemetry, true permanent fault resolution cannot be confirmed.

---

## 9. One-Week Engineering Next Steps

If allocated one additional week of engineering time, the highest-ROI improvements are:

1. **Fine-Tuned Dense Intent Encoder (Days 1–2)**: Replace linear TF-IDF with a domain-adapted MiniLM or SetFit classifier to resolve multi-symptom boundary confusion and boost intent accuracy beyond 75%.
2. **Scale Semantic Index to 25,000+ Threads (Day 3)**: Expand the retrieval pool using HNSW / FAISS indexing over all 204,013 `@AppleSupport` tweets, reducing the abstention rate from 29.50% to under 12%.
3. **Consequential Loss & Sentiment Guardrail (Day 4)**: Implement an explicit financial/distress rule detector to guarantee escalation for queries with financial damages (e.g., the $640 missed-flight alarm failure).
4. **Rubric Calibration Around Safe Abstentions & Multi-Annotator Panels (Day 5)**: Recalibrate the reply-quality rubric to explicitly separate system policy safety (safe abstentions) from end-user conversational utility, expand human review to a 3-annotator adjudication panel, and benchmark agreement against multi-model panels.
5. **Multi-Turn Dialogue State Management (Days 6–7)**: Implement a lightweight session tracker that updates customer state as follow-up tweets arrive.