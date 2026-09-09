# SupportPilot AI - AppleSupport Dataset & Intent Taxonomy Analysis (Phase 1)

This report documents the exploratory data analysis, conversation tree reconstruction, recurring customer problem discovery, and proposed intent taxonomy for building **SupportPilot AI**, centered around the **AppleSupport** domain from the *Customer Support on Twitter* (`twcs.csv`) dataset.

---

## 1. Executive Summary & Dataset Findings

| Metric | Raw Dataset Value | Analysis Scope |
|---|---|---|
| Total Tweets in TWCS | 2,811,774 | Full dataset scanned |
| AppleSupport Mentions / Authored Tweets | 204,013 | 7.25% of total dataset |
| Raw Tweets Ingested in Current Scan | 31,401 | Streaming window |
| Reconstructed Customer-Agent Conversations | 7,488 | Root inquiry + Apple response |
| Development Sample Created | 1,000 conversations | Stratified sample |
| Target Brand | `@AppleSupport` | Primary focal entity |

### Key Observations:
1. **Dominance of AppleSupport**: Apple operates by far the highest-volume support handle on Twitter in this dataset, with over 204,000 interactions.
2. **Asynchronous Multi-turn Dialogue**:
   - Inquiries originate as public tweets to `@AppleSupport`.
   - The majority of conversations are resolved or transitioned within 2 to 3 turns (`mean: 3.26` turns).
3. **Agent Redirection & Triage Policy**:
   - **42.86%** of agent first responses redirect the user to a private Direct Message (DM) with a specialized link (`https://t.co/GDrqU22YpT`).
   - **23.86%** of replies proactively query the customer's exact iOS version (e.g. *Settings > General > About*).
   - **14.22%** prompt for the hardware model.

---

## 2. Recurring Customer Problems & Temporal Context

The dataset reflects customer inquiries during the **October - November 2017** window. During this period, several major product and software updates occurred:
1. **iOS 11 Launch Bugs**:
   - **Battery Life Complaints**: Rapid battery drainage after updating to iOS 11.0 / 11.0.3 was the #1 explicit technical complaint.
   - **The "I" Autocorrect Glitch**: A prominent bug in iOS 11.1 where typing the capital letter "I" autocorrected to "A [?]" or symbols.
   - **UI & Keyboard Stutter**: Lock screen freezes, sluggish animations, and keyboard lag on older devices (iPhone 6, 6s).
2. **iPhone X & iPhone 8 Release**:
   - High volume of shipping, pre-order status, reservation confirmation, and delivery date questions.
3. **Evergreen Technical Inquiries**:
   - **Apple ID / iCloud Security**: Locked accounts, two-factor authentication (2FA) verification code delivery issues, and forgotten passcodes.
   - **App Store / Billing**: Unrecognized subscriptions, duplicate charges, in-app purchases, and refund policies.
   - **Physical Hardware / Genius Bar**: Cracked displays, water damage, repair costs, and booking appointments at retail Apple Stores.
   - **Audio & Connectivity**: AirPods pairing drops, Wi-Fi toggles greyed out, earpiece audio distortion.

---

## 3. Top Keywords in Customer Inquiries

Top non-stopword tokens extracted from cleaned customer initial problem statements:

| Rank | Keyword | Frequency | Relevance / Context |
|---|---|---|---|
| 1 | `iphone` | 2,067 | Recurring technical issue token |
| 2 | `ios` | 1,622 | Recurring technical issue token |
| 3 | `phone` | 1,552 | Recurring technical issue token |
| 4 | `update` | 1,164 | Recurring technical issue token |
| 5 | `apple` | 932 | Recurring technical issue token |
| 6 | `help` | 812 | Recurring technical issue token |
| 7 | `battery` | 794 | Recurring technical issue token |
| 8 | `new` | 727 | Recurring technical issue token |
| 9 | `fix` | 711 | Recurring technical issue token |
| 10 | `please` | 662 | Recurring technical issue token |
| 11 | `now` | 606 | Recurring technical issue token |
| 12 | `from` | 584 | Recurring technical issue token |
| 13 | `all` | 582 | Recurring technical issue token |
| 14 | `since` | 552 | Recurring technical issue token |
| 15 | `app` | 541 | Recurring technical issue token |

---

## 4. Proposed 10-Intent Taxonomy

Based on empirical clustering of customer problem statements and agent action trees, we propose the following 10 data-derived intents:

| Intent Key | Canonical Name | Prevalence in Data | Key Problem Signatures |
|---|---|---|---|
| `general_inquiry_other` | General Inquiry Other | 33.8% (2,531) | High-volume operational intent |
| `software_update` | Software Update | 26.7% (2,002) | High-volume operational intent |
| `battery_power` | Battery Power | 12.8% (961) | High-volume operational intent |
| `device_performance` | Device Performance | 5.6% (417) | High-volume operational intent |
| `display_touch_keyboard` | Display Touch Keyboard | 5.2% (386) | High-volume operational intent |
| `apple_id_icloud` | Apple Id Icloud | 4.7% (350) | High-volume operational intent |
| `app_store_billing` | App Store Billing | 3.0% (224) | High-volume operational intent |
| `camera_photos` | Camera Photos | 2.4% (176) | High-volume operational intent |
| `network_connectivity` | Network Connectivity | 2.2% (165) | High-volume operational intent |
| `hardware_repair_service` | Hardware Repair Service | 1.8% (135) | High-volume operational intent |
| `audio_sound` | Audio Sound | 1.3% (97) | High-volume operational intent |
| `order_shipping` | Order Shipping | 0.6% (44) | High-volume operational intent |

### Detailed Taxonomy Specifications:

1. **`battery_power`**:
   - *Scope*: Rapid discharge, device overheating, charging cable failure, device shutting down at 20-30%.
   - *Example*: *"My iPhone 7 battery is draining from 100% to dead in 2 hours since updating. What can I do?"*
   - *Resolution Route*: Diagnostic check, battery health verification, low-power mode guidance.

2. **`software_update`**:
   - *Scope*: iOS installation errors ("Unable to Verify Update"), firmware restore questions, downgrade queries.
   - *Example*: *"I can't get the new iOS 11.1 update to download on my iPad, it just says error."*
   - *Resolution Route*: Storage space verification, iTunes/Finder update steps.

3. **`device_performance`**:
   - *Scope*: Device freezing, general UI lag, apps force-closing, spinning wheel, storage full warnings.
   - *Example*: *"My phone is lagging terribly and keeps freezing when opening any app."*
   - *Resolution Route*: Soft reset / force restart instructions, cache clearing.

4. **`apple_id_icloud`**:
   - *Scope*: Account locked for security reasons, 2FA SMS code not received, password reset, iCloud storage sync.
   - *Example*: *"My Apple ID has been disabled and I'm locked out of my account. Please help me reset it."*
   - *Resolution Route*: iForgot portal redirection, security question verification (Strict Escalation flag if account security issue).

5. **`app_store_billing`**:
   - *Scope*: Unrecognized charges, subscription renewal disputes, refund requests, declined payment methods.
   - *Example*: *"I was charged $9.99 for an app subscription I never signed up for. How do I get a refund?"*
   - *Resolution Route*: reportaproblem.apple.com link, Subscriptions settings walkthrough.

6. **`display_touch_keyboard`**:
   - *Scope*: Unresponsive touchscreen, ghost touches, keyboard autocorrect bug (letter 'I'), black screen.
   - *Example*: *"Every time I type the letter I on my iPhone it changes to a weird symbol."*
   - *Resolution Route*: Text Replacement workaround, display reset.

7. **`network_connectivity`**:
   - *Scope*: Wi-Fi disconnecting, Bluetooth drops with car/headphones, Cellular 'No Service' / 'Searching'.
   - *Example*: *"My iPhone won't connect to my home Wi-Fi anymore after the update. Wi-Fi button is greyed out."*
   - *Resolution Route*: Reset Network Settings (`Settings > General > Reset > Reset Network Settings`).

8. **`audio_sound`**:
   - *Scope*: No sound on phone calls, crackling speakers, low alarm volume, mic not picking up voice.
   - *Example*: *"People cannot hear me when I speak on phone calls unless I turn on speakerphone."*
   - *Resolution Route*: Clean microphone mesh, audio balance settings check.

9. **`hardware_repair_service`**:
   - *Scope*: Cracked screen replacement, water damage, warranty status, AppleCare coverage, Genius Bar booking.
   - *Example*: *"Dropped my phone and the screen shattered. How much does it cost to fix at the Apple Store?"*
   - *Resolution Route*: Apple Support app appointment link, pricing estimator.

10. **`order_shipping`**:
    - *Scope*: iPhone X / 8 pre-order tracking, delivery date disputes, reservation pickup time slots.
    - *Example*: *"My order status says preparing for shipment but delivery was promised today."*
    - *Resolution Route*: Order status portal, logistics carrier tracking.

11. **`general_inquiry_other`**:
    - *Scope*: Store hours, polite greetings, trade-in questions, non-technical queries.
    - *Example*: *"Are your retail stores open on Sunday?"*

---

## 5. Development Sample Verification

To allow zero-friction development and reproducible evaluation without needing the 500MB+ raw CSV:
- **`data/sample/applesupport_sample_1000.jsonl`**: Contains 1,000 multi-turn conversation threads.
- **`data/sample/applesupport_sample_1000.csv`**: Tabular export containing customer queries, clean text, agent responses, turn counts, and preliminary intent labels.

### Sample Intent Distribution:
| Intent | Count in Sample | Pct in Sample |
|---|---|---|
| `general_inquiry_other` | 126 | 12.6% |
| `display_touch_keyboard` | 83 | 8.3% |
| `camera_photos` | 83 | 8.3% |
| `audio_sound` | 83 | 8.3% |
| `software_update` | 83 | 8.3% |
| `apple_id_icloud` | 83 | 8.3% |
| `battery_power` | 83 | 8.3% |
| `network_connectivity` | 83 | 8.3% |
| `device_performance` | 83 | 8.3% |
| `app_store_billing` | 83 | 8.3% |
| `hardware_repair_service` | 83 | 8.3% |
| `order_shipping` | 44 | 4.4% |

---

## 6. Recommended Next Implementation Steps (Phase 2 & 3)

1. **Classification Baselines**:
   - Implement **Majority-Class Baseline** (simple rule predicting most frequent class).
   - Implement **TF-IDF + Logistic Regression Baseline** (using scikit-learn pipeline with n-grams 1-2).
2. **Golden Evaluation Dataset**:
   - Curate a high-quality **150–250 example golden dataset** from real AppleSupport customer inquiries with verified ground-truth labels for both `intent` and `escalation_decision` (`AUTO_HANDLE` vs `ESCALATE`).
3. **Retrieval Engine (RAG)**:
   - Index resolved historical conversations using TF-IDF / BM25 / dense embeddings.
   - Retrieve top-3 historically similar resolved customer problems.
4. **Agent Reply Generation & Escalation Logic**:
   - Ground response generation in retrieved historical evidence.
   - Implement clear deterministic + LLM escalation heuristics (e.g. account lockouts, legal/billing disputes, physical hardware repairs -> ESCALATE; troubleshooting guides -> AUTO-HANDLE).
5. **Evaluation Harness**:
   - Accuracy, Macro F1, Per-intent Precision/Recall/Confusion Matrix.
   - LLM-as-a-judge reply quality rubric (Groundedness, Helpfulness, Tone, Accuracy).
   - Human-vs-LLM agreement and top 5 failure modes analysis.
