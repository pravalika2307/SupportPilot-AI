# SupportPilot AI — Golden Evaluation Set Annotation Guidelines

## 1. Golden Set Integrity Audit & Truthful Disclosure

> [!NOTE]
> **Audit Finding & Verification History**: The initial 200 examples in `data/golden_eval/golden_eval_200.jsonl` were generated using **automated stratified heuristic rules** and keyword-based escalation tagging across AppleSupport conversation trees.
>
> Consequently, the baseline contained heuristic labeling noise. For example:
> - Queries containing the word *"charged"* (e.g., *"alarm didn't go off, airline charged $640 for rebooking"*) were heuristically mapped to `app_store_billing` instead of `audio_sound`.
> - Queries mentioning *"iTunes"* keyboard media controls under macOS were heuristically tagged `app_store_billing` rather than `display_touch_keyboard`.
>
> To establish a rigorous gold standard, all 200 examples were genuinely reviewed and verified by the project author (Pravalika) using the protocol defined below and the Annotation Studio (`src/data/annotation_studio.html`), persisted to `data/golden_eval/human_verified_golden_200.jsonl`.

---

## 2. Intent Classification Taxonomy (12 Classes)

Human annotators must assign exactly one primary intent to each customer query:

| Intent Class | Definition | Boundary & Disambiguation Rules |
| :--- | :--- | :--- |
| `app_store_billing` | In-app purchases, subscriptions, refunds, Apple Pay, App Store payment errors. | Must involve financial transactions or the App Store purchasing platform. An airline charging money due to an alarm bug is NOT `app_store_billing`. |
| `apple_id_icloud` | Apple ID login, two-factor authentication, iCloud storage, Keychain, Apple ID lockouts. | Focuses on identity, authentication, and cloud sync. If locked out of device passcode (not Apple ID), tag as `general_inquiry_other` or `performance_freeze_crash`. |
| `audio_sound` | Speakers, microphone, AirPods pairing/audio, distortion, call volume, alarm volume. | Audio hardware or software sound routing issues. |
| `battery_power` | Rapid battery drain, battery health degradation, charging cables, power adapters. | If the device refuses to turn on at all, prefer `performance_freeze_crash` or `hardware_repair_service`. |
| `camera_photos` | Blurry photos, camera app black screen, flash issues, Photos app iCloud syncing. | Camera sensor or photo library behavior. |
| `display_touch_keyboard` | Unresponsive touchscreen, ghost touches, display discoloration, keyboard layout/typing lag. | Physical screen responsiveness or on-screen/hardware typing issues. |
| `general_inquiry_other` | Trade-in questions, warranty status queries, device compatibility, product recommendations. | Use ONLY when no technical fault or specialized category applies. |
| `hardware_repair_service` | Cracked glass, water damage, swollen battery, physical enclosure dent, Genius Bar reservations. | Any physical destruction or repair service request. |
| `network_connectivity` | Wi-Fi dropping, Bluetooth disconnects, Cellular data failure, "No Service" carrier errors. | Wireless and cellular networking interfaces. |
| `order_shipping` | Apple Online Store orders, package tracking, delivery delays, return authorizations. | Logistics and physical shipment of physical Apple hardware. |
| `performance_freeze_crash` | iOS spinning wheel, total device freeze, bootloops (Apple logo loop), apps crashing to home screen. | Critical system instability or unresponsive OS state. |
| `software_update` | iOS / macOS installation errors, storage verification failed during update, backup restore. | Explicit OS version update and installation workflows. |

---

## 3. Decision Policy Guidelines (`AUTO_HANDLE` vs. `ESCALATE`)

### Eligible for `AUTO_HANDLE` (Self-Service Automation)
An inquiry is eligible for automated handling **only** if:
1. The problem can be diagnosed and addressed through standard user-facing troubleshooting steps (e.g. force reboot, toggle setting, update iOS, reset network settings).
2. Official public Apple Support documentation exists for the resolution.
3. No customer-specific account access, PII, physical hardware repair, or financial transaction is needed.

### Mandatory `ESCALATE` (Human Agent Routing)
An inquiry must be escalated immediately if it triggers any of the following 6 standardized categories:

1. **`hardware_physical_damage`**: Cracked screens, liquid ingress, dropped phones, physical repairs requiring Genius Bar or mail-in service.
2. **`financial_and_billing`**: Disputed credit card charges, refund requests, App Store billing errors, Apple Pay transaction failures.
3. **`account_security_and_pii`**: Stolen devices, locked Apple IDs, hacked accounts, 2FA bypass, sharing passwords or personal credentials.
4. **`unresolved_system_crash`**: Critical hardware faults, persistent boot loops, unrecoverable iOS corruption after troubleshooting.
5. **`vague_or_abusive`**: Abusive/obscene language, legal action threats, or queries so underspecified that troubleshooting is impossible.
6. **`non_english_query`**: Inquiries written in languages other than English requiring localized routing.

---

## 4. Query Difficulty Rating

Annotators must tag each sample with one difficulty level:
- **`straightforward`**: Unambiguous problem statement, single symptom, standard solution path.
- **`ambiguous`**: Incomplete information, multi-symptom query, or unclear device context.
- **`edge_case`**: Extreme scenarios (device caught fire, angry customer demanding CEO escalation, rare legacy OS bug).

---

## 5. Human Review Tool & Annotation Workflow

To review and correct the 200 Golden Evaluation set:

1. Run the interactive CLI review tool:
   ```bash
   python -m src.data.human_review_tool
   ```
2. The tool presents each golden example with its customer query, the heuristic intent, and the heuristic escalation action.
3. The annotator enters:
   - Verified Intent (or confirms heuristic intent)
   - Verified Action (`AUTO_HANDLE` / `ESCALATE`)
   - Verified Escalation Reason (if escalated)
   - Verified Difficulty rating
   - Annotator notes
4. The tool outputs a validated dataset to `data/golden_eval/human_verified_golden_200.jsonl` with an audit manifest tracking annotator ID, timestamp, and label modifications.
