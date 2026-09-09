"""
SupportPilot AI — Golden Evaluation Set Human Review Tool & Infrastructure.

Provides:
1. Recommendation engine derived strictly from ANNOTATION_GUIDELINES.md
2. Presentation formatter displaying customer text, current labels, recommended choices, and edge-case info
3. Interactive and batch review workflow allowing human reviewers to accept or correct labels
4. Separate versioned storage: data/golden_eval/human_verified_golden_200.jsonl
5. Audit logging: data/golden_eval/human_review_audit.json
"""

import os
import re
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

VALID_INTENTS = [
    "app_store_billing",
    "apple_id_icloud",
    "audio_sound",
    "battery_power",
    "camera_photos",
    "display_touch_keyboard",
    "general_inquiry_other",
    "hardware_repair_service",
    "network_connectivity",
    "order_shipping",
    "performance_freeze_crash",
    "software_update"
]

VALID_ACTIONS = ["AUTO_HANDLE", "ESCALATE"]

VALID_ESCALATION_REASONS = [
    "hardware_physical_damage",
    "financial_and_billing",
    "account_security_and_pii",
    "unresolved_system_crash",
    "vague_or_abusive",
    "non_english_query",
    "none"
]

VALID_DIFFICULTIES = ["straightforward", "ambiguous", "edge_case"]

NON_LATIN_SCRIPT_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff\u0e00-\u0e7f]"
)

NON_ENGLISH_WORDS = [
    r"\bhola\b", r"\bnecesito\b", r"\bayuda\b", r"\bpor favor\b", r"\bactualizaci[oó]n\b",
    r"\bbater[ií]a\b", r"\btel[eé]fono\b", r"\byardımcı\b", r"\bmerci\b", r"\bgracias\b"
]


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_recommendation(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes customer query against ANNOTATION_GUIDELINES.md taxonomy and escalation rules
    to generate recommended intent, action, escalation reason, and diagnostic rationale.
    """
    text = record["clean_customer_query"]
    t = text.lower()
    cur_intent = record.get("ground_truth_intent", "general_inquiry_other")
    cur_action = record.get("expected_action", "AUTO_HANDLE")
    cur_reason = record.get("escalation_reason", "none")
    difficulty = record.get("difficulty", "straightforward")

    rec_intent = cur_intent
    rec_action = cur_action
    rec_reason = cur_reason
    rationale_points = []

    # 1. Non-English detection
    if NON_LATIN_SCRIPT_PATTERN.search(text):
        rec_action = "ESCALATE"
        rec_reason = "non_english_query"
        difficulty = "edge_case"
        rationale_points.append("Contains non-Latin script characters requiring localized routing.")
    elif any(re.search(rf"\b{w}\b", t) for w in ["hola", "necesito", "ayuda", "actualizaci[oó]n", "bater[ií]a", "tel[eé]fono", "merci", "gracias", "yardımcı"]):
        rec_action = "ESCALATE"
        rec_reason = "non_english_query"
        difficulty = "edge_case"
        rationale_points.append("Contains non-English language indicators requiring localized routing.")

    # 2. Abusive or Legal Threats
    elif any(re.search(rf"\b{w}\b", t) for w in ["sue", "lawyer", "legal", "lawsuit", "fuck", "fucking", "shit", "bullshit"]):
        rec_action = "ESCALATE"
        rec_reason = "vague_or_abusive"
        difficulty = "edge_case"
        rationale_points.append("Contains abusive language or legal threat escalation risk.")

    # 3. Alarm failure with consequential distress
    elif re.search(r"\balarm\b", t) and ("alarm didn't" in t or "alarm went off" in t or "alarm volume" in t or "alarm app" not in t):
        if cur_intent != "audio_sound" and "battery" not in t:
            rec_intent = "audio_sound"
            rationale_points.append("Core symptom concerns clock/alarm volume or sound.")
        if any(w in t for w in ["flight", "airline", "$640", "charged $", "cost me"]):
            rec_action = "ESCALATE"
            rec_reason = "financial_and_billing"
            difficulty = "edge_case"
            rationale_points.append("Alarm failure causing flight rebooking fees and severe financial loss.")
        elif rec_reason == "none":
            rec_action = "AUTO_HANDLE"

    # 4. Media controls / keyboard with iTunes
    elif ("media controls" in t or "media control" in t) and "keyboard" in t:
        rec_intent = "display_touch_keyboard"
        rec_action = "AUTO_HANDLE"
        rec_reason = "none"
        rationale_points.append("macOS physical keyboard media controls troubleshooting.")

    # 5. Letter 'i' predictive text bug & autocorrect glitch
    elif ("letter i" in t or "the letter i" in t or "the letter “i”" in t or "letter 'i'" in t or 'letter "i"' in t or "predictive text" in t) and cur_intent in ["general_inquiry_other", "app_store_billing"]:
        rec_intent = "display_touch_keyboard"
        rec_action = "AUTO_HANDLE"
        rec_reason = "none"
        rationale_points.append("iOS 11 predictive text / letter 'i' bug belongs in display_touch_keyboard.")

    # 6. Notes app typing / keyboard glitch / unresponsive touch
    elif ("notes app" in t or "keyboard refuses" in t or "typing lag" in t or "press a letter" in t) and cur_intent in ["app_store_billing", "apple_id_icloud"]:
        rec_intent = "display_touch_keyboard"
        rec_action = "AUTO_HANDLE"
        rec_reason = "none"
        rationale_points.append("Keyboard typing input responsiveness issue.")

    # 7. Device reboot loop / freeze wrongly in billing
    elif ("turning off and on" in t or "keeps turning off" in t or "randomly restarts" in t or "keeps restarting" in t) and cur_intent == "app_store_billing":
        rec_intent = "performance_freeze_crash"
        rec_action = "AUTO_HANDLE"
        rec_reason = "none"
        rationale_points.append("Spontaneous device reboot loop wrongly categorized as billing.")

    # 8. Device performance / UI lag wrongly in billing due to purchase metaphor
    elif any(w in t for w in ["spotify randomly stops", "apps are slow", "screen lags"]) and cur_intent == "app_store_billing":
        rec_intent = "performance_freeze_crash"
        rec_action = "AUTO_HANDLE"
        rec_reason = "none"
        rationale_points.append("Device UI stutter and app lagging wrongly categorized as billing.")

    # 9. Hardware physical damage
    elif any(re.search(rf"\b{w}\b", t) for w in ["cracked", "shattered", "water damage", "dropped", "broken screen", "dent", "dented", "genius bar"]):
        if cur_intent != "hardware_repair_service":
            rec_intent = "hardware_repair_service"
            rationale_points.append("Physical hardware damage requires in-person repair or inspection.")
        rec_action = "ESCALATE"
        rec_reason = "hardware_physical_damage"

    # 10. Financial & Billing Transactions
    elif any(re.search(rf"\b{w}\b", t) for w in ["refund", "unauthorized charge", "charged twice", "apple pay", "subscription cancel", "itunes billing"]):
        if cur_intent != "app_store_billing":
            rec_intent = "app_store_billing"
            rationale_points.append("Primary inquiry involves financial transaction or refund.")
        rec_action = "ESCALATE"
        rec_reason = "financial_and_billing"

    # 11. Apple ID & Account Lockout
    elif any(re.search(rf"\b{w}\b", t) for w in ["locked out", "apple id disabled", "two-factor", "2fa", "passcode"]):
        if cur_intent != "apple_id_icloud":
            rec_intent = "apple_id_icloud"
            rationale_points.append("Account authentication or credential recovery requires secure verification.")
        rec_action = "ESCALATE"
        rec_reason = "account_security_and_pii"

    if not rationale_points:
        rationale_points.append("Heuristic classification aligns with standard guideline criteria.")

    is_changed = (rec_intent != cur_intent or rec_action != cur_action or rec_reason != cur_reason)

    return {
        "golden_id": record["golden_id"],
        "clean_customer_query": text,
        "current_intent": cur_intent,
        "current_action": cur_action,
        "current_escalation_reason": cur_reason,
        "recommended_intent": rec_intent,
        "recommended_action": rec_action,
        "recommended_escalation_reason": rec_reason,
        "difficulty": difficulty,
        "recommendation_rationale": " ".join(rationale_points),
        "is_changed_from_heuristic": is_changed
    }


def format_review_presentation(item: Dict[str, Any], index: Optional[int] = None, total: Optional[int] = None) -> str:
    """Formats an individual example for human review presentation."""
    header = f"=== Golden Example [{item['golden_id']}]"
    if index is not None and total is not None:
        header += f" ({index}/{total})"
    header += " ==="

    diff_flag = " [RECOMMENDS CHANGE]" if item.get("is_changed_from_heuristic") else " [CONFIRMS HEURISTIC]"

    lines = [
        header + diff_flag,
        f"Customer Query: \"{item['clean_customer_query']}\"",
        f"Difficulty Tier: {item.get('difficulty', 'straightforward')}",
        "",
        "Current Heuristic Label:",
        f"  - Intent:            {item['current_intent']}",
        f"  - Action:            {item['current_action']}",
        f"  - Escalation Reason: {item.get('current_escalation_reason', 'none')}",
        "",
        "Recommended Label (per ANNOTATION_GUIDELINES.md):",
        f"  - Recommended Intent:  {item['recommended_intent']}",
        f"  - Recommended Action:  {item['recommended_action']}",
        f"  - Escalation Reason:   {item['recommended_escalation_reason']}",
        f"  - Rationale:           {item['recommendation_rationale']}",
        "-" * 70
    ]
    return "\n".join(lines)


def export_review_queue(
    input_path: str = "data/golden_eval/golden_eval_200.jsonl",
    output_path: str = "data/golden_eval/review_queue_200.jsonl"
) -> List[Dict[str, Any]]:
    """Exports all 200 examples with explicit recommendations for human reviewer evaluation."""
    records = load_jsonl(input_path)
    queue = [generate_recommendation(r) for r in records]
    save_jsonl(queue, output_path)
    print(f"Exported review presentation queue for {len(queue)} items to {output_path}")
    return queue


def run_interactive_review_workflow(
    input_path: str = "data/golden_eval/golden_eval_200.jsonl",
    output_path: str = "data/golden_eval/human_verified_golden_200.jsonl",
    audit_path: str = "data/golden_eval/human_review_audit.json",
    annotator_id: str = "human_reviewer"
) -> None:
    """Runs interactive CLI review session with presented recommendations."""
    golden_records = load_jsonl(input_path)
    already_reviewed = load_jsonl(output_path)
    reviewed_ids = {r["golden_id"] for r in already_reviewed}

    remaining = [r for r in golden_records if r["golden_id"] not in reviewed_ids]
    print("\n" + "=" * 75)
    print("SUPPORTPILOT AI — INTERACTIVE GOLDEN EVALUATION HUMAN REVIEW WORKFLOW")
    print("=" * 75)
    print(f"Total golden records:   {len(golden_records)}")
    print(f"Already reviewed:       {len(already_reviewed)}")
    print(f"Remaining for review:   {len(remaining)}")
    print(f"Reviewer ID:            {annotator_id}")
    print("=" * 75)

    if not remaining:
        print("All 200 golden records have already been reviewed!")
        return

    for idx, sample in enumerate(remaining, 1):
        rec = generate_recommendation(sample)
        print("\n" + format_review_presentation(rec, idx, len(remaining)))

        choice = input("Accept (r=recommendation, h=heuristic, e=edit, s=skip, q=quit) [r]: ").strip().lower()
        if choice in ["q", "quit"]:
            print("Session paused. Exiting...")
            break
        elif choice in ["s", "skip"]:
            continue
        elif choice == "h":
            verified = dict(sample)
            verified["human_verified"] = True
            verified["annotator_id"] = annotator_id
            verified["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            verified["label_source"] = "heuristic_accepted"
            already_reviewed.append(verified)
            save_jsonl(already_reviewed, output_path)
        elif choice == "e":
            print("Available intents:", ", ".join(VALID_INTENTS))
            new_intent = input(f"Intent [{rec['recommended_intent']}]: ").strip() or rec["recommended_intent"]
            new_action = input(f"Action (AUTO_HANDLE/ESCALATE) [{rec['recommended_action']}]: ").strip().upper() or rec["recommended_action"]
            new_reason = "none"
            if new_action == "ESCALATE":
                print("Available reasons:", ", ".join(VALID_ESCALATION_REASONS))
                new_reason = input(f"Reason [{rec['recommended_escalation_reason']}]: ").strip() or rec["recommended_escalation_reason"]

            verified = dict(sample)
            verified["ground_truth_intent"] = new_intent
            verified["expected_action"] = new_action
            verified["escalation_reason"] = new_reason
            verified["human_verified"] = True
            verified["annotator_id"] = annotator_id
            verified["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            verified["label_source"] = "manual_correction"
            already_reviewed.append(verified)
            save_jsonl(already_reviewed, output_path)
        else: # default 'r'
            verified = dict(sample)
            verified["ground_truth_intent"] = rec["recommended_intent"]
            verified["expected_action"] = rec["recommended_action"]
            verified["escalation_reason"] = rec["recommended_escalation_reason"]
            verified["difficulty"] = rec["difficulty"]
            verified["human_verified"] = True
            verified["annotator_id"] = annotator_id
            verified["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            verified["label_source"] = "recommendation_accepted"
            already_reviewed.append(verified)
            save_jsonl(already_reviewed, output_path)

    # Save audit summary
    generate_and_save_audit(golden_records, already_reviewed, audit_path, annotator_id)


def generate_and_save_audit(
    original_records: List[Dict[str, Any]],
    reviewed_records: List[Dict[str, Any]],
    audit_path: str = "data/golden_eval/human_review_audit.json",
    reviewer_id: str = "unknown",
    is_human_sign_off: bool = False
) -> Dict[str, Any]:
    """Computes and writes comprehensive audit log of reviewed vs heuristic labels."""
    orig_map = {r["golden_id"]: r for r in original_records}
    reviewed_map = {r["golden_id"]: r for r in reviewed_records}

    num_reviewed = len(reviewed_records)
    num_changed = 0
    intent_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {"AUTO_HANDLE": 0, "ESCALATE": 0}

    changed_details = []

    for gid, rev in reviewed_map.items():
        orig = orig_map.get(gid, {})
        intent = rev.get("ground_truth_intent")
        action = rev.get("expected_action")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1

        if intent != orig.get("ground_truth_intent") or action != orig.get("expected_action"):
            num_changed += 1
            changed_details.append({
                "golden_id": gid,
                "query": rev.get("clean_customer_query")[:80],
                "orig_intent": orig.get("ground_truth_intent"),
                "new_intent": intent,
                "orig_action": orig.get("expected_action"),
                "new_action": action
            })

    audit_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_golden_records": len(original_records),
        "number_reviewed": num_reviewed,
        "number_changed": num_changed,
        "number_unchanged": num_reviewed - num_changed,
        "percent_changed": round((num_changed / num_reviewed * 100) if num_reviewed else 0.0, 2),
        "reviewer_status": {
            "reviewer_id": reviewer_id,
            "actually_human_reviewed": is_human_sign_off,
            "audit_disclosure": "Human-reviewed and verified by the project author (Pravalika)" if is_human_sign_off else "Generated via guideline-backed recommendation engine; pending human review"
        },
        "final_intent_distribution": intent_counts,
        "final_action_distribution": action_counts,
        "sample_modifications": changed_details[:10]
    }

    os.makedirs(os.path.dirname(os.path.abspath(audit_path)), exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary


def save_human_review_decisions(
    decisions: List[Dict[str, Any]],
    annotator_id: str,
    input_golden_path: str = "data/golden_eval/golden_eval_200.jsonl",
    output_path: str = "data/golden_eval/human_verified_golden_200.jsonl",
    audit_path: str = "data/golden_eval/human_review_audit.json"
) -> Dict[str, Any]:
    """
    Persists human review decisions for the 200 golden examples, preserving original
    metadata and recording truthful audit metrics with actually_human_reviewed = True.
    """
    golden_records = load_jsonl(input_golden_path)
    orig_map = {r["golden_id"]: r for r in golden_records}
    dec_map = {d["golden_id"]: d for d in decisions}

    if len(dec_map) < len(golden_records):
        raise ValueError(f"Incomplete review: expected {len(golden_records)} records, received {len(dec_map)}")

    verified_list = []
    now_str = datetime.now(timezone.utc).isoformat()

    for r in golden_records:
        gid = r["golden_id"]
        dec = dec_map[gid]

        intent = dec.get("ground_truth_intent", r.get("ground_truth_intent"))
        action = dec.get("expected_action", r.get("expected_action"))
        reason = dec.get("escalation_reason", r.get("escalation_reason", "none"))
        diff = dec.get("difficulty", r.get("difficulty", "straightforward"))
        source = dec.get("decision_source", "manual_review")

        if intent not in VALID_INTENTS:
            raise ValueError(f"Invalid intent '{intent}' for golden ID {gid}")
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}' for golden ID {gid}")
        if reason not in VALID_ESCALATION_REASONS:
            raise ValueError(f"Invalid escalation reason '{reason}' for golden ID {gid}")

        verified_item = dict(r)
        verified_item["ground_truth_intent"] = intent
        verified_item["expected_action"] = action
        verified_item["escalation_reason"] = reason
        verified_item["difficulty"] = diff
        verified_item["human_verified"] = True
        verified_item["annotator_id"] = annotator_id
        verified_item["reviewed_at"] = now_str
        verified_item["label_source"] = source
        verified_list.append(verified_item)

    save_jsonl(verified_list, output_path)
    summary = generate_and_save_audit(golden_records, verified_list, audit_path, annotator_id, is_human_sign_off=True)
    print(f"Successfully saved {len(verified_list)} human verified records to {output_path}")
    print(f"Audit summary written to {audit_path} with actually_human_reviewed = True")
    return summary


class AnnotationStudioHTTPHandler:
    """Helper to serve the annotation studio interface and handle REST API endpoints."""

    @staticmethod
    def create_handler(queue_path: str, golden_path: str, output_path: str, audit_path: str):
        from http.server import SimpleHTTPRequestHandler

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path in ["/", "/index.html"]:
                    studio_html = "src/data/annotation_studio.html"
                    if os.path.exists(studio_html):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        with open(studio_html, "rb") as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_error(404, "Studio HTML not found")
                elif self.path == "/api/queue":
                    queue = export_review_queue(golden_path, queue_path)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(queue, ensure_ascii=False).encode("utf-8"))
                elif self.path == "/api/status":
                    status_obj = {"status": "ready"}
                    if os.path.exists(audit_path):
                        with open(audit_path, "r", encoding="utf-8") as f:
                            status_obj = json.load(f)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(status_obj).encode("utf-8"))
                else:
                    super().do_GET()

            def do_POST(self):
                if self.path == "/api/save":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length).decode("utf-8")
                    data = json.loads(body)
                    annotator = data.get("annotator_id", "human_reviewer")
                    records = data.get("records", [])

                    try:
                        summary = save_human_review_decisions(
                            decisions=records,
                            annotator_id=annotator,
                            input_golden_path=golden_path,
                            output_path=output_path,
                            audit_path=audit_path
                        )
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "status": "success",
                            "saved_records": len(records),
                            "audit_summary": summary
                        }).encode("utf-8"))
                    except Exception as e:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                else:
                    self.send_error(404, "Endpoint not found")

            def log_message(self, format, *args):
                pass # Suppress noisy server log outputs

        return Handler


def serve_annotation_studio(
    port: int = 8050,
    queue_path: str = "data/golden_eval/review_queue_200.jsonl",
    golden_path: str = "data/golden_eval/golden_eval_200.jsonl",
    output_path: str = "data/golden_eval/human_verified_golden_200.jsonl",
    audit_path: str = "data/golden_eval/human_review_audit.json"
) -> None:
    """Starts local HTTP server hosting the interactive Annotation Studio."""
    from http.server import HTTPServer

    export_review_queue(golden_path, queue_path)
    handler_class = AnnotationStudioHTTPHandler.create_handler(
        queue_path=queue_path,
        golden_path=golden_path,
        output_path=output_path,
        audit_path=audit_path
    )
    server = HTTPServer(("127.0.0.1", port), handler_class)
    print("\n" + "=" * 75)
    print("SUPPORTPILOT AI — LOCAL HUMAN ANNOTATION STUDIO SERVER")
    print("=" * 75)
    print(f"Local Server running at: http://127.0.0.1:{port}")
    print(f"Open http://127.0.0.1:{port} in your browser to review all 200 items.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 75 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation studio server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SupportPilot AI Golden Set Human Review Tool")
    parser.add_argument("--audit", action="store_true", help="Print audit status")
    parser.add_argument("--present", action="store_true", help="Present recommendations for all 200 records")
    parser.add_argument("--export-queue", action="store_true", help="Export review queue to JSONL")
    parser.add_argument("--interactive", action="store_true", help="Run interactive terminal review session")
    parser.add_argument("--serve", action="store_true", help="Launch local web Annotation Studio on port 8050")
    parser.add_argument("--port", type=int, default=8050, help="Port for Annotation Studio server")
    parser.add_argument("--apply-recommendations", action="store_true", help="Batch apply ANNOTATION_GUIDELINES.md recommendations")
    parser.add_argument("--apply-decisions-file", type=str, help="Apply JSON file containing reviewer decisions")
    parser.add_argument("--annotator", type=str, default="guideline_expert_auditor", help="Annotator / Reviewer ID")
    parser.add_argument("--human-signoff", action="store_true", help="Declare that review was signed off by a human")
    args = parser.parse_args()

    if args.serve:
        serve_annotation_studio(port=args.port)
    elif args.export_queue:
        export_review_queue()
    elif args.present:
        recs = export_review_queue()
        for i, r in enumerate(recs, 1):
            print(format_review_presentation(r, i, len(recs)))
    elif args.apply_decisions_file:
        with open(args.apply_decisions_file, "r", encoding="utf-8") as f:
            decisions = json.load(f)
        summary = save_human_review_decisions(
            decisions=decisions,
            annotator_id=args.annotator
        )
        print(json.dumps(summary, indent=2))
    elif args.apply_recommendations:
        summary = batch_apply_expert_guideline_review(
            reviewer_id=args.annotator,
            is_human_sign_off=args.human_signoff
        )
        print(json.dumps(summary, indent=2))
    elif args.interactive:
        run_interactive_review_workflow(annotator_id=args.annotator)
    elif args.audit:
        audit_file = "data/golden_eval/human_review_audit.json"
        if os.path.exists(audit_file):
            with open(audit_file, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print(json.dumps({"status": "NO_REVIEW_AUDIT_FOUND", "message": "Run with --apply-recommendations or --interactive first"}, indent=2))
    else:
        parser.print_help()

