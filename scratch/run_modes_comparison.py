"""
Comparison script for Deliverable 4 & 5:
- 5-turn conversation token savings panel numbers (actual vs whole-files, conversation, full-repo baselines)
- Economy vs Balanced vs Quality modes tested on 3 questions
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

def req(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    h = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None
    request = urllib.request.Request(url, data=body, headers=h, method=method)
    with urllib.request.urlopen(request) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_modes_benchmark():
    # Use existing sample_repos/requests directly or a workspace
    ws_res = req("/api/workspaces", method="POST", data={
        "local_path": "sample_repos/requests",
        "name": "modes-bench-ws",
        "edit_in_place": False,
    })
    ws_id = ws_res["workspace"]["id"]

    questions = [
        "How does Session.send handle adapter retries?",
        "What does HTTPAdapter.build_response do?",
        "Where is prepared request authentication handled?"
    ]
    modes = ["economy", "balanced", "quality"]

    print("=== DELIVERABLE 5: MODES COMPARISON (Economy vs Balanced vs Quality) ===")
    results = {}
    for mode in modes:
        conv_res = req("/api/conversations", method="POST", data={
            "workspace_id": ws_id,
            "title": f"Bench {mode}",
            "mode": mode,
        })
        conv_id = conv_res["id"]
        results[mode] = []

        print(f"\n--- Testing Mode: {mode.upper()} ---")
        for q in questions:
            msg_res = req(f"/api/conversations/{conv_id}/messages", method="POST", data={
                "content": q,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "mode": mode,
                "dry_run": True,
            })
            tokens = msg_res["turn_tokens"]
            needs_context = "NEEDS_CONTEXT:" in msg_res["assistant_message"]["content"]
            results[mode].append({
                "question": q,
                "actual_context_tokens": tokens["actual_context_tokens"],
                "actual_total_sent": tokens["actual_total_sent"],
                "budget": tokens["budget"],
                "needs_context": needs_context,
                "whole_files_baseline": tokens["whole_files_baseline"],
                "whole_saved_pct": tokens["whole_files_saved_pct"],
            })
            print(f"Q: '{q}'")
            print(f"   Context Tokens: {tokens['actual_context_tokens']} / {tokens['budget']} budget")
            print(f"   Total Sent: {tokens['actual_total_sent']} (Whole-Files: {tokens['whole_files_baseline']}, saved {tokens['whole_files_saved_pct']}%)")
            print(f"   NEEDS_CONTEXT: {needs_context}")

    # Deliverable 4: 5-Turn Conversation
    print("\n=== DELIVERABLE 4: 5-TURN CONVERSATION TOKEN ACCUMULATION ===")
    conv5_res = req("/api/conversations", method="POST", data={
        "workspace_id": ws_id,
        "title": "5-Turn Requests Thread",
        "mode": "balanced",
    })
    conv5_id = conv5_res["id"]
    five_turns = [
        "How does Session.send work?",
        "optimize it",
        "What exception does it raise on timeout?",
        "Show how adapters hook into it",
        "Summarize all changes we discussed",
    ]
    turn_records = []
    for idx, q in enumerate(five_turns, 1):
        res = req(f"/api/conversations/{conv5_id}/messages", method="POST", data={
            "content": q,
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "mode": "balanced",
            "budget": 4000,
            "dry_run": True,
        })
        tok = res["turn_tokens"]
        turn_records.append({
            "turn": idx,
            "user_prompt": q,
            "actual_sent": tok["actual_total_sent"],
            "actual_context": tok["actual_context_tokens"],
            "whole_files_baseline": tok["whole_files_baseline"],
            "whole_saved_pct": tok["whole_files_saved_pct"],
            "conversation_baseline": tok["conversation_baseline"],
            "full_repo_baseline": tok["full_repo_baseline"],
        })
        print(f"Turn {idx} [{q}]:")
        print(f"   Sent: {tok['actual_total_sent']} | Context: {tok['actual_context_tokens']} | Conv Baseline: {tok['conversation_baseline']} | Whole-Files Baseline: {tok['whole_files_baseline']} ({tok['whole_files_saved_pct']}% saved)")

    # Clean up workspace
    req(f"/api/workspaces/{ws_id}", method="DELETE")
    print(f"\nCleaned up workspace {ws_id}.")

if __name__ == "__main__":
    run_modes_benchmark()
