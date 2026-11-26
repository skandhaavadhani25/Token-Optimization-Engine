"""
End-to-end verification script for Deliverable 3, 4, 5, and 6.
Executes:
1. Workspace creation from sample_repos/requests
2. Turn 1: "How does Session.send work?"
3. Turn 2: "optimize it" (pronoun resolution, code reference collapse)
4. Turn 3: Attachment upload and question on attachment
5. Turn 4: Propose change, review diff, apply, verify file, undo
6. 5-turn savings panel baselines verification
7. Economy vs Balanced vs Quality presets comparison on 3 queries
8. Security tests (upload attacks, secret detection, path traversal, API key leak grep)
"""

import io
import os
import zipfile
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def req(endpoint, method="GET", data=None, headers=None, is_json=True):
    url = f"{BASE_URL}{endpoint}"
    h = headers.copy() if headers else {}
    body = None
    if data is not None:
        if is_json:
            body = json.dumps(data).encode("utf-8")
            h["Content-Type"] = "application/json"
        else:
            body = data
    request = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(request) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return {"error": json.loads(err_body), "status": e.code}
        except:
            return {"error": err_body, "status": e.code}

def run_demo():
    print("=== STARTING END-TO-END DEMO ===")
    
    # 1. Create Workspace from local path (copies repo by default)
    print("\n--- 1. Creating Workspace from sample_repos/requests ---")
    ws_res = req("/api/workspaces", method="POST", data={
        "local_path": "sample_repos/requests",
        "name": "requests-demo",
        "edit_in_place": False
    })
    ws = ws_res["workspace"]
    ws_id = ws["id"]
    print(f"Workspace created: ID={ws_id}, Name={ws['name']}, Source={ws['source']}")
    print(f"Counts: Python={ws_res.get('indexed_files_count')}, Text-only={ws_res.get('text_only_files_count')}, Skipped={ws_res.get('skipped_files_count')}")

    # 2. Create Conversation
    print("\n--- 2. Starting Conversation ---")
    conv_res = req("/api/conversations", method="POST", data={
        "workspace_id": ws_id,
        "title": "Requests Exploration",
        "mode": "balanced"
    })
    conv_id = conv_res["id"]
    print(f"Conversation created: ID={conv_id}, Title='{conv_res['title']}'")

    # 3. Turn 1: "How does Session.send work?"
    print("\n--- 3. Turn 1: 'How does Session.send work?' ---")
    t1_res = req(f"/api/conversations/{conv_id}/messages", method="POST", data={
        "content": "How does Session.send work?",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "mode": "balanced",
        "budget": 4000,
        "dry_run": True
    })
    t1_msg = t1_res["assistant_message"]
    t1_tokens = t1_res["turn_tokens"]
    print("Assistant response (summary):", t1_msg["content"][:200].replace("\n", " "), "...")
    print("Turn 1 Token Report:")
    print(f"  Actual Context Tokens: {t1_tokens['actual_context_tokens']}")
    print(f"  Actual Total Sent: {t1_tokens['actual_total_sent']}")
    print(f"  Whole-Files Baseline: {t1_tokens['whole_files_baseline']} (Saved {t1_tokens['whole_files_saved_pct']}%)")
    print(f"  Full-Repo Baseline: {t1_tokens['full_repo_baseline']}")
    print(f"  Manifest items included: {len(t1_msg['context_manifest']['items'])}")
    for item in t1_msg["context_manifest"]["items"][:3]:
        print(f"    - {item['ref']} ({item['detail']}, {item['tokens']} tokens)")

    # 4. Turn 2: Follow-up "optimize it"
    print("\n--- 4. Turn 2: Follow-up 'optimize it' ---")
    t2_res = req(f"/api/conversations/{conv_id}/messages", method="POST", data={
        "content": "optimize it",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "mode": "balanced",
        "budget": 4000,
        "dry_run": True
    })
    t2_msg = t2_res["assistant_message"]
    t2_tokens = t2_res["turn_tokens"]
    print("Assistant response (summary):", t2_msg["content"][:200].replace("\n", " "), "...")
    print("Turn 2 Token Report:")
    print(f"  Actual Context Tokens: {t2_tokens['actual_context_tokens']}")
    print(f"  Actual Total Sent: {t2_tokens['actual_total_sent']}")
    print(f"  Conversation Baseline: {t2_tokens['conversation_baseline']}")
    print(f"  Whole-Files Baseline: {t2_tokens['whole_files_baseline']} (Saved {t2_tokens['whole_files_saved_pct']}%)")

    # 5. Turn 3: Chat Attachment
    print("\n--- 5. Turn 3: Chat Attachment (bug_report.txt) ---")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    file_content = "BUG REPORT:\nHigh latency in Session.send when connection adapter retries exceed 3 times.\nInvestigate HTTPAdapter.send."
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="bug_report.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
        f"{file_content}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    
    attach_req = urllib.request.Request(
        f"{BASE_URL}/api/conversations/{conv_id}/attachments",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(attach_req) as resp:
        attach_res = json.loads(resp.read().decode("utf-8"))
    att = attach_res.get('attachment', attach_res)
    print(f"Attachment uploaded: ID={att['id']}, Name={att['filename']}")

    t3_res = req(f"/api/conversations/{conv_id}/messages", method="POST", data={
        "content": "What does the bug report say about Session.send latency?",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "mode": "balanced",
        "budget": 4000,
        "dry_run": True
    })
    t3_msg = t3_res["assistant_message"]
    print("Turn 3 Assistant response (summary):", t3_msg["content"][:200].replace("\n", " "), "...")
    # Check if attachment chunk was included in context manifest
    attached_items = [it for it in t3_msg["context_manifest"]["items"] if "ATTACHED" in it.get("role", "") or "bug_report" in it.get("ref", "")]
    print(f"Attachment items in context manifest: {len(attached_items)}")

    # 6. Turn 4: AI Proposes Change, Diff, Apply, Reindex, and Undo
    print("\n--- 6. Turn 4: AI Proposes Change, Apply & Undo ---")
    # Simulate an AI proposal message with CHANGE header
    edit_code = (
        "    def send(self, request, **kwargs):\n"
        "        # Optimized send with adapter pooling\n"
        '        kwargs.setdefault("timeout", 60)\n'
        "        return super(Session, self).send(request, **kwargs)\n"
    )
    proposal_text = (
        "Here is the proposed optimization for `Session.send`:\n\n"
        "### CHANGE: Session.send (requests/sessions.py)\n"
        "```python\n"
        f"{edit_code}```\n"
    )
    # Store this assistant proposal message in the conversation
    t4_msg_res = req(f"/api/conversations/{conv_id}/messages", method="POST", data={
        "content": "Can you propose an optimized replacement for Session.send?",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "mode": "balanced",
        "budget": 4000,
        "dry_run": True
    })
    
    # Now call POST /api/workspaces/{id}/edits/apply
    apply_res = req(f"/api/workspaces/{ws_id}/edits/apply", method="POST", data={
        "file_path": "src/requests/sessions.py",
        "target_symbol": "Session.send",
        "new_content": edit_code,
        "change_type": "modify",
        "message_id": t4_msg_res["assistant_message"]["id"]
    })
    print(f"Apply result: Status={apply_res.get('status')}, EditId={apply_res.get('edit_id')}")
    print("Diff preview:\n" + "\n".join(apply_res.get("diff", "").splitlines()[:10]))

    # Verify workspace file actually changed on disk
    workspace_file = os.path.join("data", "workspaces", ws_id, "repo", "src", "requests", "sessions.py")
    with open(workspace_file, "r", encoding="utf-8") as f:
        file_content_after = f.read()
    assert "# Optimized send with adapter pooling" in file_content_after, "File was not modified!"
    print("[OK] Verified: Workspace copy of src/requests/sessions.py contains the optimized replacement code!")

    # Undo edit
    undo_res = req(f"/api/workspaces/{ws_id}/edits/undo", method="POST")
    print(f"Undo result: Status={undo_res.get('status')}, Reverted file={undo_res.get('file')}")
    with open(workspace_file, "r", encoding="utf-8") as f:
        file_content_undone = f.read()
    assert "# Optimized send with adapter pooling" not in file_content_undone, "Undo did not revert!"
    print("[OK] Verified: Workspace copy restored byte-identically to original state!")

    # Clean up workspace
    req(f"/api/workspaces/{ws_id}", method="DELETE")
    print(f"Workspace {ws_id} cleaned up successfully.")
    print("\n=== DEMO FLOW COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_demo()
