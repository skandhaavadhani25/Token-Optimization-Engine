"""End-to-end walkthrough verification script matching deliverables:
(a) Fresh start with no workspace -> verify empty states & chat response
(b) Upload ONE small Python file with 3 functions calling each other and 1 external library -> describe resulting graph (nodes, edges, counts)
(c) Ask a question in chat about it and show answer cites that file and lines
(d) Upload a zip of a small project into a new workspace and show graph switch
(e) Switch back to first workspace and show graph and chat follow
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.api.server import app

client = TestClient(app)

print("=== WALKTHROUGH STEP (a): Fresh start with no workspace ===")
# 1. Check stats without workspace
stats_resp = client.get("/api/stats")
print(f"Stats without workspace status: {stats_resp.status_code}")
# 2. Check chat query without workspace or attachments
chat_resp = client.post("/api/conversations", json={"title": "Fresh Chat", "mode": "economy"})
conv_id = chat_resp.json()["id"]
msg_resp = client.post(f"/api/conversations/{conv_id}/messages", json={"content": "Explain the auth system"})
print(f"Chat response without workspace: '{msg_resp.json()['assistant_message']['content']}'")

print("\n=== WALKTHROUGH STEP (b): Upload ONE small Python file with 3 functions + 1 external library ===")
py_content = (
    "import math\n\n"
    "def calculate_hypotenuse(a: float, b: float) -> float:\n"
    "    return math.sqrt(a * a + b * b)\n\n"
    "def compute_perimeter(a: float, b: float) -> float:\n"
    "    hyp = calculate_hypotenuse(a, b)\n"
    "    return a + b + hyp\n\n"
    "def run_geometry_pipeline() -> dict:\n"
    "    peri = compute_perimeter(3.0, 4.0)\n"
    "    return {'perimeter': peri}\n"
)

files = [("files", ("geometry.py", io.BytesIO(py_content.encode("utf-8")), "text/x-python"))]
ws_resp = client.post(
    "/api/workspaces",
    files=files,
    data={"name": "Geometry Workspace", "relative_paths": ["geometry.py"]},
)
assert ws_resp.status_code == 200, ws_resp.text
ws_b = ws_resp.json()["workspace"]
ws_b_id = ws_b["id"]
print(f"Created Workspace 1: ID={ws_b_id}, Name={ws_b['name']}")

graph_b = client.get(f"/api/graph?workspace_id={ws_b_id}").json()
nodes_b = graph_b["nodes"]
edges_b = graph_b["edges"]
print(f"Code Graph Tab Data for Workspace 1:")
print(f"  Total Nodes: {len(nodes_b)}")
print(f"  Total Edges: {len(edges_b)}")
node_types = {}
for n in nodes_b:
    t = n.get("type", "unknown")
    node_types[t] = node_types.get(t, 0) + 1
print(f"  Node Type Breakdown: {node_types}")
print(f"  Nodes detail:")
for n in nodes_b:
    print(f"    - [{n.get('type')}] id='{n.get('id')}', label='{n.get('label')}'")

edge_types = {}
for e in edges_b:
    r = e.get("rel", "unknown")
    edge_types[r] = edge_types.get(r, 0) + 1
print(f"  Edge Rel Breakdown: {edge_types}")
for e in edges_b:
    print(f"    - {e.get('source')} --({e.get('rel')})--> {e.get('target')}")

print("\n=== WALKTHROUGH STEP (c): Ask a question in chat citing file and lines ===")
conv_b_resp = client.post("/api/conversations", json={"workspace_id": ws_b_id, "title": "Geometry Chat", "mode": "balanced"})
conv_b_id = conv_b_resp.json()["id"]

# Send question in chat
ask_resp = client.post(
    f"/api/conversations/{conv_b_id}/messages",
    json={"content": "How does run_geometry_pipeline calculate the perimeter and what functions does it call?", "dry_run": True},
)
assert ask_resp.status_code == 200
msg_c = ask_resp.json()["assistant_message"]
print(f"Assistant Message Content Preview:\n{msg_c['content']}")
print(f"Turn Tokens Metrics:")
print(f"  {json.dumps(ask_resp.json().get('turn_tokens', {}), indent=2)}")

# Also test retrieval context directly citing file and lines
ret_b = client.post("/api/retrieve", json={"workspace_id": ws_b_id, "query": "run_geometry_pipeline perimeter"}).json()
print(f"Retrieved Files Included: {ret_b['context']['files_included']}")
print(f"Context Text Sample (showing file and lines citation):\n{ret_b['context']['text'][:350]}...")

print("\n=== WALKTHROUGH STEP (d): Upload a zip of a small project into a new workspace ===")
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    zf.writestr("app/main.py", "def start_app():\n    return 'app started'\n")
    zf.writestr("app/auth.py", "def login(user):\n    return True\n")
    zf.writestr("README.md", "# Project Auth App\n")

zip_buf.seek(0)
ws_zip_resp = client.post(
    "/api/workspaces",
    files={"file": ("project_app.zip", zip_buf, "application/zip")},
    data={"name": "Project App Zip"},
)
assert ws_zip_resp.status_code == 200
ws_d = ws_zip_resp.json()["workspace"]
ws_d_id = ws_d["id"]
print(f"Created Workspace 2 (from zip): ID={ws_d_id}, Name={ws_d['name']}")

graph_d = client.get(f"/api/graph?workspace_id={ws_d_id}").json()
print(f"Code Graph Tab Data for Workspace 2:")
print(f"  Nodes: {len(graph_d['nodes'])}, Edges: {len(graph_d['edges'])}")
for n in graph_d["nodes"]:
    print(f"    - [{n.get('type')}] {n.get('label')}")

print("\n=== WALKTHROUGH STEP (e): Switch back to first workspace and show graph and chat follow ===")
# Verify Workspace 1 graph is intact and scoped
graph_switch_back = client.get(f"/api/graph?workspace_id={ws_b_id}").json()
nodes_sb = [n.get("label") for n in graph_switch_back["nodes"]]
print(f"Switched back to Workspace 1 (Geometry):")
print(f"  Nodes count: {len(graph_switch_back['nodes'])}")
print(f"  Labels in Workspace 1: {nodes_sb}")
assert any("run_geometry_pipeline" in lbl for lbl in nodes_sb)
assert not any("login" in lbl for lbl in nodes_sb), "Isolation breached!"

# Verify Workspace 1 conversation history
convs_w1 = client.get(f"/api/conversations?workspace_id={ws_b_id}").json()
print(f"Conversations filtered to Workspace 1: {[c['title'] for c in convs_w1]}")
print("Walkthrough finished successfully!")
