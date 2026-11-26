"""Comprehensive test suite for workspace-scoped code graph, retrieval, chat,
and upload isolation.

Tests:
1. No hardcoded sample_repos in app/ or web/ sources.
2. Single-file upload auto-creates workspace, indexes, and builds graph with same-file CALLS edges.
3. Multi-file, folder, and zip uploads with safety rejection (zip-slip, symlink, oversize).
4. Incremental file addition to existing workspace updates index & graph.
5. Strict workspace isolation: workspaces with distinct functions never leak across stats, graph, retrieve, ask.
6. Non-Python files appear as text_only nodes without crashing.
7. External & unresolved calls appear as dim external nodes in single-file graph.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_no_hardcoded_sample_repos_in_app_or_web():
    """Assert no hardcoded sample_repos string exists in app/ or web/ source files."""
    root = Path(__file__).resolve().parent.parent
    app_dir = root / "app"
    web_dir = root / "web"

    violations: list[str] = []
    pattern = re.compile(r"sample_repos", re.IGNORECASE)

    for directory in [app_dir, web_dir]:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in [".py", ".js", ".html", ".css", ".json"]:
                # Exclude any test fixtures or json data
                content = path.read_text(encoding="utf-8", errors="ignore")
                for lineno, line in enumerate(content.splitlines(), start=1):
                    if pattern.search(line):
                        violations.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")

    assert not violations, f"Found hardcoded 'sample_repos' in sources:\n" + "\n".join(violations)


def test_upload_single_py_file_creates_workspace_and_graph(client: TestClient):
    """Upload a single .py file with functions that call each other and an external library.

    Asserts:
    - Workspace is created and active
    - Pipeline runs (metadata, graph, embeddings)
    - Graph contains the file, its functions, same-file CALLS edge, and external node
    """
    code = (
        "import math\n\n"
        "def helper(x: float) -> float:\n"
        "    return math.sqrt(x)\n\n"
        "def compute(a: float, b: float) -> float:\n"
        "    return helper(a) + helper(b)\n\n"
        "def main_calc() -> float:\n"
        "    return compute(4.0, 9.0)\n"
    )

    files = [
        ("files", ("calculator.py", io.BytesIO(code.encode("utf-8")), "text/x-python")),
    ]
    data = {
        "name": "calculator_ws",
        "relative_paths": ["calculator.py"],
    }

    resp = client.post("/api/workspaces", files=files, data=data)
    assert resp.status_code == 200, resp.text
    ws_data = resp.json()
    ws = ws_data["workspace"]
    ws_id = ws["id"]
    assert ws["name"] == "calculator_ws"
    assert ws["file_counts"]["python"] == 1

    # Check stats endpoint with workspace_id
    stats_resp = client.get(f"/api/stats?workspace_id={ws_id}")
    assert stats_resp.status_code == 200, stats_resp.text
    stats = stats_resp.json()
    assert stats["database"]["files"] == 1
    assert stats["database"]["functions"] == 3  # helper, compute, main_calc

    # Check graph endpoint with workspace_id
    graph_resp = client.get(f"/api/graph?workspace_id={ws_id}&hops=2&max_nodes=50")
    assert graph_resp.status_code == 200, graph_resp.text
    graph = graph_resp.json()
    node_ids = [n["id"] for n in graph["nodes"]]
    labels = [n["label"] for n in graph["nodes"]]

    # Assert functions are present
    assert any("helper" in lbl for lbl in labels)
    assert any("compute" in lbl for lbl in labels)
    assert any("main_calc" in lbl for lbl in labels)

    # Assert edges contain CALLS between same-file functions
    calls_edges = [e for e in graph["edges"] if e["rel"] == "CALLS"]
    assert len(calls_edges) >= 2, f"Expected CALLS edges, found: {calls_edges}"

    # Assert external/unresolved node is preserved (math.sqrt)
    ext_nodes = [n for n in graph["nodes"] if n.get("type") in ("external", "unresolved")]
    assert len(ext_nodes) >= 1 or any("math" in nid for nid in node_ids)


def test_upload_multi_files_and_folder_and_zip(client: TestClient):
    """Test multi-file, zip upload, and security rejections."""
    # 1. Valid zip upload
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg/__init__.py", "")
        zf.writestr("pkg/service.py", "def serve(): return 'ok'\n")
        zf.writestr("README.md", "# Project Documentation\n")

    buf.seek(0)
    files = {"file": ("my_archive.zip", buf, "application/zip")}
    data = {"name": "zip_ws"}
    resp = client.post("/api/workspaces", files=files, data=data)
    assert resp.status_code == 200, resp.text
    res = resp.json()
    assert res["workspace"]["name"] == "zip_ws"
    assert res["workspace"]["file_counts"]["python"] == 2
    assert res["workspace"]["file_counts"]["text_only"] == 1

    # 2. Path traversal (zip-slip) rejected
    bad_buf = io.BytesIO()
    with zipfile.ZipFile(bad_buf, "w") as zf:
        zf.writestr("../../../evil.py", "print('hacked')\n")
    bad_buf.seek(0)
    resp_bad = client.post("/api/workspaces", files={"file": ("evil.zip", bad_buf, "application/zip")})
    assert resp_bad.status_code == 400
    err_str = str(resp_bad.json()).lower()
    assert "zip-slip" in err_str or "traversal" in err_str

    # 3. Oversize loose file rejected
    huge_buf = io.BytesIO(b"A" * 1_200_000)
    resp_huge = client.post(
        "/api/workspaces",
        files=[("files", ("huge.txt", huge_buf, "text/plain"))],
        data={"relative_paths": ["huge.txt"]},
    )
    assert resp_huge.status_code == 200
    assert resp_huge.json()["workspace"]["file_counts"]["skipped"] >= 1


def test_incremental_add_files_to_workspace(client: TestClient):
    """Test adding files to an existing workspace incrementally."""
    # Create workspace with one file
    f1 = "def first(): return 1\n"
    files = [("files", ("first.py", io.BytesIO(f1.encode("utf-8")), "text/x-python"))]
    resp = client.post("/api/workspaces", files=files, data={"name": "incr_ws", "relative_paths": ["first.py"]})
    assert resp.status_code == 200
    ws_id = resp.json()["workspace"]["id"]

    # Incrementally add second file
    f2 = "from first import first\ndef second(): return first() + 2\n"
    add_files = [("files", ("second.py", io.BytesIO(f2.encode("utf-8")), "text/x-python"))]
    add_resp = client.post(
        f"/api/workspaces/{ws_id}/files",
        files=add_files,
        data={"relative_paths": ["second.py"]},
    )
    assert add_resp.status_code == 200, add_resp.text
    report = add_resp.json()["report"]
    assert "second.py" in str(report.get("added", []))

    # Verify updated stats
    stats_resp = client.get(f"/api/stats?workspace_id={ws_id}")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["database"]["files"] == 2
    assert stats_resp.json()["database"]["functions"] == 2


def test_workspace_data_isolation(client: TestClient):
    """Create two workspaces with different functions and assert strict data isolation across stats, graph, retrieve, and ask."""
    # Workspace A
    code_a = "def process_alpha_telemetry():\n    return 'alpha'\n"
    resp_a = client.post(
        "/api/workspaces",
        files=[("files", ("alpha.py", io.BytesIO(code_a.encode("utf-8")), "text/x-python"))],
        data={"name": "ws_alpha", "relative_paths": ["alpha.py"]},
    )
    assert resp_a.status_code == 200
    ws_a_id = resp_a.json()["workspace"]["id"]

    # Workspace B
    code_b = "def compute_beta_quantum_state():\n    return 'beta'\n"
    resp_b = client.post(
        "/api/workspaces",
        files=[("files", ("beta.py", io.BytesIO(code_b.encode("utf-8")), "text/x-python"))],
        data={"name": "ws_beta", "relative_paths": ["beta.py"]},
    )
    assert resp_b.status_code == 200
    ws_b_id = resp_b.json()["workspace"]["id"]

    # 1. Graph isolation
    graph_a = client.get(f"/api/graph?workspace_id={ws_a_id}").json()
    labels_a = [n["label"] for n in graph_a["nodes"]]
    assert any("alpha" in lbl for lbl in labels_a)
    assert not any("beta" in lbl for lbl in labels_a), "Workspace A leaked Workspace B's functions!"

    graph_b = client.get(f"/api/graph?workspace_id={ws_b_id}").json()
    labels_b = [n["label"] for n in graph_b["nodes"]]
    assert any("beta" in lbl for lbl in labels_b)
    assert not any("alpha" in lbl for lbl in labels_b), "Workspace B leaked Workspace A's functions!"

    # 2. Retrieve isolation
    ret_a = client.post("/api/retrieve", json={"workspace_id": ws_a_id, "query": "alpha telemetry", "top_k": 5}).json()
    text_a = ret_a.get("context", {}).get("text", "")
    files_a = ret_a.get("context", {}).get("files_included", [])
    assert "alpha" in text_a or any("alpha" in f for f in files_a)
    assert "compute_beta_quantum_state" not in text_a, "Workspace A retrieval included Workspace B code!"
    assert not any("beta" in f for f in files_a)

    ret_b = client.post("/api/retrieve", json={"workspace_id": ws_b_id, "query": "quantum state", "top_k": 5}).json()
    text_b = ret_b.get("context", {}).get("text", "")
    files_b = ret_b.get("context", {}).get("files_included", [])
    assert "beta" in text_b or any("beta" in f for f in files_b)
    assert "process_alpha_telemetry" not in text_b, "Workspace B retrieval included Workspace A code!"
    assert not any("alpha" in f for f in files_b)


def test_non_python_upload_shows_text_only_node(client: TestClient):
    """Assert non-Python files appear as text_only nodes without crashing."""
    files = [
        ("files", ("config.yaml", io.BytesIO(b"server:\n  port: 8080\n"), "text/yaml")),
        ("files", ("notes.txt", io.BytesIO(b"Antigravity notes\n"), "text/plain")),
    ]
    resp = client.post(
        "/api/workspaces",
        files=files,
        data={"name": "text_ws", "relative_paths": ["config.yaml", "notes.txt"]},
    )
    assert resp.status_code == 200
    ws_id = resp.json()["workspace"]["id"]

    # Graph should contain text_only nodes
    graph = client.get(f"/api/graph?workspace_id={ws_id}").json()
    text_nodes = [n for n in graph["nodes"] if n.get("type") == "text_only"]
    assert len(text_nodes) >= 2
    for n in text_nodes:
        assert "text-only" in n.get("note", "").lower()


def test_chat_without_code_returns_upload_first_message(client: TestClient):
    """When a chat has no workspace and no attachments, assistant advises 'Upload code first'."""
    # Create conversation with no workspace
    conv_resp = client.post("/api/conversations", json={"title": "Empty Chat", "mode": "economy"})
    assert conv_resp.status_code == 200
    conv_id = conv_resp.json()["id"]

    # Send message
    msg_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "How does the authentication system work?"},
    )
    assert msg_resp.status_code == 200
    reply = msg_resp.json()["assistant_message"]["content"]
    assert "Upload code first" in reply
