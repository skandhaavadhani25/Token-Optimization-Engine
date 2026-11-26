import json
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"
REPO_PATH = "sample_repos/requests"

client = httpx.Client(base_url=BASE_URL, timeout=120.0)

def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def log_request(method: str, path: str, payload=None, params=None):
    print(f"\n--> {method} {path}")
    if params:
        print(f"    Params: {params}")
    if payload:
        print(f"    Body: {json.dumps(payload)}")
    t0 = time.perf_counter()
    if method == "GET":
        r = client.get(path, params=params)
    elif method == "POST":
        r = client.post(path, json=payload, params=params)
    latency = time.perf_counter() - t0
    print(f"<-- Status: {r.status_code} ({latency:.3f}s)")
    try:
        data = r.json()
        formatted = json.dumps(data, indent=2)
        lines = formatted.splitlines()
        if len(lines) > 30:
            print("\n".join(lines[:30]) + f"\n... [{len(lines) - 30} lines truncated]")
        else:
            print(formatted)
        return data, latency
    except Exception:
        print(r.text[:500])
        return None, latency

print_section("1. GET /health")
log_request("GET", "/health")

print_section("2. POST /index (re-indexing sample_repos/requests)")
log_request("POST", "/index", payload={"repo_path": REPO_PATH, "force": False})

print_section("3. GET /stats")
log_request("GET", "/stats", params={"repo_path": REPO_PATH})

print_section("4. POST /search")
log_request("POST", "/search", payload={"repo_path": REPO_PATH, "query": "cookie jar management", "top_k": 3})

print_section("5. POST /retrieve")
log_request("POST", "/retrieve", payload={"repo_path": REPO_PATH, "query": "Session.send", "max_tokens": 1500})

print_section("6. GET /graph (Overview - no center)")
log_request("GET", "/graph", params={"repo_path": REPO_PATH, "max_nodes": 20})

print_section("7. GET /graph (Centered on Session.send)")
log_request("GET", "/graph", params={"repo_path": REPO_PATH, "center": "Session.send", "hops": 1, "max_nodes": 30})

print_section("8. GET /graph/node (Node details for src.requests.sessions.Session.send)")
log_request("GET", "/graph/node", params={"repo_path": REPO_PATH, "id": "src.requests.sessions.Session.send"})

print_section("9. POST /ask with dry_run=True")
log_request("POST", "/ask", payload={"repo_path": REPO_PATH, "query": "Optimize Session.send", "dry_run": True})
