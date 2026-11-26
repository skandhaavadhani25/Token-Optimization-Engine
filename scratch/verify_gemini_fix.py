"""End-to-end verification script for Gemini developer instruction fix and response handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.api.server import app
from app.ai.base_provider import ChatMessage, ProviderConfig
from app.ai.gemini_provider import GeminiProvider, _model_supports_system_instruction
from app.services.pipeline import RepositoryService
from google.genai import errors

client = TestClient(app)

print("=== 1. VERIFY MODEL SUPPORT DETECTION ===")
assert _model_supports_system_instruction("gemini-1.0-pro") is False
assert _model_supports_system_instruction("models/gemini-1.0-pro-001") is False
assert _model_supports_system_instruction("gemini-pro") is False
assert _model_supports_system_instruction("gemma-2-9b-it") is False
assert _model_supports_system_instruction("gemini-2.5-flash") is True
assert _model_supports_system_instruction("gemini-1.5-flash") is True
assert _model_supports_system_instruction("gemini-2.0-flash") is True
print("[OK] Model support detection working accurately for all variants")

print("\n=== 2. VERIFY RETRIEVAL, BM25, GRAPH SEARCH & CONTEXT BUDGET PRESERVED ===")
ret_res = client.post(
    "/api/retrieve",
    json={
        "repo_path": "sample_repos/requests",
        "query": "How does Session.send handle cookies and redirects?",
        "budget": 4000,
    },
)
assert ret_res.status_code == 200, ret_res.text
ret_data = ret_res.json()
assert "retrieval" in ret_data
assert "context" in ret_data
assert len(ret_data["retrieval"]["candidates"]) > 0
assert ret_data["context"]["tokens"] <= 4000
print(f"[OK] Hybrid retrieval succeeded: {len(ret_data['retrieval']['candidates'])} candidates, {ret_data['context']['tokens']} context tokens")

print("\n=== 3. VERIFY ASSEMBLED PROMPT PREVIEW (DRY RUN) ===")
dry_res = client.post(
    "/api/ask",
    json={
        "repo_path": "sample_repos/requests",
        "query": "How does Session.send handle cookies and redirects?",
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "dry_run": True,
        "max_tokens": 4000,
    },
)
assert dry_res.status_code == 200, dry_res.text
dry_data = dry_res.json()
assert dry_data["dry_run"] is True
assert len(dry_data["system_prompt"]) > 0
assert len(dry_data["messages"]) > 0
assert "Session.send" in dry_data["messages"][0]["content"]
print("[OK] Assembled prompt preview intact and includes packed context and user query")

print("\n=== 4. VERIFY GEMINI PROVIDER FALLBACK ON 'Developer instruction is not enabled' ===")
mock_genai_client = MagicMock()
dev_err = errors.APIError(400, {
    "error": {
        "code": 400,
        "message": "Developer instruction is not enabled for this model",
        "status": "INVALID_ARGUMENT",
    }
})
mock_ok_resp = MagicMock()
mock_ok_resp.text = "### ANALYSIS\nSession.send resolves cookies and redirects via resolve_redirects.\n\n### RISKS\n- Infinite redirect loops"
mock_ok_resp.candidates = [MagicMock(finish_reason="STOP")]
mock_ok_resp.usage_metadata.prompt_token_count = 210
mock_ok_resp.usage_metadata.candidates_token_count = 85
mock_ok_resp.usage_metadata.total_token_count = 295

mock_genai_client.models.generate_content.side_effect = [dev_err, mock_ok_resp]

cfg = ProviderConfig(provider="gemini", model="gemini-2.5-flash")
prov = GeminiProvider(config=cfg, client=mock_genai_client)

res = prov.complete("System prompt instructions", [ChatMessage(role="user", content="Explain Session.send")])
assert "Session.send resolves cookies" in res.text
assert mock_genai_client.models.generate_content.call_count == 2
# Second call had system_instruction=None and instructions incorporated into user message
second_call = mock_genai_client.models.generate_content.call_args_list[1].kwargs
assert second_call["config"].system_instruction is None
assert "System prompt instructions" in second_call["contents"][0].parts[0].text
assert "Explain Session.send" in second_call["contents"][0].parts[0].text
print("[OK] GeminiProvider successfully intercepted developer instruction error, safely folded instructions, and returned completion")

print("\n=== 5. VERIFY END-TO-END /ask API WITH GEMINI PROVIDER (MOCKED CLIENT) ===")
with patch("app.ai.gemini_provider.genai.Client", return_value=mock_genai_client):
    mock_genai_client.models.generate_content.side_effect = [dev_err, mock_ok_resp]
    ask_resp = client.post(
        "/api/ask",
        headers={"X-Provider-Key": "AIzaSyD-mock-test-key-for-gemini-12345"},
        json={
            "repo_path": "sample_repos/requests",
            "query": "How does Session.send handle cookies and redirects?",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "dry_run": False,
        },
    )
    assert ask_resp.status_code == 200, ask_resp.text
    ask_data = ask_resp.json()
    assert ask_data["completion"] is not None
    assert "Session.send resolves cookies" in ask_data["completion"]["text"]
    assert ask_data["answer"]["analysis"] is not None
    assert ask_data["key_source"] == "user"
    print(f"[OK] /ask returned 200 with completion text: '{ask_data['completion']['text'][:60]}...'")

print("\n=== 6. VERIFY LIVE GEMINI API IF USER KEY IS IN ENVIRONMENT ===")
import os
live_key = os.environ.get("GEMINI_API_KEY")
if live_key:
    print("Found live GEMINI_API_KEY in environment, testing live call...")
    live_prov = GeminiProvider(config=ProviderConfig(provider="gemini", model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")), api_key=live_key)
    live_res = live_prov.complete("Be concise.", [ChatMessage(role="user", content="Reply 'PONG'.")], max_output_tokens=10)
    print(f"[OK] Live Gemini call succeeded: '{live_res.text.strip()}' (Tokens: {live_res.usage.total_tokens})")
else:
    print("No GEMINI_API_KEY environment variable found (BYOK key handled via header in web UI).")

print("\n=== ALL E2E VERIFICATIONS PASSED SUCCESSFULLY ===")
