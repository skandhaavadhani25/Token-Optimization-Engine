"""Static file mounting and UI routing for the single-page dashboard."""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def setup_static_and_ui(app: FastAPI, web_dir: Path | str = "web") -> None:
    """Mount static web UI directory and configure backwards-compatible /api redirects."""
    web_path = Path(web_dir).resolve()
    if not web_path.exists():
        web_path.mkdir(parents=True, exist_ok=True)

    # 1. Backwards-compatible redirects from old un-prefixed endpoints to /api/*
    unprefixed_endpoints = [
        ("GET", "/health"),
        ("GET", "/config"),
        ("GET", "/providers"),
        ("POST", "/providers/{name}/validate"),
        ("GET", "/providers/{name}/models"),
        ("POST", "/index"),
        ("POST", "/index/update"),
        ("GET", "/index/status/{job_id}"),
        ("POST", "/search"),
        ("POST", "/retrieve"),
        ("POST", "/ask"),
        ("GET", "/graph"),
        ("GET", "/graph/node"),
        ("GET", "/stats"),
        ("GET", "/benchmark/querysets"),
        ("POST", "/benchmark-one"),
        ("POST", "/benchmark"),
        ("POST", "/workspaces"),
        ("GET", "/workspaces"),
        ("GET", "/workspaces/{workspace_id}"),
        ("DELETE", "/workspaces/{workspace_id}"),
        ("POST", "/workspaces/{workspace_id}/reindex"),
        ("POST", "/conversations"),
        ("GET", "/conversations"),
        ("GET", "/conversations/{conversation_id}"),
        ("DELETE", "/conversations/{conversation_id}"),
        ("POST", "/conversations/{conversation_id}/messages"),
        ("POST", "/conversations/{conversation_id}/attachments"),
        ("GET", "/conversations/{conversation_id}/attachments"),
    ]

    for method, path in unprefixed_endpoints:
        def make_redirect_handler(p: str):
            async def redirect_handler(request: Request) -> RedirectResponse:
                query = request.url.query
                # Resolve path parameters if any
                req_path = request.url.path
                target = f"/api{req_path}"
                if query:
                    target = f"{target}?{query}"
                return RedirectResponse(url=target, status_code=307)
            return redirect_handler

        app.add_api_route(
            path,
            make_redirect_handler(path),
            methods=[method],
            include_in_schema=False,
        )

    # 2. Explicit route for index.html at root GET /
    index_html = web_path / "index.html"

    @app.get("/", include_in_schema=False)
    async def serve_root():
        if index_html.exists():
            return FileResponse(index_html, media_type="text/html")
        return {"message": "Web UI directory not initialized."}

    # 3. Mount remaining static assets (styles.css, app.js, components, etc.)
    app.mount("/", StaticFiles(directory=str(web_path), html=True), name="static")
