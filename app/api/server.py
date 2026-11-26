"""FastAPI application factory, middleware, and exception handlers."""

from __future__ import annotations

import logging
import re
import sys
import time
import traceback
import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ai.base_provider import (
    ApiKeyInvalidError,
    ApiKeyRequiredError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.api.routes import QueryTooLongError, router
from app.api.schemas import ErrorDetail, ErrorResponse
from app.config import get_settings
from app.context.token_budget import BudgetTooSmallError
from app.security import (
    current_request_key,
    redact_sensitive_text,
    setup_key_redaction_logging,
)
from app.storage.mongo import StorageUnavailableError
from app.services.pipeline import IndexingInProgressError, PrerequisiteMissingError

logger = logging.getLogger("code_context_engine.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)

setup_key_redaction_logging()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="Code Context Engine API",
        description="High-precision retrieval and token-budgeted AI coding assistant service",
        version="0.8.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Middleware
    origins = settings.get_cors_origins_list()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Logging & Correlation Middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        request.state.request_id = request_id

        # Log request path without body or secrets
        logger.info(f"[{request_id}] --> {request.method} {request.url.path}")

        raw_key = request.headers.get("x-provider-key")
        cleaned_key = raw_key.strip() if raw_key else None
        ctx_token = current_request_key.set(cleaned_key)

        try:
            response = await call_next(request)
            if cleaned_key or "/ask" in request.url.path or "/validate" in request.url.path or "/models" in request.url.path:
                response.headers["Cache-Control"] = "no-store"
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
            logger.info(f"[{request_id}] <-- {response.status_code} ({duration_ms:.1f}ms)")
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            route = f"{request.method} {request.url.path}"
            tb_str = traceback.format_exc()
            clean_tb = redact_sensitive_text(tb_str)
            logger.error(
                f"[{request_id}] Unhandled server error on {route}: {exc.__class__.__name__}: {exc}\n{clean_tb}"
            )
            return _internal_error(exc, request)
        finally:
            current_request_key.reset(ctx_token)

    # 3. Exception Handlers
    def _format_error(
        status_code: int,
        code: str,
        message: str,
        request: Request | None = None,
        details: Any = None,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) if request and hasattr(request, "state") else None
        clean_message = redact_sensitive_text(message)
        clean_details = (
            redact_sensitive_text(details) if isinstance(details, str) else details
        )
        error_payload = ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=clean_message,
                details=clean_details,
                request_id=request_id,
            )
        )
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if request and (request.headers.get("x-provider-key") or "/ask" in getattr(request.url, "path", "") or "/validate" in getattr(request.url, "path", "")):
            headers["Cache-Control"] = "no-store"
        return JSONResponse(status_code=status_code, content=error_payload.model_dump(), headers=headers)

    def _internal_error(exc: Exception, request: Request | None) -> JSONResponse:
        """Generic 500: no message text from the exception, only its type for support."""
        return _format_error(
            status_code=500,
            code="internal_error",
            message="An internal server error occurred.",
            request=request,
            details={"error_type": exc.__class__.__name__},
        )

    @app.exception_handler(StorageUnavailableError)
    async def storage_unavailable_handler(request: Request, exc: StorageUnavailableError):
        return _format_error(
            status_code=503,
            code="storage_unavailable",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code = "rate_limit_exceeded" if exc.status_code == 429 else "http_error"
        return _format_error(
            status_code=exc.status_code,
            code=code,
            message=str(exc.detail),
            request=request,
        )

    @app.exception_handler(QueryTooLongError)
    async def query_too_long_handler(request: Request, exc: QueryTooLongError):
        return _format_error(
            status_code=400,
            code="QUERY_TOO_LONG",
            message=str(exc),
            request=request,
            details={"query_length": exc.query_length, "max_length": exc.max_length},
        )

    @app.exception_handler(BudgetTooSmallError)
    async def budget_too_small_handler(request: Request, exc: BudgetTooSmallError):
        return _format_error(
            status_code=400,
            code="BUDGET_TOO_SMALL",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(IndexingInProgressError)
    async def indexing_in_progress_handler(request: Request, exc: IndexingInProgressError):
        return _format_error(
            status_code=409,
            code="indexing_in_progress",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(PrerequisiteMissingError)
    async def prerequisite_missing_handler(request: Request, exc: PrerequisiteMissingError):
        return _format_error(
            status_code=424,
            code="prerequisite_missing",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(ApiKeyRequiredError)
    async def api_key_required_handler(request: Request, exc: ApiKeyRequiredError):
        return _format_error(
            status_code=401,
            code="api_key_required",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(ApiKeyInvalidError)
    async def api_key_invalid_handler(request: Request, exc: ApiKeyInvalidError):
        return _format_error(
            status_code=401,
            code="api_key_invalid",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(ProviderAuthError)
    async def provider_auth_handler(request: Request, exc: ProviderAuthError):
        err_msg = str(exc)
        lower_msg = err_msg.lower()
        if "required" in lower_msg or "not configured with an api key" in lower_msg:
            return _format_error(
                status_code=401,
                code="api_key_required",
                message=err_msg,
                request=request,
            )
        if (
            "unauthorized" in lower_msg
            or "invalid" in lower_msg
            or "authentication" in lower_msg
            or "api_key_invalid" in lower_msg
        ):
            return _format_error(
                status_code=401,
                code="api_key_invalid",
                message=err_msg,
                request=request,
            )
        return _format_error(
            status_code=503,
            code="provider_not_configured",
            message=err_msg,
            request=request,
        )

    @app.exception_handler(ProviderRateLimitError)
    async def provider_rate_limit_handler(request: Request, exc: ProviderRateLimitError):
        return _format_error(
            status_code=429,
            code="provider_rate_limited",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(ProviderTimeoutError)
    async def provider_timeout_handler(request: Request, exc: ProviderTimeoutError):
        return _format_error(
            status_code=504,
            code="provider_timeout",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(ProviderBadRequestError)
    async def provider_bad_request_handler(request: Request, exc: ProviderBadRequestError):
        return _format_error(
            status_code=400,
            code="provider_bad_request",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable_handler(request: Request, exc: ProviderUnavailableError):
        return _format_error(
            status_code=502,
            code="provider_unavailable",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        return _format_error(
            status_code=404,
            code="not_found",
            message=str(exc).strip("'\""),
            request=request,
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        return _format_error(
            status_code=404,
            code="not_found",
            message=str(exc),
            request=request,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        route = f"{request.method} {request.url.path}"
        tb_str = traceback.format_exc()
        clean_tb = re.sub(r"sk-ant-[a-zA-Z0-9_\-]+", "sk-ant-***", tb_str)
        logger.error(
            f"[{request_id}] Unhandled server error on {route}: {exc.__class__.__name__}: {exc}\n{clean_tb}"
        )
        return _internal_error(exc, request)


    # 4. Include API Routers under /api prefix
    app.include_router(router, prefix="/api")

    from app.api.workspace_routes import router as workspace_router
    from app.api.chat_routes import router as chat_router
    app.include_router(workspace_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")

    # 5. Static UI and backwards-compatible redirects
    from app.api.static import setup_static_and_ui
    setup_static_and_ui(app)

    return app


app = create_app()

