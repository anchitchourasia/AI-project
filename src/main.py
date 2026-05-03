"""
Valura AI — FastAPI application entry point.

Endpoints:
  POST /query        — single-turn JSON response
  POST /query/stream — streaming SSE response
  GET  /health       — liveness check
"""
from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.models import QueryRequest
from src.safety import check as safety_check
from src.classifier import classify
from src.router import route
from src.memory import memory

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Valura AI",
    description="AI-powered wealth management assistant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_user(request: QueryRequest) -> dict[str, Any] | None:
    """
    Return the user profile dict from the request.
    In production this would hit a user service / DB.
    For the assignment we accept inline user dict directly.
    """
    return request.user


def _process(request: QueryRequest) -> dict[str, Any]:
    """
    Core pipeline:
      1. Safety check  (local, no LLM)
      2. Session memory (load prior turns)
      3. Intent classification (LLM)
      4. Agent routing + execution
      5. Persist turns to memory
    """
    # 1. Safety
    verdict = safety_check(request.query)
    if verdict.blocked:
        return {
            "blocked":  True,
            "category": verdict.category,
            "message":  verdict.message,
        }

    # 2. Session memory
    session_id = request.session_id or str(uuid.uuid4())
    session    = memory.get_or_create(session_id)
    prior_turns = session.prior_user_turns()

    # 3. Classify
    classifier_result = classify(
        query=request.query,
        prior_turns=prior_turns,
    )

    # 4. Route
    user     = _resolve_user(request)
    response = route(classifier_result, user=user)

    # 5. Persist
    session.add_turn("user",      request.query)
    session.add_turn("assistant", json.dumps(response))

    return {
        "session_id": session_id,
        "intent":     classifier_result.intent,
        "agent":      classifier_result.agent,
        "response":   response,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness check."""
    return JSONResponse({"status": "ok", "service": "valura-ai"})


@app.post("/query")
async def query_endpoint(request: QueryRequest) -> JSONResponse:
    """
    Single-turn query — returns full JSON response.
    """
    try:
        result = _process(request)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("Unhandled error in /query: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest) -> EventSourceResponse:
    """
    Streaming SSE endpoint.

    Emits events:
      - type: "safety"     — if blocked
      - type: "meta"       — intent + agent classification
      - type: "chunk"      — response content chunks
      - type: "done"       — signals end of stream
      - type: "error"      — on unhandled exception
    """
    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # Safety check first
            verdict = safety_check(request.query)
            if verdict.blocked:
                yield {
                    "event": "safety",
                    "data":  json.dumps({
                        "blocked":  True,
                        "category": verdict.category,
                        "message":  verdict.message,
                    }),
                }
                return

            # Session memory
            session_id  = request.session_id or str(uuid.uuid4())
            session     = memory.get_or_create(session_id)
            prior_turns = session.prior_user_turns()

            # Classify
            classifier_result = classify(
                query=request.query,
                prior_turns=prior_turns,
            )

            # Emit meta event
            yield {
                "event": "meta",
                "data":  json.dumps({
                    "session_id": session_id,
                    "intent":     classifier_result.intent,
                    "agent":      classifier_result.agent,
                }),
            }

            # Route + execute
            user     = _resolve_user(request)
            response = route(classifier_result, user=user)

            # Persist
            session.add_turn("user",      request.query)
            session.add_turn("assistant", json.dumps(response))

            # Stream response as chunks
            response_str = json.dumps(response)
            chunk_size   = 200
            for i in range(0, len(response_str), chunk_size):
                yield {
                    "event": "chunk",
                    "data":  response_str[i: i + chunk_size],
                }

            yield {"event": "done", "data": ""}

        except Exception as exc:
            logger.exception("Unhandled error in /query/stream: %s", exc)
            yield {
                "event": "error",
                "data":  json.dumps({"error": str(exc)}),
            }

    return EventSourceResponse(event_generator())