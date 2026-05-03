"""
Stub responses for agents not yet fully implemented.
Returns structured JSON — never crashes, never errors.
These stubs correctly classify and acknowledge the intent
but defer execution to a future implementation.
"""
from __future__ import annotations
from typing import Any
from src.models import ClassifierResult


def run_stub(result: ClassifierResult) -> dict[str, Any]:
    """
    Return a structured stub response for unimplemented agents.

    Parameters
    ----------
    result : ClassifierResult from the intent classifier.

    Returns
    -------
    dict — always returns, never raises.
    """
    return {
        "intent":    result.intent,
        "agent":     result.agent,
        "entities":  result.entities.model_dump(exclude_none=True),
        "message": (
            f"The '{result.agent}' agent is recognised but not yet implemented "
            "in this build. Your query has been correctly classified and would "
            "be routed here in production."
        ),
        "implemented": False,
    }