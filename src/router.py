"""
Agent router — dispatches a ClassifierResult to the correct agent.

Routing table:
  portfolio_health → src.agents.portfolio_health
  all others       → src.agents.stubs (structured stub response)
"""
from __future__ import annotations
from typing import Any

from src.models import ClassifierResult
from src.agents import portfolio_health
from src.agents.stubs import run_stub


def route(
    result: ClassifierResult,
    user: dict[str, Any] | None,
    llm: Any = None,
) -> dict[str, Any]:
    """
    Route the classified query to the appropriate agent.

    Parameters
    ----------
    result : ClassifierResult from the intent classifier.
    user   : User profile dict (required for portfolio_health).
    llm    : Optional injectable LLM (passed through to agents).

    Returns
    -------
    Plain dict — always JSON-serialisable, never raises.
    """
    if result.agent == "portfolio_health":
        if user is None:
            return {
                "intent":      result.intent,
                "agent":       result.agent,
                "message":     (
                    "No user profile provided. "
                    "Please log in to view your portfolio health."
                ),
                "implemented": True,
            }
        return portfolio_health.run(user, llm=llm)

    # All other agents → structured stub
    return run_stub(result)