"""
Intent Classifier — single LLM call per query.

Returns a ClassifierResult with: intent, agent, entities, safety_verdict.
On any LLM failure → falls back to general_query (never crashes).

Accepts optional `llm` parameter so tests can inject a mock
without setting OPENAI_API_KEY.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Optional

from src.models import ClassifierEntities, ClassifierResult

logger = logging.getLogger(__name__)

AGENT_TAXONOMY = {
    "portfolio_health":       "structured assessment of the user's portfolio (concentration, performance, benchmarking)",
    "market_research":        "factual/recent info about an instrument, sector, or market event",
    "investment_strategy":    "advice/strategy questions: should I buy/sell/rebalance, allocation guidance",
    "financial_planning":     "long-term planning: retirement, goals, savings rate",
    "financial_calculator":   "deterministic numerical computation: DCA, mortgage, tax, future value, FX",
    "risk_assessment":        "risk metrics, exposure analysis, what-if scenarios",
    "product_recommendation": "recommend specific products/funds matching user profile",
    "predictive_analysis":    "forward-looking analysis: forecasts, trend extrapolation",
    "customer_support":       "platform issues, account questions, how-to-use-app",
    "general_query":          "educational, conversational, definitions, greetings",
}

_SYSTEM_PROMPT = """You are an intent classifier for a wealth management AI platform.

Classify the user query and return ONLY valid JSON with this exact structure:
{
  "intent": "<short intent label>",
  "agent": "<agent name from taxonomy>",
  "entities": {
    "tickers": ["AAPL"],
    "topics": ["string"],
    "sectors": ["string"],
    "amount": 1000.0,
    "currency": "USD",
    "rate": 0.08,
    "period_years": 10,
    "frequency": "monthly",
    "horizon": "5_years",
    "time_period": "today",
    "index": "S&P 500",
    "action": "buy",
    "goal": "retirement"
  },
  "safety_verdict": "safe"
}

Rules:
- Only include entity fields explicitly present in the query. Omit absent fields.
- "agent" MUST be exactly one of: """ + ", ".join(AGENT_TAXONOMY.keys()) + """
- Tickers: uppercase, include exchange suffix if known (ASML.AS, HSBA.L, 7203.T).
- If query is gibberish or unclear → agent = "general_query".
- Horizon values: 6_months, 1_year, 3_years, 5_years, 10_years.
- Action values: buy, sell, hold, hedge, rebalance.
- Goal values: retirement, education, house, FIRE, emergency_fund.
- Frequency values: daily, weekly, monthly, yearly.
- safety_verdict: "safe" or "review" (informational only).

Agent taxonomy:
""" + "\n".join(f"- {k}: {v}" for k, v in AGENT_TAXONOMY.items())


def _build_messages(query: str, prior_turns: list[str]) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in prior_turns[-4:]:
        messages.append({"role": "user", "content": turn})
    messages.append({"role": "user", "content": query})
    return messages


def _parse_response(raw: str) -> ClassifierResult:
    data = json.loads(raw)
    entities_raw = data.get("entities", {})
    entities_clean = {k: v for k, v in entities_raw.items() if v is not None}
    return ClassifierResult(
        intent=data.get("intent", "general_query"),
        agent=data.get("agent", "general_query"),
        entities=ClassifierEntities(**entities_clean),
        safety_verdict=data.get("safety_verdict"),
    )


def _fallback(query: str) -> ClassifierResult:
    return ClassifierResult(
        intent="general_query",
        agent="general_query",
        entities=ClassifierEntities(),
        safety_verdict="safe",
    )


def classify(
    query: str,
    prior_turns: list[str] | None = None,
    llm: Any = None,
) -> ClassifierResult:
    """
    Classify a user query.

    Parameters
    ----------
    query       : The user's message.
    prior_turns : Previous user turns in the session (for follow-up resolution).
    llm         : Optional injectable LLM callable (used in tests to mock).
                  If None, uses the real OpenAI client.

    Returns
    -------
    ClassifierResult — never raises.
    """
    prior_turns = prior_turns or []

    try:
        if llm is not None:
            # Test / mock path
            raw = llm(query=query, prior_turns=prior_turns)
            if isinstance(raw, dict):
                return ClassifierResult(
                    intent=raw.get("intent", "general_query"),
                    agent=raw.get("agent", "general_query"),
                    entities=ClassifierEntities(**(raw.get("entities") or {})),
                    safety_verdict=raw.get("safety_verdict"),
                )
            if isinstance(raw, ClassifierResult):
                return raw
            return _fallback(query)

        # Real OpenAI path
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        messages = _build_messages(query, prior_turns)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=400,
        )
        raw_content = response.choices[0].message.content
        return _parse_response(raw_content)

    except Exception as exc:
        logger.warning("Classifier failed for query %r: %s", query, exc)
        return _fallback(query)