"""
Pydantic models — single source of truth for all I/O shapes.
"""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── User / Portfolio ──────────────────────────────────────────────────────────

class KYC(BaseModel):
    status: str

class Position(BaseModel):
    ticker: str
    exchange: str
    quantity: float
    avg_cost: float
    currency: str
    purchased_at: str

class Preferences(BaseModel):
    preferred_benchmark: Optional[str] = "S&P 500"
    reporting_currency: Optional[str] = "USD"
    income_focus: Optional[bool] = False

class UserProfile(BaseModel):
    user_id: str
    name: str
    age: int
    country: str
    base_currency: str
    kyc: KYC
    risk_profile: str
    positions: list[Position] = []
    preferences: Preferences = Field(default_factory=Preferences)


# ── Safety ────────────────────────────────────────────────────────────────────

class SafetyVerdict(BaseModel):
    blocked: bool
    category: Optional[str] = None
    message: Optional[str] = None


# ── Classifier ────────────────────────────────────────────────────────────────

class ClassifierEntities(BaseModel):
    tickers: Optional[list[str]] = None
    topics: Optional[list[str]] = None
    sectors: Optional[list[str]] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    rate: Optional[float] = None
    period_years: Optional[int] = None
    frequency: Optional[str] = None
    horizon: Optional[str] = None
    time_period: Optional[str] = None
    index: Optional[str] = None
    action: Optional[str] = None
    goal: Optional[str] = None

class ClassifierResult(BaseModel):
    intent: str
    agent: str
    entities: ClassifierEntities = Field(default_factory=ClassifierEntities)
    safety_verdict: Optional[str] = None


# ── HTTP Request / Response ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    user: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    prior_turns: list[str] = []


# ── Portfolio Health ──────────────────────────────────────────────────────────

class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: Literal["low", "medium", "high"]

class Performance(BaseModel):
    total_return_pct: float
    annualized_return_pct: float

class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float

class Observation(BaseModel):
    severity: Literal["info", "warning", "critical"]
    text: str

class PortfolioHealthResponse(BaseModel):
    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: BenchmarkComparison
    observations: list[Observation]
    disclaimer: str
    build_guidance: Optional[str] = None


# ── Agent Stub ────────────────────────────────────────────────────────────────

class StubResponse(BaseModel):
    intent: str
    agent: str
    entities: dict[str, Any]
    message: str
    implemented: bool = False