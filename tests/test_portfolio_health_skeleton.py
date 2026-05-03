"""
Portfolio Health agent tests.

All tests use mock_llm (passed to run() but not used internally —
the agent does pure computation). No OPENAI_API_KEY required.
"""
from src.agents.portfolio_health import run


def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user, mock_llm):
    """
    usr_004 has no positions. Agent must not crash and must return disclaimer.
    """
    user     = load_user("usr_004")
    response = run(user, llm=mock_llm)

    assert response is not None
    assert "disclaimer" in response
    assert response["disclaimer"]


def test_portfolio_health_flags_concentration(load_user, mock_llm):
    """
    usr_003 has ~60% in NVDA. Agent must flag high concentration.
    """
    user     = load_user("usr_003")
    response = run(user, llm=mock_llm)

    assert response["concentration_risk"]["flag"] in {"high", "warning"}


def test_portfolio_health_includes_disclaimer(load_user, mock_llm):
    """
    Every response must include a non-investment-advice disclaimer.
    """
    user     = load_user("usr_001")
    response = run(user, llm=mock_llm)

    assert response["disclaimer"]
    assert "not" in response["disclaimer"].lower() or \
           "does not" in response["disclaimer"].lower()
    assert "investment advice" in response["disclaimer"].lower()


def test_portfolio_health_empty_portfolio_has_build_guidance(load_user, mock_llm):
    """
    Empty portfolio must return build_guidance to help user get started.
    """
    user     = load_user("usr_004")
    response = run(user, llm=mock_llm)

    assert "build_guidance" in response
    assert response["build_guidance"]


def test_portfolio_health_returns_all_required_keys(load_user, mock_llm):
    """
    Response must always contain all required top-level keys.
    """
    user     = load_user("usr_001")
    response = run(user, llm=mock_llm)

    required_keys = {
        "concentration_risk",
        "performance",
        "benchmark_comparison",
        "observations",
        "disclaimer",
    }
    assert required_keys.issubset(response.keys())


def test_portfolio_health_multi_currency(load_user, mock_llm):
    """
    usr_006 has EUR, GBP, JPY positions. Agent must not crash.
    """
    user     = load_user("usr_006")
    response = run(user, llm=mock_llm)

    assert response is not None
    assert "concentration_risk" in response
    assert "disclaimer" in response


def test_portfolio_health_retiree(load_user, mock_llm):
    """
    usr_008 is a retiree with income_focus=True. Agent must not crash.
    """
    user     = load_user("usr_008")
    response = run(user, llm=mock_llm)

    assert response is not None
    assert "observations" in response
    assert len(response["observations"]) > 0


def test_portfolio_health_concentration_risk_shape(load_user, mock_llm):
    """
    concentration_risk must have correct shape and valid flag value.
    """
    user     = load_user("usr_001")
    response = run(user, llm=mock_llm)

    cr = response["concentration_risk"]
    assert "top_position_pct"    in cr
    assert "top_3_positions_pct" in cr
    assert "flag"                in cr
    assert cr["flag"] in {"low", "medium", "high"}
    assert 0 <= cr["top_position_pct"]    <= 100
    assert 0 <= cr["top_3_positions_pct"] <= 100