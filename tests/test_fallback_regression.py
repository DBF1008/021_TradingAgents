"""Regression tests for structured-output fallback paths.

Covers the gaps left by test_structured_agents.py and test_memory_log.py:

1. Invoke-time fallback for Trader, Research Manager, and Portfolio Manager
   (only Sentiment Analyst was covered previously).
2. render → parse_rating full-enumeration consistency.
3. Realistic free-text variants → parse_rating end-to-end.
4. Free-text → SignalProcessor end-to-end.
5. Free-text → TradingMemoryLog end-to-end (store, load, past_context).
6. Trader FINAL TRANSACTION PROPOSAL marker under fallback.
7. Degradation edge cases (empty input, multiple labels).
8. Mixed fallback scenarios across a pipeline.
"""

import logging
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating
from tradingagents.graph.signal_processing import SignalProcessor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _invoke_time_fallback_llm(freetext_content, exception=None):
    """LLM whose structured binding raises on invoke; plain invoke returns freetext."""
    if exception is None:
        exception = ValueError("bad JSON from model")
    structured = MagicMock()
    structured.invoke.side_effect = exception
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = MagicMock(content=freetext_content)
    return llm


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
    }


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _make_pm_state(past_context=""):
    return {
        "company_of_interest": "NVDA",
        "past_context": past_context,
        "risk_debate_state": {
            "history": "Risk debate history.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_plan": "Research plan.",
        "trader_investment_plan": "Trader plan.",
    }


def _make_memory_log(tmp_path, filename="trading_memory.md"):
    return TradingMemoryLog({"memory_log_path": str(tmp_path / filename)})


# ---------------------------------------------------------------------------
# Section 1: Invoke-time fallback for Trader, RM, PM
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInvokeTimeFallback:
    """Structured call raises at invoke time; agent must fall back to free-text."""

    def test_trader_invoke_time_fallback(self):
        freetext = (
            "**Action**: Sell\n\n"
            "**Reasoning**: Guidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = _invoke_time_fallback_llm(freetext)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == freetext
        assert result["messages"][0].content == freetext

    def test_research_manager_invoke_time_fallback(self):
        freetext = "**Recommendation**: Hold\n\n**Rationale**: Balanced.\n\n**Strategic Actions**: Wait."
        llm = _invoke_time_fallback_llm(freetext)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == freetext
        assert result["investment_debate_state"]["judge_decision"] == freetext

    def test_portfolio_manager_invoke_time_fallback(self):
        freetext = "**Rating**: Sell\n\n**Executive Summary**: Exit.\n\n**Investment Thesis**: Weak outlook."
        llm = _invoke_time_fallback_llm(freetext)
        pm = create_portfolio_manager(llm)
        result = pm(_make_pm_state())
        assert result["final_trade_decision"] == freetext

    def test_invoke_time_fallback_logs_warning(self, caplog):
        freetext = "**Rating**: Hold\nDetails."
        llm = _invoke_time_fallback_llm(freetext)
        pm = create_portfolio_manager(llm)
        with caplog.at_level(logging.WARNING, logger="tradingagents.agents.utils.structured"):
            pm(_make_pm_state())
        assert any("structured-output invocation failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Section 2: render → parse_rating full enumeration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderToParseRatingEnumeration:
    """Every enum value, when rendered, must be correctly extracted by parse_rating."""

    @pytest.mark.parametrize("rating", list(PortfolioRating))
    def test_render_pm_decision_all_ratings_parse_correctly(self, rating):
        decision = PortfolioDecision(
            rating=rating,
            executive_summary="Summary.",
            investment_thesis="Thesis.",
        )
        md = render_pm_decision(decision)
        assert parse_rating(md) == rating.value

    @pytest.mark.parametrize("rating", list(PortfolioRating))
    def test_render_research_plan_all_ratings_extractable(self, rating):
        """RM render uses 'Recommendation' not 'Rating', so parse_rating uses
        the word-scan fallback (Pass 2).  This still works because the rating
        value itself appears as a standalone word."""
        plan = ResearchPlan(
            recommendation=rating,
            rationale="Rationale.",
            strategic_actions="Actions.",
        )
        md = render_research_plan(plan)
        assert parse_rating(md) == rating.value

    @pytest.mark.parametrize("action", list(TraderAction))
    def test_render_trader_proposal_markers_present(self, action):
        proposal = TraderProposal(action=action, reasoning="Reasoning.")
        md = render_trader_proposal(proposal)
        assert f"**Action**: {action.value}" in md
        assert f"FINAL TRANSACTION PROPOSAL: **{action.value.upper()}**" in md


# ---------------------------------------------------------------------------
# Section 3: Fallback free-text → parse_rating end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFreetextToParseRating:
    """Realistic LLM free-text variants fed to parse_rating."""

    def test_explicit_rating_label(self):
        assert parse_rating("Rating: Buy\nEnter at $190.") == "Buy"

    def test_bold_rating_label(self):
        assert parse_rating("**Rating**: Overweight\nBuild position.") == "Overweight"

    def test_bold_rating_value(self):
        assert parse_rating("Rating: **Sell**\nExit immediately.") == "Sell"

    def test_rating_in_prose_no_label(self):
        """No 'Rating:' label; word scan picks first rating word."""
        assert parse_rating("My assessment is Underweight given the headwinds.") == "Underweight"

    def test_no_rating_word_defaults_hold(self):
        assert parse_rating("Complex situation with no clear direction.") == "Hold"

    def test_label_wins_over_conflicting_prose(self):
        text = "The buy thesis is weak.\nRating: Sell\nExit before earnings."
        assert parse_rating(text) == "Sell"

    def test_first_rating_word_wins_without_label(self):
        """When no label exists, Pass 2 picks the first rating word.
        This documents a known limitation of the word-scan fallback:
        the first rating word may be in a dismissive context."""
        text = "The buy thesis is weak. We recommend a sell."
        assert parse_rating(text) == "Buy"

    @pytest.mark.parametrize("rating", RATINGS_5_TIER)
    def test_all_5_ratings_with_label_format(self, rating):
        assert parse_rating(f"Rating: {rating}\nDetails.") == rating


# ---------------------------------------------------------------------------
# Section 4: Fallback free-text → SignalProcessor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFreetextToSignalProcessor:
    """SignalProcessor.process_signal with fallback-like PM text."""

    def test_with_rating_label(self):
        sp = SignalProcessor()
        assert sp.process_signal("Rating: Overweight\nBuild position.") == "Overweight"

    def test_with_rating_word_in_prose_no_label(self):
        sp = SignalProcessor()
        assert sp.process_signal("We recommend a buy at these levels.") == "Buy"

    def test_with_completely_unstructured_text(self):
        sp = SignalProcessor()
        assert sp.process_signal("Market outlook is uncertain.") == "Hold"


# ---------------------------------------------------------------------------
# Section 5: Fallback free-text → Memory Log end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFreetextToMemoryLog:
    """Memory log storage and retrieval with both structured and fallback PM output."""

    def test_stores_correct_rating_from_structured_pm_output(self, tmp_path):
        decision = PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="Build position.",
            investment_thesis="AI thesis intact.",
        )
        md = render_pm_decision(decision)
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", md)
        assert log.load_entries()[0]["rating"] == "Overweight"

    def test_stores_correct_rating_from_freetext_with_label(self, tmp_path):
        freetext = "Rating: Buy\nEnter at $190, 6% portfolio cap."
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", freetext)
        assert log.load_entries()[0]["rating"] == "Buy"

    def test_stores_hold_when_freetext_has_no_rating(self, tmp_path):
        freetext = "Complex situation. No clear directional signal."
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", freetext)
        entry = log.load_entries()[0]
        assert entry["rating"] == "Hold"
        raw_text = (tmp_path / "trading_memory.md").read_text(encoding="utf-8")
        assert "| Hold |" in raw_text

    def test_decision_content_preserved_after_fallback(self, tmp_path):
        freetext = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Strong AI thesis.\n\n"
            "**Investment Thesis**: Capex cycle drives revenue growth.\n\n"
            "**Price Target**: 250.0"
        )
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", freetext)
        entry = log.load_entries()[0]
        assert "Strong AI thesis" in entry["decision"]
        assert "Capex cycle" in entry["decision"]

    def test_roundtrip_structured_pm_through_past_context(self, tmp_path):
        """render → store → resolve → get_past_context for a structured PM output."""
        decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Enter at $189.",
            investment_thesis="AI capex cycle intact.",
        )
        md = render_pm_decision(decision)
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", md)
        log.update_with_outcome("NVDA", "2026-01-05", 0.05, 0.02, 5, "Correct call.")
        ctx = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in ctx
        assert "Buy" in ctx
        assert "Correct call." in ctx
        assert "DECISION:" in ctx

    def test_roundtrip_fallback_pm_through_past_context(self, tmp_path):
        """Free-text PM output → store → resolve → get_past_context."""
        freetext = "Rating: Sell\nExit position. Fundamentals deteriorating."
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-05", freetext)
        log.update_with_outcome("NVDA", "2026-01-05", -0.03, -0.01, 5, "Wrong call.")
        ctx = log.get_past_context("NVDA")
        assert "Past analyses of NVDA" in ctx
        assert "Sell" in ctx
        assert "Wrong call." in ctx

    @pytest.mark.parametrize("rating", list(PortfolioRating))
    def test_all_5_ratings_via_render_pm_store_load(self, tmp_path, rating):
        """Each PortfolioRating, rendered and stored, must load with the correct rating."""
        decision = PortfolioDecision(
            rating=rating,
            executive_summary="Summary.",
            investment_thesis="Thesis.",
        )
        md = render_pm_decision(decision)
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", md)
        assert log.load_entries()[0]["rating"] == rating.value


# ---------------------------------------------------------------------------
# Section 6: Trader FINAL TRANSACTION PROPOSAL marker under fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraderProposalMarker:
    """Document the behavior of the FINAL TRANSACTION PROPOSAL marker under fallback."""

    def test_fallback_with_marker_present(self):
        freetext = (
            "**Action**: Buy\n\n"
            "Strong setup.\n\n"
            "FINAL TRANSACTION PROPOSAL: **BUY**"
        )
        llm = _invoke_time_fallback_llm(freetext)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in result["trader_investment_plan"]

    def test_fallback_without_marker(self):
        """When the LLM free-text omits the marker, the output simply lacks it.
        This is acceptable degradation: conditional_logic.py uses count-based
        routing, not marker-based."""
        freetext = "I recommend buying NVDA based on strong technicals."
        llm = _invoke_time_fallback_llm(freetext)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert "FINAL TRANSACTION PROPOSAL" not in result["trader_investment_plan"]
        assert result["trader_investment_plan"] == freetext


# ---------------------------------------------------------------------------
# Section 7: Degradation edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDegradationEdgeCases:
    """Boundary conditions with worst-case LLM outputs."""

    def test_parse_rating_empty_string(self):
        assert parse_rating("") == "Hold"

    def test_parse_rating_whitespace_only(self):
        assert parse_rating("   \n\n  ") == "Hold"

    def test_parse_rating_multiple_labels_first_wins(self):
        text = "Rating: Buy\nSome analysis.\nRating: Sell"
        assert parse_rating(text) == "Buy"

    def test_memory_log_stores_empty_freetext(self, tmp_path):
        log = _make_memory_log(tmp_path)
        log.store_decision("NVDA", "2026-01-10", "")
        entry = log.load_entries()[0]
        assert entry["rating"] == "Hold"

    def test_signal_processor_empty_text(self):
        sp = SignalProcessor()
        assert sp.process_signal("") == "Hold"


# ---------------------------------------------------------------------------
# Section 8: Mixed fallback scenarios across a pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMixedFallbackScenarios:
    """Simulate a pipeline where agents have different fallback states."""

    def test_rm_structured_trader_fallback_pm_structured(self):
        """RM succeeds with structured output, Trader falls back, PM succeeds."""
        # RM: structured
        rm_plan = ResearchPlan(
            recommendation=PortfolioRating.BUY,
            rationale="Bull case stronger.",
            strategic_actions="Build position.",
        )
        rm_structured = MagicMock()
        rm_structured.invoke.return_value = rm_plan
        rm_llm = MagicMock()
        rm_llm.with_structured_output.return_value = rm_structured
        rm = create_research_manager(rm_llm)
        rm_result = rm(_make_rm_state())
        assert "**Recommendation**: Buy" in rm_result["investment_plan"]

        # Trader: invoke-time fallback
        trader_freetext = "I recommend buying. FINAL TRANSACTION PROPOSAL: **BUY**"
        trader_llm = _invoke_time_fallback_llm(trader_freetext)
        trader = create_trader(trader_llm)
        trader_state = _make_trader_state()
        trader_state["investment_plan"] = rm_result["investment_plan"]
        trader_result = trader(trader_state)
        assert trader_result["trader_investment_plan"] == trader_freetext

        # PM: structured
        pm_decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Enter at $189.",
            investment_thesis="AI thesis intact.",
        )
        pm_structured = MagicMock()
        pm_structured.invoke.return_value = pm_decision
        pm_llm = MagicMock()
        pm_llm.with_structured_output.return_value = pm_structured
        pm = create_portfolio_manager(pm_llm)
        pm_state = _make_pm_state()
        pm_state["investment_plan"] = rm_result["investment_plan"]
        pm_state["trader_investment_plan"] = trader_result["trader_investment_plan"]
        pm_result = pm(pm_state)

        # End-to-end: PM structured output → parse_rating succeeds
        assert "**Rating**: Buy" in pm_result["final_trade_decision"]
        assert parse_rating(pm_result["final_trade_decision"]) == "Buy"

    def test_all_agents_invoke_time_fallback(self):
        """Every agent hits invoke-time fallback; downstream parse still works."""
        rm_freetext = "**Recommendation**: Hold\n\n**Rationale**: Balanced.\n\n**Strategic Actions**: Wait."
        rm_llm = _invoke_time_fallback_llm(rm_freetext)
        rm = create_research_manager(rm_llm)
        rm_result = rm(_make_rm_state())
        assert rm_result["investment_plan"] == rm_freetext

        trader_freetext = "Hold position. No action needed."
        trader_llm = _invoke_time_fallback_llm(trader_freetext)
        trader = create_trader(trader_llm)
        trader_state = _make_trader_state()
        trader_state["investment_plan"] = rm_result["investment_plan"]
        trader_result = trader(trader_state)
        assert trader_result["trader_investment_plan"] == trader_freetext

        pm_freetext = "Rating: Hold\nMaintain current position."
        pm_llm = _invoke_time_fallback_llm(pm_freetext)
        pm = create_portfolio_manager(pm_llm)
        pm_state = _make_pm_state()
        pm_state["investment_plan"] = rm_result["investment_plan"]
        pm_state["trader_investment_plan"] = trader_result["trader_investment_plan"]
        pm_result = pm(pm_state)

        # PM free-text → downstream parse
        assert parse_rating(pm_result["final_trade_decision"]) == "Hold"
        sp = SignalProcessor()
        assert sp.process_signal(pm_result["final_trade_decision"]) == "Hold"

    def test_all_agents_bind_time_fallback(self):
        """Every agent's provider lacks with_structured_output entirely."""

        def _bind_time_fallback_llm(freetext_content):
            llm = MagicMock()
            llm.with_structured_output.side_effect = NotImplementedError("not supported")
            llm.invoke.return_value = MagicMock(content=freetext_content)
            return llm

        rm_freetext = "**Recommendation**: Sell\n\n**Rationale**: Bear case wins.\n\n**Strategic Actions**: Exit."
        rm = create_research_manager(_bind_time_fallback_llm(rm_freetext))
        rm_result = rm(_make_rm_state())
        assert rm_result["investment_plan"] == rm_freetext

        trader_freetext = "**Action**: Sell\n\nGuidance cut.\n\nFINAL TRANSACTION PROPOSAL: **SELL**"
        trader = create_trader(_bind_time_fallback_llm(trader_freetext))
        trader_state = _make_trader_state()
        trader_state["investment_plan"] = rm_result["investment_plan"]
        trader_result = trader(trader_state)
        assert trader_result["trader_investment_plan"] == trader_freetext

        pm_freetext = "**Rating**: Sell\n\n**Executive Summary**: Exit now.\n\n**Investment Thesis**: Deteriorating."
        pm = create_portfolio_manager(_bind_time_fallback_llm(pm_freetext))
        pm_state = _make_pm_state()
        pm_state["investment_plan"] = rm_result["investment_plan"]
        pm_state["trader_investment_plan"] = trader_result["trader_investment_plan"]
        pm_result = pm(pm_state)

        assert parse_rating(pm_result["final_trade_decision"]) == "Sell"
        sp = SignalProcessor()
        assert sp.process_signal(pm_result["final_trade_decision"]) == "Sell"
