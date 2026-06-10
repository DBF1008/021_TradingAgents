"""Tests for structured-output agents (Trader, Research Manager, Sentiment Analyst).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader, Research Manager, and Sentiment Analyst
so they share the same deterministic output shape.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
    recover_research_plan,
    recover_trader_proposal,
    recover_pm_decision,
    recover_sentiment_report,
    _extract_section,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.structured import (
    _extract_json_block,
    _try_parse_json,
)


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5" in md
        assert "**Stop Loss**: 178.0" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_optional_fields_omitted_when_absent(self):
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        assert "Entry Price" not in md
        assert "Stop Loss" not in md
        assert "Position Sizing" not in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "**Entry Price**: 189.5" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_investment_plan(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The investment plan is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Proposed Investment Plan" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        # Tier 3 recovery reconstructs the output with stable headers
        assert "**Action**: Sell" in plan
        assert "**Reasoning**:" in plan
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in plan


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


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


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced view across both sides.",
            strategic_actions="Hold current position; reassess after earnings.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; AI tailwind intact.",
            strategic_actions="Build position gradually over two weeks.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**Strategic Actions**: Build position" in ip

    def test_prompt_uses_5_tier_rating_scale(self):
        """The RM prompt must list all five tiers so the schema enum matches user expectations."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Recommendation**: Sell\n\n**Rationale**: ...\n\n**Strategic Actions**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band, overall_score=5.0,
                confidence="medium", narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )


def _make_sentiment_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "asset_type": "stock",
        "messages": [],
    }


def _structured_sentiment_llm(captured: dict, report: SentimentReport | None = None):
    """MagicMock LLM whose structured binding captures the prompt and returns
    a real SentimentReport so render_sentiment_report works."""
    if report is None:
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH, overall_score=7.5,
            confidence="high",
            narrative="StockTwits 75% bullish. News constructive. Reddit upbeat.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or report
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestSentimentAnalystAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="Mixed signals across sources.",
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, report))
        sr = analyst(_make_sentiment_state())["sentiment_report"]
        assert "**Overall Sentiment:** **Mildly Bearish**" in sr
        assert "(Score: 4.0/10)" in sr
        assert "Mixed signals across sources." in sr

    def test_sentiment_report_also_in_messages(self):
        captured = {}
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured))
        result = analyst(_make_sentiment_state())
        assert len(result["messages"]) == 1
        assert result["sentiment_report"] == result["messages"][0].content

    def test_prompt_contains_ticker(self):
        captured = {}
        create_sentiment_analyst(_structured_sentiment_llm(captured))(_make_sentiment_state())
        assert any("NVDA" in str(m) for m in captured["prompt"])

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain = "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n**Confidence:** Low\n\nLimited data."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain

    def test_falls_back_to_freetext_when_structured_call_fails(self):
        plain = "Fallback free-text sentiment."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON from model")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain


# ---------------------------------------------------------------------------
# JSON extraction helpers (Tier 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractJsonBlock:
    def test_clean_json(self):
        text = '{"action": "Buy", "reasoning": "Strong setup."}'
        assert _extract_json_block(text) == text

    def test_json_in_code_fence(self):
        text = 'Here is the output:\n```json\n{"action": "Buy", "reasoning": "ok"}\n```\nDone.'
        result = _extract_json_block(text)
        assert result is not None
        assert '"action": "Buy"' in result

    def test_json_in_bare_fence(self):
        text = '```\n{"action": "Sell"}\n```'
        result = _extract_json_block(text)
        assert result is not None
        assert '"action": "Sell"' in result

    def test_json_with_surrounding_prose(self):
        text = 'Based on my analysis:\n{"rating": "Buy", "executive_summary": "Enter now."}\nThank you.'
        result = _extract_json_block(text)
        assert result is not None
        assert '"rating": "Buy"' in result

    def test_trailing_comma_cleaned(self):
        text = '{"action": "Hold", "reasoning": "Balanced",}'
        result = _extract_json_block(text)
        assert result is not None
        assert result.endswith("}")
        assert ",}" not in result

    def test_no_json_returns_none(self):
        assert _extract_json_block("Just plain text with no JSON.") is None

    def test_broken_json_still_extracted(self):
        # _extract_json_block extracts, it doesn't validate
        text = '{"action": "Buy", "reasoning": }'
        result = _extract_json_block(text)
        assert result is not None


@pytest.mark.unit
class TestTryParseJson:
    def test_valid_trader_json(self):
        text = '{"action": "Buy", "reasoning": "Strong setup."}'
        result = _try_parse_json(text, TraderProposal)
        assert result is not None
        assert result.action == TraderAction.BUY
        assert result.reasoning == "Strong setup."

    def test_valid_json_in_fence(self):
        text = '```json\n{"action": "Sell", "reasoning": "Guidance cut."}\n```'
        result = _try_parse_json(text, TraderProposal)
        assert result is not None
        assert result.action == TraderAction.SELL

    def test_json_with_trailing_comma(self):
        text = '{"action": "Hold", "reasoning": "Balanced view.",}'
        result = _try_parse_json(text, TraderProposal)
        assert result is not None
        assert result.action == TraderAction.HOLD

    def test_valid_pm_json(self):
        text = '{"rating": "Overweight", "executive_summary": "Add gradually.", "investment_thesis": "AI cycle."}'
        result = _try_parse_json(text, PortfolioDecision)
        assert result is not None
        assert result.rating == PortfolioRating.OVERWEIGHT

    def test_invalid_json_returns_none(self):
        assert _try_parse_json("not json at all", TraderProposal) is None

    def test_valid_json_wrong_schema_returns_none(self):
        # Missing required fields for PortfolioDecision
        text = '{"action": "Buy"}'
        assert _try_parse_json(text, PortfolioDecision) is None

    def test_json_with_optional_fields(self):
        text = '{"action": "Buy", "reasoning": "ok", "entry_price": 189.5, "stop_loss": 178.0}'
        result = _try_parse_json(text, TraderProposal)
        assert result is not None
        assert result.entry_price == 189.5
        assert result.stop_loss == 178.0


# ---------------------------------------------------------------------------
# Section extraction helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractSection:
    def test_bold_header(self):
        text = "**Rating**: Buy\n\n**Executive Summary**: Enter at $189."
        assert _extract_section(text, "Executive Summary") == "Enter at $189."

    def test_plain_header(self):
        text = "Rating: Buy\nExecutive Summary: Enter now.\n**Investment Thesis**: Strong."
        assert _extract_section(text, "Executive Summary") == "Enter now."

    def test_multiline_content(self):
        text = "**Rationale**: Bull case carried.\nStrong tailwinds remain.\n\n**Strategic Actions**: Build position."
        result = _extract_section(text, "Rationale")
        assert result is not None
        assert "Bull case carried." in result
        assert "Strong tailwinds" in result

    def test_missing_header_returns_none(self):
        text = "**Rating**: Buy\n\n**Executive Summary**: ok"
        assert _extract_section(text, "Rationale") is None

    def test_last_section_captured(self):
        text = "**Rating**: Buy\n\n**Investment Thesis**: AI capex cycle intact."
        result = _extract_section(text, "Investment Thesis")
        assert result is not None
        assert "AI capex cycle intact" in result


# ---------------------------------------------------------------------------
# Per-schema recover functions (Tier 3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecoverResearchPlan:
    def test_full_headers(self):
        text = "**Recommendation**: Buy\n\n**Rationale**: Bull won.\n\n**Strategic Actions**: Build position."
        result = recover_research_plan(text)
        assert result is not None
        assert result.recommendation == PortfolioRating.BUY
        assert "Bull won" in result.rationale
        assert "Build position" in result.strategic_actions

    def test_all_5_tier_ratings(self):
        for rating in PortfolioRating:
            text = f"**Recommendation**: {rating.value}\n\n**Rationale**: r\n\n**Strategic Actions**: s"
            result = recover_research_plan(text)
            assert result is not None
            assert result.recommendation == rating

    def test_no_rating_returns_none(self):
        text = "The market is uncertain. No clear direction."
        assert recover_research_plan(text) is None

    def test_recovered_output_renders_stably(self):
        text = "**Recommendation**: Overweight\n\n**Rationale**: Thesis intact.\n\n**Strategic Actions**: Add slowly."
        result = recover_research_plan(text)
        rendered = render_research_plan(result)
        assert "**Recommendation**: Overweight" in rendered
        assert "**Rationale**:" in rendered
        assert "**Strategic Actions**:" in rendered


@pytest.mark.unit
class TestRecoverTraderProposal:
    def test_full_headers(self):
        text = "**Action**: Buy\n\n**Reasoning**: Strong setup.\n\nFINAL TRANSACTION PROPOSAL: **BUY**"
        result = recover_trader_proposal(text)
        assert result is not None
        assert result.action == TraderAction.BUY
        assert "Strong setup" in result.reasoning

    def test_action_from_final_proposal_line(self):
        text = "Based on analysis, I recommend buying.\nFINAL TRANSACTION PROPOSAL: **SELL**"
        result = recover_trader_proposal(text)
        assert result is not None
        assert result.action == TraderAction.SELL

    def test_with_optional_fields(self):
        text = (
            "**Action**: Buy\n\n**Reasoning**: ok\n\n"
            "**Entry Price**: 189.5\n\n**Stop Loss**: 178.0\n\n"
            "**Position Sizing**: 6% of portfolio\n\n"
            "FINAL TRANSACTION PROPOSAL: **BUY**"
        )
        result = recover_trader_proposal(text)
        assert result is not None
        assert result.entry_price == 189.5
        assert result.stop_loss == 178.0
        assert "6% of portfolio" in result.position_sizing

    def test_no_action_returns_none(self):
        text = "The market looks interesting but I can't decide."
        assert recover_trader_proposal(text) is None

    def test_all_3_actions(self):
        for action in TraderAction:
            text = f"**Action**: {action.value}\n\n**Reasoning**: reason."
            result = recover_trader_proposal(text)
            assert result is not None
            assert result.action == action

    def test_recovered_output_renders_stably(self):
        text = "**Action**: Sell\n\n**Reasoning**: Guidance cut.\n\nFINAL TRANSACTION PROPOSAL: **SELL**"
        result = recover_trader_proposal(text)
        rendered = render_trader_proposal(result)
        assert "**Action**: Sell" in rendered
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in rendered


@pytest.mark.unit
class TestRecoverPmDecision:
    def test_full_headers(self):
        text = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Enter at $189.\n\n"
            "**Investment Thesis**: AI capex cycle intact."
        )
        result = recover_pm_decision(text)
        assert result is not None
        assert result.rating == PortfolioRating.BUY
        assert "Enter at $189" in result.executive_summary
        assert "AI capex" in result.investment_thesis

    def test_with_optional_fields(self):
        text = (
            "**Rating**: Overweight\n\n"
            "**Executive Summary**: Add gradually.\n\n"
            "**Investment Thesis**: Thesis intact.\n\n"
            "**Price Target**: 215.0\n\n"
            "**Time Horizon**: 3-6 months"
        )
        result = recover_pm_decision(text)
        assert result is not None
        assert result.price_target == 215.0
        assert "3-6 months" in result.time_horizon

    def test_all_5_tier_ratings(self):
        for rating in PortfolioRating:
            text = (
                f"**Rating**: {rating.value}\n\n"
                f"**Executive Summary**: s\n\n"
                f"**Investment Thesis**: t"
            )
            result = recover_pm_decision(text)
            assert result is not None
            assert result.rating == rating

    def test_no_rating_returns_none(self):
        assert recover_pm_decision("No clear signal from the market.") is None

    def test_recovered_output_renders_stably(self):
        text = (
            "**Rating**: Sell\n\n"
            "**Executive Summary**: Exit position.\n\n"
            "**Investment Thesis**: Bear case prevailed."
        )
        result = recover_pm_decision(text)
        rendered = render_pm_decision(result)
        assert "**Rating**: Sell" in rendered
        assert "**Executive Summary**:" in rendered
        assert "**Investment Thesis**:" in rendered

    def test_parse_rating_works_on_recovered_output(self):
        """Critical: memory log + signal processor depend on parse_rating."""
        from tradingagents.agents.utils.rating import parse_rating
        text = "**Rating**: Underweight\n\n**Executive Summary**: Trim.\n\n**Investment Thesis**: Weak."
        result = recover_pm_decision(text)
        rendered = render_pm_decision(result)
        assert parse_rating(rendered) == "Underweight"


@pytest.mark.unit
class TestRecoverSentimentReport:
    def test_full_headers(self):
        text = (
            "**Overall Sentiment:** **Bullish** (Score: 7.2/10)\n"
            "**Confidence:** High\n\n"
            "Source breakdown here."
        )
        result = recover_sentiment_report(text)
        assert result is not None
        assert result.overall_band == SentimentBand.BULLISH
        assert result.overall_score == 7.2
        assert result.confidence == "high"

    def test_mildly_bullish_not_confused_with_bullish(self):
        text = "The sentiment is Mildly Bullish with a score of 6.0/10."
        result = recover_sentiment_report(text)
        assert result is not None
        assert result.overall_band == SentimentBand.MILDLY_BULLISH

    def test_all_six_bands(self):
        for band in SentimentBand:
            text = f"Sentiment is {band.value}. Score: 5.0/10. Confidence: medium."
            result = recover_sentiment_report(text)
            assert result is not None
            assert result.overall_band == band

    def test_score_defaults_to_5_when_missing(self):
        text = "The overall sentiment is Bearish. No score available."
        result = recover_sentiment_report(text)
        assert result is not None
        assert result.overall_score == 5.0

    def test_confidence_defaults_to_medium(self):
        text = "Sentiment is Bullish. Score: 8.0/10."
        result = recover_sentiment_report(text)
        assert result is not None
        assert result.confidence == "medium"

    def test_no_band_returns_none(self):
        assert recover_sentiment_report("The market data is inconclusive.") is None

    def test_recovered_output_renders_stably(self):
        text = "Sentiment: Mildly Bearish, score 4.0/10, confidence low. Details here."
        result = recover_sentiment_report(text)
        rendered = render_sentiment_report(result)
        assert "**Overall Sentiment:** **Mildly Bearish**" in rendered
        assert "(Score: 4.0/10)" in rendered
        assert "**Confidence:** Low" in rendered

    def test_score_clamped_to_range(self):
        text = "Bullish sentiment. Score: 15.0/10."
        result = recover_sentiment_report(text)
        assert result is not None
        assert result.overall_score == 10.0

    def test_narrative_excludes_header_lines(self):
        """Ensure recovered narrative doesn't duplicate headers when rendered."""
        text = (
            "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n"
            "**Confidence:** Low\n\n"
            "Limited data."
        )
        result = recover_sentiment_report(text)
        rendered = render_sentiment_report(result)
        # The rendered output should not double the header lines
        assert rendered.count("**Overall Sentiment:**") == 1
        assert rendered.count("**Confidence:**") == 1


# ---------------------------------------------------------------------------
# Integration: full degradation chain (all 4 tiers)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDegradationChainTrader:
    """Verify the Trader agent's output is stable across degradation tiers."""

    def test_tier2_dirty_json_recovery(self):
        """Structured call fails, model returns JSON in a code fence -> Tier 2 recovers."""
        dirty_json = '```json\n{"action": "Buy", "reasoning": "Strong setup."}\n```'
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=dirty_json)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan

    def test_tier3_heuristic_recovery(self):
        """Structured call fails, model returns prose with headers -> Tier 3 recovers."""
        prose = "**Action**: Sell\n\n**Reasoning**: Guidance cut hard.\n\nFINAL TRANSACTION PROPOSAL: **SELL**"
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=prose)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Sell" in plan
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in plan

    def test_tier4_raw_text_fallback(self):
        """Structured call fails, model returns unstructured prose -> raw text passthrough."""
        raw = "I think we should probably consider selling given the weak outlook."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=raw)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == raw


@pytest.mark.unit
class TestDegradationChainResearchManager:
    def test_tier2_dirty_json_recovery(self):
        dirty_json = (
            'Sure, here is the plan:\n'
            '```json\n{"recommendation": "Overweight", "rationale": "Bull won.", "strategic_actions": "Add."}\n```'
        )
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("parse error")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=dirty_json)
        from tradingagents.agents.managers.research_manager import create_research_manager
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**:" in ip

    def test_tier3_heuristic_recovery(self):
        prose = "**Recommendation**: Sell\n\n**Rationale**: Bear prevailed.\n\n**Strategic Actions**: Exit."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("parse error")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=prose)
        from tradingagents.agents.managers.research_manager import create_research_manager
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert "**Recommendation**: Sell" in result["investment_plan"]


@pytest.mark.unit
class TestDegradationChainPortfolioManager:
    def _make_pm_state(self):
        return {
            "company_of_interest": "NVDA",
            "risk_debate_state": {
                "history": "debate",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "count": 1,
            },
            "investment_plan": "plan",
            "trader_investment_plan": "trade",
        }

    def test_tier2_dirty_json_recovery(self):
        dirty_json = (
            '{"rating": "Buy", "executive_summary": "Enter now.", '
            '"investment_thesis": "AI cycle.", "price_target": 215.0,}'
        )
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("parse error")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=dirty_json)
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        pm = create_portfolio_manager(llm)
        result = pm(self._make_pm_state())
        ftd = result["final_trade_decision"]
        assert "**Rating**: Buy" in ftd
        assert "**Executive Summary**:" in ftd
        assert "**Price Target**: 215.0" in ftd

    def test_tier2_output_parseable_by_parse_rating(self):
        """Critical: SignalProcessor and memory log depend on parse_rating."""
        from tradingagents.agents.utils.rating import parse_rating
        dirty_json = '{"rating": "Underweight", "executive_summary": "Trim.", "investment_thesis": "Weak."}'
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("parse error")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=dirty_json)
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
        pm = create_portfolio_manager(llm)
        ftd = pm(self._make_pm_state())["final_trade_decision"]
        assert parse_rating(ftd) == "Underweight"
