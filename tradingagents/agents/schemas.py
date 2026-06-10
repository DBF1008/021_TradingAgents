"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tradingagents.agents.utils.rating import parse_rating


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Heuristic recovery from free-text (Tier 3 of the degradation chain)
# ---------------------------------------------------------------------------
#
# Each ``recover_*`` function mirrors the corresponding ``render_*``:
# it attempts to reconstruct a typed Pydantic instance from prose that
# *should* contain the same section headers but may have drifted in
# formatting.  Returns ``None`` when the text is too unstructured to
# salvage, letting the caller fall through to Tier 4 (raw text).
# ---------------------------------------------------------------------------

# -- shared helpers --

# Matches a section that starts with an optional-bold header label followed
# by a colon, and runs until the next bold header or end of text.
_SECTION_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _extract_section(text: str, header: str) -> Optional[str]:
    """Extract the content following a ``**Header**: …`` label.

    Tolerates optional markdown bold, colon or full-width colon, and
    captures everything up to the next ``**…**:`` header or end-of-text.
    """
    if header not in _SECTION_RE_CACHE:
        _SECTION_RE_CACHE[header] = re.compile(
            rf"\*{{0,2}}{re.escape(header)}\*{{0,2}}"  # optional bold
            rf"\s*[:：]\s*"                              # colon separator
            rf"(.*?)"                                    # content (lazy)
            rf"(?=\n\s*\*\*\w|\nFINAL TRANSACTION|\Z)",  # next header or EOF
            re.DOTALL | re.IGNORECASE,
        )
    m = _SECTION_RE_CACHE[header].search(text)
    if m:
        val = m.group(1).strip()
        return val if val else None
    return None


# Lines that look like bold-header metadata or the FINAL TRANSACTION marker.
_META_LINE_RE = re.compile(
    r"^\s*(?:\*\*\w.*?[:：]|FINAL TRANSACTION PROPOSAL)", re.IGNORECASE,
)


def _body_text(text: str) -> str:
    """Return *text* with bold-header lines and marker lines stripped.

    Used as a last-resort fallback for text fields (e.g. ``reasoning``)
    when ``_extract_section`` cannot find a matching header.  Stripping
    metadata lines prevents the ``render_*`` functions from producing
    duplicate headers in the final output.
    """
    body = []
    for line in text.splitlines():
        if _META_LINE_RE.match(line):
            continue
        body.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()
    return result


_PORTFOLIO_RATING_MAP = {r.value.lower(): r for r in PortfolioRating}
_TRADER_ACTION_MAP = {a.value.lower(): a for a in TraderAction}
_SENTIMENT_BAND_MAP = {b.value.lower(): b for b in SentimentBand}

_ACTION_LABEL_RE = re.compile(
    r"(?:action|final\s+transaction\s+proposal)\*{0,2}\s*[:：]\s*\*{0,2}\s*(\w+)",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(
    r"(?:score)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*/\s*10",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"confidence\s*[:：]?\s*\*{0,2}\s*(low|medium|high)",
    re.IGNORECASE,
)
_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")


# -- per-schema recovery functions --


def recover_research_plan(text: str) -> Optional[ResearchPlan]:
    """Heuristically reconstruct a ResearchPlan from free-text prose."""
    rating_str = parse_rating(text, default="")
    if not rating_str:
        return None
    rating = _PORTFOLIO_RATING_MAP.get(rating_str.lower())
    if rating is None:
        return None

    rationale = _extract_section(text, "Rationale") or _body_text(text)
    strategic_actions = _extract_section(text, "Strategic Actions") or ""

    return ResearchPlan(
        recommendation=rating,
        rationale=rationale,
        strategic_actions=strategic_actions,
    )


def recover_trader_proposal(text: str) -> Optional[TraderProposal]:
    """Heuristically reconstruct a TraderProposal from free-text prose."""
    action = None
    m = _ACTION_LABEL_RE.search(text)
    if m:
        action = _TRADER_ACTION_MAP.get(m.group(1).strip().lower())
    if action is None:
        return None

    reasoning = _extract_section(text, "Reasoning") or _body_text(text)

    entry_price = None
    ep_str = _extract_section(text, "Entry Price")
    if ep_str:
        fm = _FLOAT_RE.search(ep_str)
        if fm:
            entry_price = float(fm.group(1))

    stop_loss = None
    sl_str = _extract_section(text, "Stop Loss")
    if sl_str:
        fm = _FLOAT_RE.search(sl_str)
        if fm:
            stop_loss = float(fm.group(1))

    position_sizing = _extract_section(text, "Position Sizing")

    return TraderProposal(
        action=action,
        reasoning=reasoning,
        entry_price=entry_price,
        stop_loss=stop_loss,
        position_sizing=position_sizing,
    )


def recover_pm_decision(text: str) -> Optional[PortfolioDecision]:
    """Heuristically reconstruct a PortfolioDecision from free-text prose."""
    rating_str = parse_rating(text, default="")
    if not rating_str:
        return None
    rating = _PORTFOLIO_RATING_MAP.get(rating_str.lower())
    if rating is None:
        return None

    executive_summary = _extract_section(text, "Executive Summary") or _body_text(text)
    investment_thesis = _extract_section(text, "Investment Thesis") or ""

    price_target = None
    pt_str = _extract_section(text, "Price Target")
    if pt_str:
        fm = _FLOAT_RE.search(pt_str)
        if fm:
            price_target = float(fm.group(1))

    time_horizon = _extract_section(text, "Time Horizon")

    return PortfolioDecision(
        rating=rating,
        executive_summary=executive_summary,
        investment_thesis=investment_thesis,
        price_target=price_target,
        time_horizon=time_horizon,
    )


def recover_sentiment_report(text: str) -> Optional[SentimentReport]:
    """Heuristically reconstruct a SentimentReport from free-text prose."""
    # Band: search for any known band value in the text
    band = None
    # Try longer names first to avoid "Bullish" matching before "Mildly Bullish"
    for band_val in sorted(_SENTIMENT_BAND_MAP, key=len, reverse=True):
        if band_val in text.lower():
            band = _SENTIMENT_BAND_MAP[band_val]
            break
    if band is None:
        return None

    # Score
    score = 5.0  # default neutral
    m = _SCORE_RE.search(text)
    if m:
        score = min(10.0, max(0.0, float(m.group(1))))

    # Confidence
    confidence = "medium"
    m = _CONFIDENCE_RE.search(text)
    if m:
        confidence = m.group(1).lower()

    return SentimentReport(
        overall_band=band,
        overall_score=score,
        confidence=confidence,
        narrative=_body_text(text) or text.strip(),
    )
