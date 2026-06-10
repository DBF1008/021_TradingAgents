"""Test checkpoint resume: crash mid-analysis, re-run resumes from last node."""

import functools
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock, patch, call

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from tradingagents.graph.checkpointer import (
    checkpoint_step,
    clear_checkpoint,
    get_checkpointer,
    has_checkpoint,
    thread_id,
)
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Mutable flag to simulate crash on first run
_should_crash = False


class _SimpleState(TypedDict):
    count: int


def _node_a(state: _SimpleState) -> dict:
    return {"count": state["count"] + 1}


def _node_b(state: _SimpleState) -> dict:
    if _should_crash:
        raise RuntimeError("simulated mid-analysis crash")
    return {"count": state["count"] + 10}


def _build_graph() -> StateGraph:
    builder = StateGraph(_SimpleState)
    builder.add_node("analyst", _node_a)
    builder.add_node("trader", _node_b)
    builder.set_entry_point("analyst")
    builder.add_edge("analyst", "trader")
    builder.add_edge("trader", END)
    return builder


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        self.date = "2026-04-20"

    def test_crash_and_resume(self):
        """Crash at 'trader' node, then resume from checkpoint."""
        global _should_crash
        builder = _build_graph()
        tid = thread_id(self.ticker, self.date)
        cfg = {"configurable": {"thread_id": tid}}

        # Run 1: crash at trader node
        _should_crash = True
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config=cfg)

        # Checkpoint should exist at step 1 (analyst completed)
        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))
        step = checkpoint_step(self.tmpdir, self.ticker, self.date)
        self.assertEqual(step, 1)

        # Run 2: resume — trader succeeds this time
        _should_crash = False
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke(None, config=cfg)

        # analyst added 1, trader added 10 → 11
        self.assertEqual(result["count"], 11)

    def test_clear_checkpoint_allows_fresh_start(self):
        """After clearing, the graph starts from scratch."""
        global _should_crash
        builder = _build_graph()
        tid = thread_id(self.ticker, self.date)
        cfg = {"configurable": {"thread_id": tid}}

        # Create a checkpoint by crashing
        _should_crash = True
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config=cfg)

        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))

        # Clear it
        clear_checkpoint(self.tmpdir, self.ticker, self.date)
        self.assertFalse(has_checkpoint(self.tmpdir, self.ticker, self.date))

        # Fresh run succeeds from scratch
        _should_crash = False
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config=cfg)

        self.assertEqual(result["count"], 11)


    def test_different_date_starts_fresh(self):
        """A different date must NOT resume from an existing checkpoint."""
        global _should_crash
        builder = _build_graph()
        date2 = "2026-04-21"

        # Run with date1 — crash to leave a checkpoint
        _should_crash = True
        tid1 = thread_id(self.ticker, self.date)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid1}})

        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))

        # date2 should have no checkpoint
        self.assertFalse(has_checkpoint(self.tmpdir, self.ticker, date2))

        # Run with date2 — should start fresh and succeed
        _should_crash = False
        tid2 = thread_id(self.ticker, date2)
        self.assertNotEqual(tid1, tid2)

        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid2}})

        # Fresh run: analyst +1, trader +10 = 11
        self.assertEqual(result["count"], 11)

        # Original date checkpoint still exists (untouched)
        self.assertTrue(has_checkpoint(self.tmpdir, self.ticker, self.date))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Regression: checkpoint stability after resume fixes
# ---------------------------------------------------------------------------

def _fake_state():
    """Minimal final state dict for _run_graph."""
    return {
        "final_trade_decision": "Rating: Buy\nBuy NVDA.",
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-20",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "",
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "", "judge_decision": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1, "latest_speaker": "",
        },
    }


def _mock_graph(tmp_path, checkpoint_enabled=True):
    """Build a MagicMock TradingAgentsGraph with real memory log and paths."""
    g = MagicMock(spec=TradingAgentsGraph)
    g.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    g.log_states_dict = {}
    g.debug = False
    g.ticker = "NVDA"
    g.config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "checkpoint_enabled": checkpoint_enabled,
    }
    g.graph = MagicMock()
    g.graph.invoke.return_value = _fake_state()
    g.propagator = MagicMock()
    g.propagator.create_initial_state.return_value = _fake_state()
    g.propagator.get_graph_args.return_value = {}
    g.signal_processor = MagicMock()
    g.signal_processor.process_signal.return_value = "Buy"
    return g


class TestCheckpointStability:
    """Regression tests for the three checkpoint resume bugs."""

    def test_clear_checkpoint_before_store_decision(self, tmp_path):
        """clear_checkpoint must execute BEFORE store_decision in _run_graph
        so a crash between them cannot leave a stale checkpoint."""
        g = _mock_graph(tmp_path, checkpoint_enabled=True)

        call_order = []
        original_store = g.memory_log.store_decision

        def tracked_store(*a, **kw):
            call_order.append("store_decision")
            return original_store(*a, **kw)

        g.memory_log.store_decision = tracked_store

        with patch(
            "tradingagents.graph.trading_graph.clear_checkpoint"
        ) as mock_clear:
            mock_clear.side_effect = lambda *a, **kw: call_order.append("clear_checkpoint")

            g._run_graph = functools.partial(TradingAgentsGraph._run_graph, g)
            g._run_graph("NVDA", "2026-04-20")

        assert "clear_checkpoint" in call_order
        assert "store_decision" in call_order
        assert call_order.index("clear_checkpoint") < call_order.index("store_decision"), \
            f"clear_checkpoint must run before store_decision, got: {call_order}"

    def test_stale_checkpoint_cleared_when_result_exists(self, tmp_path):
        """If a checkpoint exists but the JSON result log is already on disk,
        propagate() must clear the stale checkpoint instead of resuming."""
        ticker = "NVDA"
        date = "2026-04-20"
        data_dir = str(tmp_path / "cache")
        results_dir = str(tmp_path / "results")

        # 1. Create a checkpoint
        tid = thread_id(ticker, date)
        with get_checkpointer(data_dir, ticker) as saver:
            builder = StateGraph(_SimpleState)
            builder.add_node("a", _node_a)
            builder.set_entry_point("a")
            builder.add_edge("a", END)
            graph = builder.compile(checkpointer=saver)
            graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid}})

        assert has_checkpoint(data_dir, ticker, date)

        # 2. Create the JSON result log (simulating a completed run)
        log_dir = Path(results_dir) / ticker / "TradingAgentsStrategy_logs"
        log_dir.mkdir(parents=True)
        (log_dir / f"full_states_log_{date}.json").write_text("{}", encoding="utf-8")

        # 3. Call propagate — it should detect and clear the stale checkpoint
        g = _mock_graph(tmp_path, checkpoint_enabled=True)
        g.config["data_cache_dir"] = data_dir
        g.config["results_dir"] = results_dir
        g.workflow = MagicMock()
        g._checkpointer_ctx = None
        g._resolve_pending_entries = MagicMock()
        g._run_graph = MagicMock(return_value=(_fake_state(), "Buy"))

        with patch("tradingagents.graph.trading_graph.get_checkpointer") as mock_cp, \
             patch("tradingagents.graph.trading_graph.checkpoint_step") as mock_step, \
             patch("tradingagents.graph.trading_graph.clear_checkpoint") as mock_clear:
            mock_cp.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cp.return_value.__exit__ = MagicMock(return_value=False)
            mock_step.return_value = 3  # pretend step 3 exists

            TradingAgentsGraph.propagate(g, ticker, date)

            mock_clear.assert_called_once_with(data_dir, ticker, date)

    def test_genuine_crash_resumes_not_cleared(self, tmp_path):
        """A checkpoint without a result JSON must NOT be cleared — the run
        genuinely crashed and should resume."""
        ticker = "NVDA"
        date = "2026-04-20"
        data_dir = str(tmp_path / "cache")
        results_dir = str(tmp_path / "results")

        # No JSON result log on disk — this is a genuine incomplete run.

        g = _mock_graph(tmp_path, checkpoint_enabled=True)
        g.config["data_cache_dir"] = data_dir
        g.config["results_dir"] = results_dir
        g.workflow = MagicMock()
        g._checkpointer_ctx = None
        g._resolve_pending_entries = MagicMock()
        g._run_graph = MagicMock(return_value=(_fake_state(), "Buy"))

        with patch("tradingagents.graph.trading_graph.get_checkpointer") as mock_cp, \
             patch("tradingagents.graph.trading_graph.checkpoint_step") as mock_step, \
             patch("tradingagents.graph.trading_graph.clear_checkpoint") as mock_clear:
            mock_cp.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cp.return_value.__exit__ = MagicMock(return_value=False)
            mock_step.return_value = 2  # genuine in-progress checkpoint

            TradingAgentsGraph.propagate(g, ticker, date)

            # clear_checkpoint must NOT be called during propagate — the
            # checkpoint is valid and the graph should resume from it.
            mock_clear.assert_not_called()

    def test_successful_run_leaves_no_checkpoint(self, tmp_path):
        """After a successful _run_graph with checkpoint_enabled, no
        checkpoint remains for that ticker+date."""
        g = _mock_graph(tmp_path, checkpoint_enabled=True)
        data_dir = g.config["data_cache_dir"]
        ticker = "NVDA"
        date = "2026-04-20"

        # Seed a checkpoint so clear_checkpoint has something to work on
        tid = thread_id(ticker, date)
        with get_checkpointer(data_dir, ticker) as saver:
            builder = StateGraph(_SimpleState)
            builder.add_node("a", _node_a)
            builder.set_entry_point("a")
            builder.add_edge("a", END)
            graph = builder.compile(checkpointer=saver)
            graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid}})

        assert has_checkpoint(data_dir, ticker, date)

        # Run _run_graph — it should clear the checkpoint
        g._run_graph = functools.partial(TradingAgentsGraph._run_graph, g)
        g._run_graph(ticker, date)

        assert not has_checkpoint(data_dir, ticker, date)

    def test_no_duplicate_memory_entries_across_crash_resume_cycle(self, tmp_path):
        """Full crash-resume cycle must not produce duplicate memory log entries.

        Scenario: run completes → decision stored → checkpoint cleared.
        Next run for same ticker+date (user intentionally re-runs): resolved
        entry is replaced, not duplicated.
        """
        g = _mock_graph(tmp_path, checkpoint_enabled=False)
        g._run_graph = functools.partial(TradingAgentsGraph._run_graph, g)

        # First run
        g._run_graph("NVDA", "2026-04-20")
        entries = g.memory_log.load_entries()
        assert len(entries) == 1
        assert entries[0]["pending"] is True

        # Resolve the pending entry (simulating _resolve_pending_entries on next run)
        g.memory_log.update_with_outcome(
            "NVDA", "2026-04-20", 0.05, 0.02, 5, "Correct."
        )
        entries = g.memory_log.load_entries()
        assert len(entries) == 1
        assert entries[0]["pending"] is False

        # Second run — same ticker+date
        g._run_graph("NVDA", "2026-04-20")
        entries = g.memory_log.load_entries()
        assert len(entries) == 1, (
            f"expected exactly 1 entry after re-run, got {len(entries)}"
        )
        assert entries[0]["pending"] is True
