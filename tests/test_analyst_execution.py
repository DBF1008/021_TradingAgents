import unittest

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_keys,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"], concurrency_limit=2)

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.concurrency_limit, 2)
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].tool_node, "tools_news")
        self.assertEqual(plan.specs[0].clear_node, "Msg Clear News")

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "macro"])

    def test_requires_positive_concurrency_limit(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market"], concurrency_limit=0)

    def test_get_initial_analyst_node_uses_plan_metadata(self):
        plan = build_analyst_execution_plan(["fundamentals", "news"])

        self.assertEqual(
            get_initial_analyst_node(plan),
            "Fundamentals Analyst",
        )

    def test_social_key_displays_as_sentiment_analyst(self):
        # The wire key stays "social" for saved-config back-compat, but the
        # user-visible agent_node label must match the v0.2.5 rename so the
        # wall-time summary and any future consumer of agent_node says
        # "Sentiment Analyst" rather than the legacy "Social Analyst".
        plan = build_analyst_execution_plan(["social"])
        spec = plan.specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")


class AnalystWallTimeTrackerTests(unittest.TestCase):
    def test_records_wall_time_when_analyst_completes(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=10.0)
        tracker.mark_completed("market", completed_at=13.5)

        self.assertEqual(tracker.get_wall_times(), {"market": 3.5})

    def test_formats_summary_in_plan_order(self):
        plan = build_analyst_execution_plan(["news", "market"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=20.0)
        tracker.mark_completed("market", completed_at=22.25)
        tracker.mark_started("news", started_at=10.0)
        tracker.mark_completed("news", completed_at=14.0)

        self.assertEqual(
            tracker.format_summary(),
            "Analyst wall time: News 4.00s | Market 2.25s",
        )

    def test_syncs_wall_time_from_sequential_chunks(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(tracker.get_wall_times(), {})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done"},
            now=13.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=18.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 5.0},
        )


class PartitionsTests(unittest.TestCase):
    def test_serial_partitions_one_per_wave(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social", "fundamentals"], concurrency_limit=1
        )
        parts = plan.partitions()
        self.assertEqual(len(parts), 4)
        self.assertEqual([len(p) for p in parts], [1, 1, 1, 1])
        self.assertEqual(
            [parts[i][0].key for i in range(4)],
            ["market", "news", "social", "fundamentals"],
        )

    def test_limit_2_produces_two_waves(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social", "fundamentals"], concurrency_limit=2
        )
        parts = plan.partitions()
        self.assertEqual(len(parts), 2)
        self.assertEqual([s.key for s in parts[0]], ["market", "news"])
        self.assertEqual([s.key for s in parts[1]], ["social", "fundamentals"])

    def test_limit_ge_count_produces_single_wave(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social"], concurrency_limit=5
        )
        parts = plan.partitions()
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(parts[0]), 3)

    def test_limit_3_with_4_analysts_produces_uneven_waves(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social", "fundamentals"], concurrency_limit=3
        )
        parts = plan.partitions()
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0]), 3)
        self.assertEqual(len(parts[1]), 1)
        self.assertEqual(parts[1][0].key, "fundamentals")

    def test_single_analyst_always_single_wave(self):
        plan = build_analyst_execution_plan(["market"], concurrency_limit=4)
        parts = plan.partitions()
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(parts[0]), 1)


class InitialAnalystKeysTests(unittest.TestCase):
    def test_serial_returns_single_key(self):
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=1)
        self.assertEqual(get_initial_analyst_keys(plan), ["market"])

    def test_concurrent_returns_first_wave_keys(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social"], concurrency_limit=2
        )
        self.assertEqual(get_initial_analyst_keys(plan), ["market", "news"])

    def test_full_concurrency_returns_all_keys(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social", "fundamentals"], concurrency_limit=4
        )
        self.assertEqual(
            get_initial_analyst_keys(plan),
            ["market", "news", "social", "fundamentals"],
        )


class ConcurrentWallTimeTrackerTests(unittest.TestCase):
    def test_concurrent_marks_both_started_when_wave_active(self):
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertIn("market", tracker._started_at)
        self.assertIn("news", tracker._started_at)
        self.assertEqual(tracker.get_wall_times(), {})

    def test_concurrent_partial_completion(self):
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=13.0
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})
        self.assertNotIn("news", tracker.get_wall_times())

    def test_concurrent_all_complete_then_next_wave(self):
        plan = build_analyst_execution_plan(
            ["market", "news", "social", "fundamentals"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=15.0,
        )
        self.assertIn("social", tracker._started_at)
        self.assertIn("fundamentals", tracker._started_at)
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 5.0, "news": 5.0},
        )

    def test_serial_behavior_unchanged(self):
        """concurrency_limit=1 must behave identically to old code."""
        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=1)
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertIn("market", tracker._started_at)
        self.assertNotIn("news", tracker._started_at)

        sync_analyst_tracker_from_chunk(
            tracker, {"market_report": "done"}, now=13.0
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})
        self.assertIn("news", tracker._started_at)

    def test_format_summary_with_concurrent_completion(self):
        plan = build_analyst_execution_plan(
            ["market", "news"], concurrency_limit=2
        )
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=10.0)
        tracker.mark_started("news", started_at=10.0)
        tracker.mark_completed("news", completed_at=12.0)
        tracker.mark_completed("market", completed_at=14.0)

        self.assertEqual(
            tracker.format_summary(),
            "Analyst wall time: Market 4.00s | News 2.00s",
        )


class ConcurrentAnalystStatusTests(unittest.TestCase):
    """Test update_analyst_statuses with partition-aware concurrency."""

    def _make_buffer(self, selected):
        from cli.main import ANALYST_AGENT_NAMES, ANALYST_REPORT_MAP

        buf = type("MockBuffer", (), {
            "selected_analysts": set(selected),
            "report_sections": {ANALYST_REPORT_MAP[k]: None for k in selected},
            "agent_status": {},
        })()
        for k in selected:
            buf.agent_status[ANALYST_AGENT_NAMES[k]] = "pending"
        buf.agent_status["Bull Researcher"] = "pending"
        buf.update_agent_status = lambda name, status: buf.agent_status.__setitem__(name, status)
        buf.update_report_section = lambda key, val: buf.report_sections.__setitem__(key, val)
        return buf

    def test_concurrent_both_in_progress(self):
        from cli.main import update_analyst_statuses

        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        buf = self._make_buffer(["market", "news"])
        update_analyst_statuses(buf, {}, plan=plan)
        self.assertEqual(buf.agent_status["Market Analyst"], "in_progress")
        self.assertEqual(buf.agent_status["News Analyst"], "in_progress")

    def test_concurrent_one_complete_other_in_progress(self):
        from cli.main import update_analyst_statuses

        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        buf = self._make_buffer(["market", "news"])
        update_analyst_statuses(buf, {"market_report": "done"}, plan=plan)
        self.assertEqual(buf.agent_status["Market Analyst"], "completed")
        self.assertEqual(buf.agent_status["News Analyst"], "in_progress")

    def test_concurrent_all_complete_triggers_bull(self):
        from cli.main import update_analyst_statuses

        plan = build_analyst_execution_plan(["market", "news"], concurrency_limit=2)
        buf = self._make_buffer(["market", "news"])
        update_analyst_statuses(
            buf, {"market_report": "done", "news_report": "done"}, plan=plan
        )
        self.assertEqual(buf.agent_status["Market Analyst"], "completed")
        self.assertEqual(buf.agent_status["News Analyst"], "completed")
        self.assertEqual(buf.agent_status["Bull Researcher"], "in_progress")

    def test_two_waves_second_pending_until_first_done(self):
        from cli.main import update_analyst_statuses

        plan = build_analyst_execution_plan(
            ["market", "news", "social", "fundamentals"], concurrency_limit=2
        )
        buf = self._make_buffer(["market", "news", "social", "fundamentals"])
        update_analyst_statuses(buf, {}, plan=plan)
        self.assertEqual(buf.agent_status["Market Analyst"], "in_progress")
        self.assertEqual(buf.agent_status["News Analyst"], "in_progress")
        self.assertEqual(buf.agent_status["Sentiment Analyst"], "pending")
        self.assertEqual(buf.agent_status["Fundamentals Analyst"], "pending")

    def test_serial_plan_none_fallback(self):
        from cli.main import update_analyst_statuses

        buf = self._make_buffer(["market", "news"])
        update_analyst_statuses(buf, {}, plan=None)
        self.assertEqual(buf.agent_status["Market Analyst"], "in_progress")
        self.assertEqual(buf.agent_status["News Analyst"], "pending")


class GraphWiringTests(unittest.TestCase):
    """Verify setup_graph produces correct edge topology for various concurrency limits."""

    def _get_edges(self, selected, concurrency_limit):
        from unittest.mock import MagicMock, patch
        from langgraph.graph import START

        mock_llm = MagicMock()
        mock_tool_nodes = {
            "market": MagicMock(),
            "social": MagicMock(),
            "news": MagicMock(),
            "fundamentals": MagicMock(),
        }

        with patch("tradingagents.graph.setup.create_market_analyst", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_sentiment_analyst", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_news_analyst", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_fundamentals_analyst", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_bull_researcher", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_bear_researcher", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_research_manager", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_trader", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_aggressive_debator", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_neutral_debator", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_conservative_debator", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_portfolio_manager", return_value=lambda s: s), \
             patch("tradingagents.graph.setup.create_msg_delete", return_value=lambda s: {}):

            from tradingagents.graph.setup import GraphSetup
            from tradingagents.graph.conditional_logic import ConditionalLogic

            gs = GraphSetup(
                mock_llm, mock_llm, mock_tool_nodes,
                ConditionalLogic(),
                analyst_concurrency_limit=concurrency_limit,
            )
            workflow = gs.setup_graph(selected)

        # Extract simple edges (non-conditional)
        edges = set()
        for src, dst in workflow._edges.items():
            edges.add((src, dst))
        return edges, START

    def test_serial_limit_1_produces_chain(self):
        edges, START = self._get_edges(["market", "news"], 1)
        self.assertIn((START, "Market Analyst"), edges)
        self.assertIn(("Msg Clear Market", "News Analyst"), edges)
        self.assertIn(("Msg Clear News", "Bull Researcher"), edges)
        # No fan-out from START to News
        self.assertNotIn((START, "News Analyst"), edges)

    def test_limit_2_fans_out_from_start(self):
        edges, START = self._get_edges(["market", "news"], 2)
        self.assertIn((START, "Market Analyst"), edges)
        self.assertIn((START, "News Analyst"), edges)
        self.assertIn(("Msg Clear Market", "Bull Researcher"), edges)
        self.assertIn(("Msg Clear News", "Bull Researcher"), edges)

    def test_limit_2_with_4_analysts_cross_joins(self):
        edges, START = self._get_edges(
            ["market", "news", "social", "fundamentals"], 2
        )
        # Wave 0 fan-out
        self.assertIn((START, "Market Analyst"), edges)
        self.assertIn((START, "News Analyst"), edges)
        # Cross-join between wave 0 and wave 1
        self.assertIn(("Msg Clear Market", "Sentiment Analyst"), edges)
        self.assertIn(("Msg Clear Market", "Fundamentals Analyst"), edges)
        self.assertIn(("Msg Clear News", "Sentiment Analyst"), edges)
        self.assertIn(("Msg Clear News", "Fundamentals Analyst"), edges)
        # Wave 1 fan-in
        self.assertIn(("Msg Clear Sentiment", "Bull Researcher"), edges)
        self.assertIn(("Msg Clear Fundamentals", "Bull Researcher"), edges)

    def test_single_analyst_any_limit(self):
        edges, START = self._get_edges(["market"], 4)
        self.assertIn((START, "Market Analyst"), edges)
        self.assertIn(("Msg Clear Market", "Bull Researcher"), edges)
