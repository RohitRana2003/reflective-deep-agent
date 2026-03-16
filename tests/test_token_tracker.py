"""Tests for the token tracking / LLM call logging module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from plan_once.token_tracker import LLMCallRecord, TokenTracker, get_tracker


def _make_response(
    *,
    prompt_eval_count: int = 100,
    eval_count: int = 50,
    tool_calls: list | None = None,
) -> MagicMock:
    """Create a mock AIMessage-like response with token metadata."""
    resp = MagicMock()
    resp.response_metadata = {
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    resp.tool_calls = tool_calls or []
    return resp


class TestTokenTracker:
    """Tests for ``TokenTracker``."""

    def test_record_returns_call_record(self) -> None:
        tracker = TokenTracker()
        rec = tracker.record(
            role="orchestrator",
            response=_make_response(),
            model_name="test-model",
            duration_seconds=1.5,
        )
        assert isinstance(rec, LLMCallRecord)
        assert rec.role == "orchestrator"
        assert rec.model_name == "test-model"
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.total_tokens == 150
        assert rec.duration_seconds == 1.5

    def test_record_extracts_ollama_tokens(self) -> None:
        tracker = TokenTracker()
        rec = tracker.record(
            role="worker",
            response=_make_response(prompt_eval_count=200, eval_count=80),
            model_name="qwen3-coder:480b-cloud",
            duration_seconds=2.0,
            worker_task="Count ERROR lines",
        )
        assert rec.input_tokens == 200
        assert rec.output_tokens == 80
        assert rec.total_tokens == 280
        assert rec.worker_task == "Count ERROR lines"

    def test_record_extracts_openai_style_usage(self) -> None:
        resp = MagicMock()
        resp.response_metadata = {
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 120,
                "total_tokens": 420,
            },
        }
        resp.tool_calls = []
        tracker = TokenTracker()
        rec = tracker.record(role="synthesizer", response=resp, duration_seconds=1.0)
        assert rec.input_tokens == 300
        assert rec.output_tokens == 120
        assert rec.total_tokens == 420

    def test_record_counts_tool_calls(self) -> None:
        tracker = TokenTracker()
        fake_tools = [{"name": "t1"}, {"name": "t2"}, {"name": "t3"}]
        rec = tracker.record(
            role="orchestrator",
            response=_make_response(tool_calls=fake_tools),
            duration_seconds=0.5,
        )
        assert rec.tool_calls_emitted == 3

    def test_record_handles_missing_metadata(self) -> None:
        resp = MagicMock()
        resp.response_metadata = {}
        resp.tool_calls = []
        tracker = TokenTracker()
        rec = tracker.record(role="worker", response=resp, duration_seconds=0.1)
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.total_tokens == 0

    def test_record_handles_none_metadata(self) -> None:
        resp = MagicMock()
        resp.response_metadata = None
        resp.tool_calls = None
        tracker = TokenTracker()
        rec = tracker.record(role="worker", response=resp, duration_seconds=0.1)
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.tool_calls_emitted == 0

    def test_totals_groups_by_role(self) -> None:
        tracker = TokenTracker()
        tracker.record(
            role="orchestrator",
            response=_make_response(prompt_eval_count=100, eval_count=50),
            duration_seconds=1.0,
        )
        tracker.record(
            role="worker",
            response=_make_response(prompt_eval_count=200, eval_count=80),
            duration_seconds=2.0,
            worker_task="task1",
        )
        tracker.record(
            role="worker",
            response=_make_response(prompt_eval_count=150, eval_count=60),
            duration_seconds=1.5,
            worker_task="task2",
        )
        tracker.record(
            role="synthesizer",
            response=_make_response(prompt_eval_count=400, eval_count=100),
            duration_seconds=3.0,
        )

        t = tracker.totals()
        assert t["total_calls"] == 4
        assert t["total_input_tokens"] == 100 + 200 + 150 + 400
        assert t["total_output_tokens"] == 50 + 80 + 60 + 100
        assert t["by_role"]["orchestrator"]["calls"] == 1
        assert t["by_role"]["worker"]["calls"] == 2
        assert t["by_role"]["worker"]["input_tokens"] == 350
        assert t["by_role"]["synthesizer"]["calls"] == 1

    def test_reset_clears_records(self) -> None:
        tracker = TokenTracker()
        tracker.record(role="worker", response=_make_response(), duration_seconds=0.5)
        assert len(tracker.records) == 1
        tracker.reset()
        assert len(tracker.records) == 0
        assert tracker.totals()["total_calls"] == 0

    def test_call_ids_are_sequential(self) -> None:
        tracker = TokenTracker()
        r1 = tracker.record(role="orchestrator", response=_make_response(), duration_seconds=0.1)
        r2 = tracker.record(role="worker", response=_make_response(), duration_seconds=0.2)
        r3 = tracker.record(role="synthesizer", response=_make_response(), duration_seconds=0.3)
        assert r1.call_id == 1
        assert r2.call_id == 2
        assert r3.call_id == 3

    def test_reset_resets_call_counter(self) -> None:
        tracker = TokenTracker()
        tracker.record(role="worker", response=_make_response(), duration_seconds=0.1)
        tracker.reset()
        r = tracker.record(role="worker", response=_make_response(), duration_seconds=0.1)
        assert r.call_id == 1

    def test_summary_returns_string(self) -> None:
        tracker = TokenTracker()
        tracker.record(
            role="orchestrator",
            response=_make_response(),
            model_name="test-model",
            duration_seconds=1.0,
        )
        tracker.record(
            role="worker",
            response=_make_response(),
            model_name="test-model",
            duration_seconds=2.0,
            worker_task="analyze errors",
        )
        tracker.record(
            role="synthesizer",
            response=_make_response(),
            model_name="test-model",
            duration_seconds=1.5,
        )
        s = tracker.summary()
        assert "LLM CALL LOG" in s
        assert "TOKEN USAGE SUMMARY" in s
        assert "TOTAL" in s
        assert "ORCHESTRATOR" in s
        assert "WORKER" in s
        assert "SYNTHESIZER" in s
        assert "3 calls" in s
        assert "analyze errors" in s

    def test_summary_empty_tracker(self) -> None:
        tracker = TokenTracker()
        assert tracker.summary() == "No LLM calls recorded."

    def test_records_returns_copy(self) -> None:
        tracker = TokenTracker()
        tracker.record(role="worker", response=_make_response(), duration_seconds=0.1)
        records = tracker.records
        records.clear()
        assert len(tracker.records) == 1  # original not affected


class TestGetTracker:
    """Tests for the global singleton."""

    def test_returns_same_instance(self) -> None:
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_is_token_tracker(self) -> None:
        assert isinstance(get_tracker(), TokenTracker)
