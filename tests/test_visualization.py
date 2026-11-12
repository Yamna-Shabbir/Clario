"""
tests/test_visualization.py
---------------------------
Unit tests for chart and knowledge-graph generation in utils/visualization.py.

Verifies:
  - build_knowledge_graph
  - generate_assignee_bar_chart
  - generate_priority_donut_chart
  - generate_status_pie_chart

Each tested with:
  (a) normal task list
  (b) empty list
  (c) tasks with missing or unusual fields ("Not specified", very long names, unicode)

Also tests concurrent chart generation from multiple threads.
"""

import os
import threading
import pytest

# Set matplotlib backend to Agg BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.visualization import (
    build_knowledge_graph,
    generate_assignee_bar_chart,
    generate_priority_donut_chart,
    generate_status_pie_chart,
)


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------

NORMAL_TASKS = [
    {
        "id": 1,
        "description": "Finalize quarterly budget report.",
        "assigned_to": "Alice",
        "due_date": "2026-11-15",
        "priority": "High",
        "status": "pending",
        "keyword": "finalize",
    },
    {
        "id": 2,
        "description": "Send follow-up email to stakeholders.",
        "assigned_to": "Bob",
        "due_date": "tomorrow",
        "priority": "Medium",
        "status": "completed",
        "keyword": "email",
    },
    {
        "id": 3,
        "description": "Review pull request for auth service.",
        "assigned_to": "Charlie",
        "due_date": "Friday",
        "priority": "Low",
        "status": "pending",
        "keyword": "review",
    },
]

EMPTY_TASKS = []

UNUSUAL_TASKS = [
    {
        # Very long description, very long name, missing id and keyword
        "description": "A" * 300,
        "assigned_to": "Dr. Bartholomew Alexander Montgomery III (Senior Principal Architect)",
        "due_date": "Not specified",
        "priority": "Urgent",  # unusual non-standard priority
        "status": "in_progress",  # unusual status
    },
    {
        # Unicode characters in all fields
        "description": "顧客向けプレゼンテーション資料作成 🚀",
        "assigned_to": "田中 太郎",
        "due_date": "来週月曜日",
        "priority": "High",
        "status": "pending",
        "keyword": "作成",
    },
    {
        # Missing or empty values
        "description": "",
        "assigned_to": "Not specified",
        "due_date": "",
        "priority": "",
        "status": "",
    },
]


# ---------------------------------------------------------------------------
# Tests: build_knowledge_graph
# ---------------------------------------------------------------------------

class TestBuildKnowledgeGraph:

    def test_knowledge_graph_normal(self, tmp_path):
        out = str(tmp_path / "kg_normal.png")
        ret = build_knowledge_graph(NORMAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_knowledge_graph_empty(self, tmp_path):
        out = str(tmp_path / "kg_empty.png")
        ret = build_knowledge_graph(EMPTY_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_knowledge_graph_unusual_and_unicode(self, tmp_path):
        out = str(tmp_path / "kg_unusual.png")
        ret = build_knowledge_graph(UNUSUAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)


# ---------------------------------------------------------------------------
# Tests: generate_assignee_bar_chart
# ---------------------------------------------------------------------------

class TestAssigneeBarChart:

    def test_assignee_bar_chart_normal(self, tmp_path):
        out = str(tmp_path / "bar_normal.png")
        ret = generate_assignee_bar_chart(NORMAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_assignee_bar_chart_empty(self, tmp_path):
        out = str(tmp_path / "bar_empty.png")
        ret = generate_assignee_bar_chart(EMPTY_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_assignee_bar_chart_unusual_and_unicode(self, tmp_path):
        out = str(tmp_path / "bar_unusual.png")
        ret = generate_assignee_bar_chart(UNUSUAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)


# ---------------------------------------------------------------------------
# Tests: generate_priority_donut_chart
# ---------------------------------------------------------------------------

class TestPriorityDonutChart:

    def test_priority_donut_chart_normal(self, tmp_path):
        out = str(tmp_path / "donut_normal.png")
        ret = generate_priority_donut_chart(NORMAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_priority_donut_chart_empty(self, tmp_path):
        out = str(tmp_path / "donut_empty.png")
        ret = generate_priority_donut_chart(EMPTY_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_priority_donut_chart_unusual_and_unicode(self, tmp_path):
        out = str(tmp_path / "donut_unusual.png")
        ret = generate_priority_donut_chart(UNUSUAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)


# ---------------------------------------------------------------------------
# Tests: generate_status_pie_chart
# ---------------------------------------------------------------------------

class TestStatusPieChart:

    def test_status_pie_chart_normal(self, tmp_path):
        out = str(tmp_path / "pie_normal.png")
        ret = generate_status_pie_chart(NORMAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_status_pie_chart_empty(self, tmp_path):
        out = str(tmp_path / "pie_empty.png")
        ret = generate_status_pie_chart(EMPTY_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)

    def test_status_pie_chart_unusual_and_unicode(self, tmp_path):
        out = str(tmp_path / "pie_unusual.png")
        ret = generate_status_pie_chart(UNUSUAL_TASKS, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
        assert ret == os.path.abspath(out)


# ---------------------------------------------------------------------------
# Tests: Concurrent execution from two threads
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=False,
    reason="Matplotlib pyplot maintains global state which may race or conflict under concurrent threads without a threading lock",
)
def test_concurrent_chart_generation(tmp_path):
    """
    Call two chart functions concurrently from separate threads.
    Verifies both output files are generated without unhandled errors.
    """
    out1 = str(tmp_path / "concurrent_bar.png")
    out2 = str(tmp_path / "concurrent_donut.png")
    errors = []

    def worker_bar():
        try:
            generate_assignee_bar_chart(NORMAL_TASKS, output_path=out1)
        except Exception as exc:
            errors.append(exc)

    def worker_donut():
        try:
            generate_priority_donut_chart(NORMAL_TASKS, output_path=out2)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker_bar)
    t2 = threading.Thread(target=worker_donut)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Concurrent chart generation raised errors: {errors}"
    assert os.path.exists(out1) and os.path.getsize(out1) > 0
    assert os.path.exists(out2) and os.path.getsize(out2) > 0

