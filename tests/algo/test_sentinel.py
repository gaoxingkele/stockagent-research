"""Sentinel so the Ralph verify gate (pytest tests/algo) is green before any
algorithm task lands. Each PRD task adds its own hermetic test alongside this.
"""


def test_sentinel():
    assert True
