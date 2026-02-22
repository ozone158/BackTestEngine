"""
Tests for reference data helpers (Step 1.1.2: application logic for symbol_a < symbol_b).
"""

import pytest

from src.data.storage.reference import normalize_pair, pair_id


def test_normalize_pair_order():
    """Lexicographic order: (symbol_a, symbol_b) with symbol_a < symbol_b."""
    assert normalize_pair("AAPL", "MSFT") == ("AAPL", "MSFT")
    assert normalize_pair("MSFT", "AAPL") == ("AAPL", "MSFT")
    assert normalize_pair("SPY", "QQQ") == ("QQQ", "SPY")
    assert normalize_pair("qqq", "spy") == ("QQQ", "SPY")  # uppercase


def test_normalize_pair_same_symbol_raises():
    """Same symbol for both legs is invalid."""
    with pytest.raises(ValueError, match="must differ"):
        normalize_pair("AAPL", "AAPL")
    with pytest.raises(ValueError, match="must differ"):
        normalize_pair("SPY", "SPY")


def test_pair_id_stable():
    """pair_id is stable regardless of argument order; format symbol_a_symbol_b."""
    assert pair_id("AAPL", "MSFT") == "AAPL_MSFT"
    assert pair_id("MSFT", "AAPL") == "AAPL_MSFT"
    assert pair_id("SPY", "QQQ") == "QQQ_SPY"
    assert pair_id("  aapl  ", "  msft  ") == "AAPL_MSFT"
