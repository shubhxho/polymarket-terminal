"""Tests for the signal engine — BUY/HOLD gating, confidence fusion, edge sign,
ranking order, and the missing-history path.

Pure stdlib (`unittest`), no numpy/pandas/sklearn, matching the merged modules'
test style. Runnable either directly (`python ml/test_signal_engine.py`, exits
non-zero on failure) or under any unittest-compatible runner.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble  # noqa: E402
from signal_engine import (  # noqa: E402
    BUY_NO,
    BUY_YES,
    HOLD,
    MIN_EDGE,
    Signal,
    best_signal,
    rank_signals,
)


def _informed_history(n, seed, edge=0.15):
    """Informed synthetic history: the model sits `edge` on the correct side of
    the price and outcomes follow it, so trading it is reliably profitable."""
    import random

    rng = random.Random(seed)
    out = []
    for t in range(n):
        price = rng.uniform(0.15, 0.85)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        true = min(0.95, max(0.05, price + direction * edge))
        prob = min(0.99, max(0.01, true + rng.gauss(0.0, 0.02)))
        outcome = 1.0 if rng.random() < true else 0.0
        out.append((prob, price, outcome, float(t)))
    return out


def _adversarial_history(n, seed):
    """History where the model is systematically wrong — bets consistently on the
    losing side — so its backtested reliability should land below neutral."""
    import random

    rng = random.Random(seed)
    out = []
    for t in range(n):
        price = rng.uniform(0.15, 0.85)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        true = min(0.95, max(0.05, price + direction * 0.15))
        # Model leans the WRONG way relative to the truth.
        prob = min(0.99, max(0.01, price - direction * 0.15))
        outcome = 1.0 if rng.random() < true else 0.0
        out.append((prob, price, outcome, float(t)))
    return out


class TestDirectionGating(unittest.TestCase):
    def test_buy_yes_when_underpriced(self):
        s = best_signal({"resolve": 0.90, "flow": 0.88, "smart": 0.92}, 0.60)
        self.assertEqual(s.direction, BUY_YES)
        self.assertGreater(s.edge, 0.0)                 # prob > price
        self.assertAlmostEqual(s.edge, s.prob - s.market_price, places=12)

    def test_buy_no_when_overpriced(self):
        s = best_signal({"resolve": 0.10, "flow": 0.12, "smart": 0.08}, 0.40)
        self.assertEqual(s.direction, BUY_NO)
        self.assertLess(s.edge, 0.0)                    # prob < price

    def test_hold_inside_min_edge_band(self):
        s = best_signal({"resolve": 0.515, "flow": 0.515, "smart": 0.515}, 0.505)
        self.assertLess(abs(s.edge), MIN_EDGE)
        self.assertEqual(s.direction, HOLD)

    def test_hold_when_no_contributing_models(self):
        # Junk / unknown-only preds → ensemble contributes nothing → HOLD, conf 0,
        # even though prob(0.5) vs a far price is a numerically large edge.
        s = best_signal({"nonsense": 0.9}, 0.05)
        self.assertEqual(s.direction, HOLD)
        self.assertEqual(s.confidence, 0.0)
        self.assertEqual(s.contributing, [])

    def test_edge_sign_matches_direction_across_prices(self):
        preds = {"resolve": 0.70, "flow": 0.72, "smart": 0.68}
        self.assertEqual(best_signal(preds, 0.30).direction, BUY_YES)   # under
        self.assertEqual(best_signal(preds, 0.95).direction, BUY_NO)    # over
        self.assertEqual(best_signal(preds, 0.70).direction, HOLD)      # at fair


class TestConfidenceBounds(unittest.TestCase):
    def test_confidence_in_unit_interval_over_grid(self):
        grid = [0.0, 0.05, 0.3, 0.5, 0.7, 0.95, 1.0]
        for a in grid:
            for b in grid:
                for price in grid:
                    s = best_signal({"resolve": a, "flow": b, "smart": a}, price)
                    self.assertTrue(0.0 <= s.confidence <= 1.0, (a, b, price, s.confidence))
                    self.assertTrue(0.0 <= s.prob <= 1.0, s.prob)
                    if s.reliability is not None:
                        self.assertTrue(0.0 <= s.reliability <= 1.0, s.reliability)

    def test_strong_agreement_high_confidence(self):
        s = best_signal({"resolve": 0.92, "flow": 0.90, "smart": 0.94}, 0.55)
        self.assertGreater(s.agreement, 0.9)
        self.assertGreater(s.confidence, 0.6)
        self.assertEqual(s.direction, BUY_YES)

    def test_disagreement_low_confidence_and_hold(self):
        s = best_signal({"resolve": 0.95, "flow": 0.05, "smart": 0.50}, 0.50)
        self.assertLess(s.agreement, 0.3)
        self.assertLess(s.confidence, 0.3)


class TestHistoryFusion(unittest.TestCase):
    def test_missing_history_uses_ensemble_confidence(self):
        s = best_signal({"resolve": 0.90, "flow": 0.88, "smart": 0.92}, 0.60)
        self.assertIsNone(s.reliability)
        self.assertAlmostEqual(s.confidence, s.ensemble_confidence, places=12)

    def test_empty_history_is_treated_as_missing(self):
        s = best_signal({"resolve": 0.90, "flow": 0.88, "smart": 0.92}, 0.60, history=[])
        self.assertIsNone(s.reliability)
        self.assertAlmostEqual(s.confidence, s.ensemble_confidence, places=12)

    def test_informed_history_gives_reliability_above_neutral(self):
        hist = _informed_history(400, seed=11)
        s = best_signal({"resolve": 0.90, "flow": 0.88, "smart": 0.92}, 0.60, history=hist)
        self.assertIsNotNone(s.reliability)
        self.assertGreater(s.reliability, 0.5)

    def test_adversarial_history_reliability_below_informed(self):
        preds = {"resolve": 0.90, "flow": 0.88, "smart": 0.92}
        good = best_signal(preds, 0.60, history=_informed_history(400, seed=1))
        bad = best_signal(preds, 0.60, history=_adversarial_history(400, seed=1))
        self.assertIsNotNone(bad.reliability)
        self.assertLess(bad.reliability, good.reliability)
        # A worse track record cannot raise confidence above the clean-history call.
        self.assertLessEqual(bad.confidence, good.confidence + 1e-9)

    def test_malformed_history_degrades_to_missing_not_crash(self):
        # Non-finite / wrong-shape records make run_backtest raise; the engine
        # swallows it and falls back to the ensemble confidence.
        junk = [(float("nan"), 0.5, 1.0, 0.0)]
        s = best_signal({"resolve": 0.9, "flow": 0.9, "smart": 0.9}, 0.6, history=junk)
        self.assertIsNone(s.reliability)
        self.assertAlmostEqual(s.confidence, s.ensemble_confidence, places=12)


class TestRanking(unittest.TestCase):
    def _markets(self):
        return [
            {"id": "weak",
             "model_preds": {"resolve": 0.60, "flow": 0.58, "smart": 0.62},
             "market_price": 0.55},
            {"id": "strong",
             "model_preds": {"resolve": 0.92, "flow": 0.90, "smart": 0.94},
             "market_price": 0.55},
            {"id": "hold",
             "model_preds": {"resolve": 0.52, "flow": 0.52, "smart": 0.52},
             "market_price": 0.515},
        ]

    def test_orders_by_risk_adjusted_edge(self):
        ranked = rank_signals(self._markets())
        ids = [s.market_id for s in ranked]
        self.assertEqual(ids[0], "strong")
        self.assertEqual(ids[-1], "hold")
        # Score is exactly |edge| × confidence for live calls, monotone non-increasing.
        for s in ranked:
            expected = 0.0 if s.direction == HOLD else abs(s.edge) * s.confidence
            self.assertAlmostEqual(s.score, expected, places=12)
        for i in range(1, len(ranked)):
            self.assertGreaterEqual(ranked[i - 1].score, ranked[i].score - 1e-12)

    def test_hold_signals_score_zero_and_sink(self):
        ranked = rank_signals(self._markets())
        hold = next(s for s in ranked if s.market_id == "hold")
        self.assertEqual(hold.direction, HOLD)
        self.assertEqual(hold.score, 0.0)
        self.assertEqual(ranked[-1].market_id, "hold")

    def test_accepts_ready_signal_objects(self):
        pre = best_signal({"resolve": 0.9, "flow": 0.9, "smart": 0.9}, 0.5, market_id="pre")
        ranked = rank_signals([pre, *self._markets()])
        self.assertTrue(all(isinstance(s, Signal) for s in ranked))
        self.assertIn("pre", [s.market_id for s in ranked])

    def test_rejects_market_missing_required_keys(self):
        with self.assertRaises(KeyError):
            rank_signals([{"market_price": 0.5}])
        with self.assertRaises(TypeError):
            rank_signals([42])

    def test_empty_input_ranks_to_empty(self):
        self.assertEqual(rank_signals([]), [])

    def test_falsy_but_valid_id_is_preserved(self):
        # An id of 0 (or "") is a real identifier and must survive coercion.
        ranked = rank_signals([
            {"id": 0,
             "model_preds": {"resolve": 0.9, "flow": 0.9, "smart": 0.9},
             "market_price": 0.5},
        ])
        self.assertEqual(ranked[0].market_id, "0")


class TestSelfcheckParity(unittest.TestCase):
    def test_module_selfcheck_runs(self):
        import signal_engine

        signal_engine._selfcheck()   # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
