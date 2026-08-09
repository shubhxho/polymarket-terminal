"""Tests for the model leaderboard / auto-eval harness (``evaluate_all``).

Pure stdlib + the module under test (which imports the merged ``backtest``).
Run:  python ml/test_evaluate_all.py     (exits non-zero on the first failure)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import evaluate_all as ev


# ── tiny test harness (mirrors test_ml.py style) ──────────────────────────────

_PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}".rstrip())
    _PASS += 1
    print(f"ok  {name}")


def _suite(n: int = 5000):
    return {
        "strong": ev._synth_holdout(n, seed=7, noise=0.03),
        "mediocre": ev._synth_holdout(n, seed=7, noise=0.14),
        "random": ev._synth_holdout(n, seed=7, noise=None),
    }


# ── classification-metric unit checks ─────────────────────────────────────────

def test_auc_perfect_and_random():
    # Perfectly separable scores → AUC 1.0.
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0.0, 0.0, 1.0, 1.0]
    check("auc_perfect", abs(ev._auc(scores, labels) - 1.0) < 1e-9)
    # Reversed ranking → AUC 0.0.
    check("auc_inverted", abs(ev._auc(scores, [1, 1, 0, 0]) - 0.0) < 1e-9)
    # All-tied scores → 0.5 (no discrimination).
    check("auc_all_tied", abs(ev._auc([0.5] * 4, [0, 1, 0, 1]) - 0.5) < 1e-9)
    # Single class present → undefined → 0.5.
    check("auc_one_class", ev._auc(scores, [1, 1, 1, 1]) == 0.5)


def test_brier_logloss_ece_bounds():
    scores = [0.9, 0.1, 0.8, 0.2]
    labels = [1.0, 0.0, 1.0, 0.0]
    b = ev._brier(scores, labels)
    check("brier_small_when_good", b < 0.05, f"brier={b}")
    # A confident wrong prediction must give a large but finite log-loss.
    ll = ev._log_loss([0.999999], [0.0])
    check("logloss_finite_on_miss", math.isfinite(ll) and ll > 5, f"ll={ll}")
    # Perfectly calibrated 50/50 bucket → ECE 0.
    e = ev._ece([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0], nbins=10)
    check("ece_zero_when_calibrated", abs(e) < 1e-9, f"ece={e}")
    # Badly miscalibrated (says 0.9, resolves 0) → ECE near 0.9.
    e2 = ev._ece([0.9, 0.9], [0.0, 0.0], nbins=10)
    check("ece_high_when_miscalibrated", abs(e2 - 0.9) < 1e-9, f"ece={e2}")


# ── leaderboard ordering & structure ──────────────────────────────────────────

def test_ranking_strong_beats_mediocre_beats_random():
    board = ev.evaluate_all(_suite(), threshold=0.02, fee=0.01)
    order = [m.model for m in board.ranked]
    check("rank_strong_first", order[0] == "strong", str(order))
    check("rank_mediocre_middle", order[1] == "mediocre", str(order))
    check("rank_random_last", order[-1] == "random", str(order))
    # Ranks are a dense 1..n.
    check("ranks_dense", [m.rank for m in board.ranked] == [1, 2, 3])
    # Risk-adjusted return is monotone with model quality.
    s = {m.model: m.sharpe for m in board.ranked}
    check("sharpe_monotone", s["strong"] > s["mediocre"] > s["random"],
          f"sharpe={s}")
    a = {m.model: m.auc for m in board.ranked}
    check("auc_monotone", a["strong"] > a["mediocre"] > a["random"],
          f"auc={a}")
    check("best_property", board.best.model == "strong")


def test_all_metrics_finite():
    board = ev.evaluate_all(_suite(), threshold=0.02, fee=0.01)
    for m in board.ranked:
        for field_name in ("sharpe", "pnl", "mean_return", "max_drawdown",
                            "auc", "brier", "log_loss", "ece", "hit_rate"):
            v = getattr(m, field_name)
            check(f"finite_{m.model}_{field_name}", math.isfinite(v), f"={v}")


def test_tie_break_by_brier():
    # Two models with identical holdouts → identical Sharpe; the Brier tie-break
    # (and then name) must still give a deterministic, total order.
    recs = ev._synth_holdout(2000, seed=3, noise=0.05)
    board = ev.evaluate_all({"b_model": list(recs), "a_model": list(recs)},
                            threshold=0.02, fee=0.01)
    order = [m.model for m in board.ranked]
    check("tie_deterministic_len", len(order) == 2)
    # Equal Sharpe & Brier → falls back to model-name ordering.
    check("tie_name_order", order == ["a_model", "b_model"], str(order))


# ── reported-vs-reproduced disagreement flag ──────────────────────────────────

def test_reported_agreement_and_disagreement():
    recs = ev._synth_holdout(4000, seed=11, noise=0.03)
    with tempfile.TemporaryDirectory() as d:
        # Honest reported file: AUC/Brier close to what will be reproduced.
        board0 = ev.evaluate_all({"m": list(recs)}, data_dir=d,
                                 metrics_files={"m": "m_metrics.json"})
        repro_auc = board0.ranked[0].auc
        repro_brier = board0.ranked[0].brier

        honest = {"auc": round(repro_auc, 4), "brier": round(repro_brier, 4),
                  "logloss": 0.5}
        with open(os.path.join(d, "m_metrics.json"), "w") as fh:
            json.dump(honest, fh)
        b_ok = ev.evaluate_all({"m": list(recs)}, data_dir=d,
                               metrics_files={"m": "m_metrics.json"})
        me = b_ok.ranked[0]
        check("reported_loaded", me.reported_auc is not None
              and me.metrics_file is not None)
        check("reported_agreement_not_flagged", me.disagrees is False,
              f"delta={me.auc_delta}")

        # Dishonest reported file: wildly inflated AUC → must flag.
        liar = {"auc": 0.99, "brier": 0.01}
        with open(os.path.join(d, "m_metrics.json"), "w") as fh:
            json.dump(liar, fh)
        b_bad = ev.evaluate_all({"m": list(recs)}, data_dir=d,
                                metrics_files={"m": "m_metrics.json"})
        mb = b_bad.ranked[0]
        check("reported_disagreement_flagged", mb.disagrees is True,
              f"auc_delta={mb.auc_delta} brier_delta={mb.brier_delta}")
        check("auc_delta_sign", mb.auc_delta is not None and mb.auc_delta < 0,
              f"auc_delta={mb.auc_delta}")


def test_missing_metrics_file_graceful():
    recs = ev._synth_holdout(1500, seed=5, noise=0.05)
    with tempfile.TemporaryDirectory() as d:  # empty dir → no metrics files
        board = ev.evaluate_all({"ghost": list(recs)}, data_dir=d)
        m = board.ranked[0]
    check("missing_metrics_no_file", m.metrics_file is None)
    check("missing_metrics_reported_none",
          m.reported_auc is None and m.reported_brier is None
          and m.reported_logloss is None)
    check("missing_metrics_not_flagged", m.disagrees is False)
    check("missing_metrics_still_evaluated", math.isfinite(m.auc)
          and math.isfinite(m.sharpe))


def test_nested_metrics_extraction():
    # winner/overall_best inside a models{} block should be resolved.
    obj = {
        "overall_best": "ensemble",
        "models": {
            "feature_mlp": {"val_auc": 0.60, "brier": 0.24},
            "ensemble": {"val_auc": 0.61, "brier": 0.231},
        },
    }
    rep = ev._reported_from_obj(obj)
    check("nested_auc", abs(rep["auc"] - 0.61) < 1e-9, str(rep))
    check("nested_brier", abs(rep["brier"] - 0.231) < 1e-9, str(rep))
    # Top-level auc/brier/logloss (resolve-style) also works.
    flat = {"auc": 0.99, "brier": 0.037, "logloss": 0.117}
    rep2 = ev._reported_from_obj(flat)
    check("flat_auc", rep2["auc"] == 0.99)
    check("flat_logloss", rep2["logloss"] == 0.117)


# ── error handling ────────────────────────────────────────────────────────────

def test_errors_on_empty_input():
    try:
        ev.evaluate_all({})
        raised = False
    except ValueError:
        raised = True
    check("empty_models_raises", raised)

    try:
        ev.evaluate_all({"m": []})
        raised = False
    except ValueError:
        raised = True
    check("empty_holdout_raises", raised)


# ── render & serialization ────────────────────────────────────────────────────

def test_render_and_serialize():
    board = ev.evaluate_all(_suite(2000), threshold=0.02, fee=0.01)
    table = ev.render(board)
    check("render_has_header", "sharpe" in table and "auc" in table)
    check("render_lists_models", "strong" in table and "random" in table)
    # Best model appears in the title callout.
    check("render_best_callout", "Best signal" in table)
    d = board.as_dict()
    check("serialize_best", d["best_model"] == "strong")
    check("serialize_count", d["n_models"] == 3)
    check("serialize_rows", len(d["leaderboard"]) == 3)
    # Round-trips through JSON cleanly.
    s = json.dumps(d)
    check("serialize_json_roundtrip", json.loads(s)["best_model"] == "strong")


def test_report_writes_leaderboard_json():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "leaderboard.json")
        board = ev._report(path=path)
        check("report_wrote_file", os.path.isfile(path))
        with open(path) as fh:
            obj = json.load(fh)
        check("report_json_best", obj["best_model"] == "strong")
        check("report_json_key",
              "risk-adjusted" in obj["ranking_key"])
        check("report_returns_board", board.best.model == "strong")


def main():
    tests = [
        test_auc_perfect_and_random,
        test_brier_logloss_ece_bounds,
        test_ranking_strong_beats_mediocre_beats_random,
        test_all_metrics_finite,
        test_tie_break_by_brier,
        test_reported_agreement_and_disagreement,
        test_missing_metrics_file_graceful,
        test_nested_metrics_extraction,
        test_errors_on_empty_input,
        test_render_and_serialize,
        test_report_writes_leaderboard_json,
    ]
    for t in tests:
        t()
    print(f"\n{_PASS} checks passed across {len(tests)} tests")


if __name__ == "__main__":
    main()
