"""Signal engine — the single best-signal entry point for the terminal / MCP.

Everything upstream produces *pieces*: three base models each emit a probability
(`resolve` / `flow` / `smart`), `ensemble.blend` folds those into one calibrated
`EnsembleSignal`, and `backtest.run_backtest` scores how a signal has actually
traded. This module is the one callable that ties those pieces into a **tradeable
decision**: a blended probability, its edge against the live market price, a
BUY_YES / BUY_NO / HOLD call gated by a minimum edge, and a `confidence` that
fuses the ensemble's own trust with the signal's *backtested* reliability when a
price/outcome history is supplied.

It reuses the merged modules verbatim — `ensemble.blend` / `StackWeights` for the
combine, `backtest.run_backtest` for the empirical confidence — and adds no new
modelling of its own. Pure stdlib beyond those two imports (no numpy / pandas /
sklearn / mlx), mirroring their shape: a result dataclass (`Signal`), a couple of
public entry points (`best_signal`, `rank_signals`), and a `__main__` selfcheck.

Run:  python ml/signal_engine.py     (selfcheck; exits non-zero on any failure)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

# Resolve the merged sibling modules regardless of the caller's cwd (the e2e
# recipe runs `ml/signal_engine.py` from the repo root, tests run from `ml/`).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ensemble  # noqa: E402  (merged meta-combiner — do not reimplement)
import backtest  # noqa: E402  (merged signal-quality harness — do not reimplement)

from ensemble import EnsembleSignal, Pred, StackWeights, blend  # noqa: E402
from backtest import BacktestResult, Record, run_backtest  # noqa: E402


# ── tunables ──────────────────────────────────────────────────────────────────

# Minimum |edge| (model prob − market price, in probability units) needed to act.
# Below this band the quote is not worth crossing the fee/slippage — we HOLD.
MIN_EDGE: float = 0.02
# Per-trade fee + slippage handed to the backtester, same units as MIN_EDGE.
FEE: float = 0.01
# How hard a positive/negative realized Sharpe pushes empirical reliability up/down.
_SHARPE_SCALE: float = 1.0
# Weight of the *backtested* reliability in the fused confidence (the rest is the
# ensemble's own agreement-gated confidence). 0 → ignore history, 1 → trust only it.
_RELIABILITY_WEIGHT: float = 0.5
# History this long (and splittable) earns an honest no-lookahead threshold sweep;
# shorter histories are scored in-sample at the engine's own MIN_EDGE.
_SWEEP_MIN_RECORDS: int = 40
_SWEEP_TRAIN_FRAC: float = 0.6

# Direction labels.
BUY_YES: str = "BUY_YES"
BUY_NO: str = "BUY_NO"
HOLD: str = "HOLD"


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """A tradeable decision for one market.

    prob                 blended YES probability in 0..1 (from `ensemble.blend`)
    market_price         the live YES quote it was compared against
    edge                 prob − market_price (positive → YES looks underpriced)
    direction            BUY_YES / BUY_NO / HOLD, gated by `min_edge`
    confidence           trust in 0..1 — ensemble confidence fused with backtested
                         reliability when history was supplied, else ensemble alone
    agreement            base-model concurrence in 0..1 (from the ensemble)
    ensemble_confidence  the ensemble's own agreement-gated confidence in 0..1
    reliability          empirical 0..1 reliability from `backtest`, or None if no
                         (usable) history was provided
    score                risk-adjusted edge = |edge| × confidence for a live call,
                         0 for HOLD — the ranking key in `rank_signals`
    contributing         base models that actually fed this signal
    market_id            optional caller-supplied identifier (used by ranking)
    """

    prob: float
    market_price: float
    edge: float
    direction: str
    confidence: float
    agreement: float
    ensemble_confidence: float
    reliability: Optional[float] = None
    score: float = 0.0
    contributing: List[str] = field(default_factory=list)
    market_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "market_id": self.market_id,
            "prob": round(self.prob, 6),
            "market_price": round(self.market_price, 6),
            "edge": round(self.edge, 6),
            "direction": self.direction,
            "confidence": round(self.confidence, 6),
            "agreement": round(self.agreement, 6),
            "ensemble_confidence": round(self.ensemble_confidence, 6),
            "reliability": (None if self.reliability is None
                            else round(self.reliability, 6)),
            "score": round(self.score, 6),
            "contributing": list(self.contributing),
        }


# ── small helpers (stdlib only) ────────────────────────────────────────────────

def _clip01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _reliability_from_backtest(bt: BacktestResult) -> Optional[float]:
    """Distil a backtest into a single 0..1 empirical-reliability score.

    Blends how often the signal's directional bets actually resolved in its favour
    (`hit_rate`, ~0.5 for a coin-flip) with a logistic squash of realized Sharpe
    (so a money-losing signal — negative Sharpe — is dragged below neutral). None
    when the signal took no trades on the history: nothing empirical to say.
    """
    if bt.n_trades <= 0:
        return None
    hit = _clip01(bt.hit_rate)
    sharpe_score = _sigmoid(_SHARPE_SCALE * bt.sharpe)   # 0.5 at Sharpe 0
    return _clip01(0.5 * hit + 0.5 * sharpe_score)


def _fuse_confidence(ensemble_conf: float, reliability: Optional[float]) -> float:
    """Fuse the ensemble's confidence with backtested reliability.

    No history → the ensemble's confidence stands alone. Otherwise a weighted
    geometric mean: both must be healthy to earn a high fused confidence, and
    either being weak (model disagreement *or* poor realized track record) pulls
    it down — a strictly more demanding bar than either signal alone.
    """
    ec = _clip01(ensemble_conf)
    if reliability is None:
        return ec
    r = _clip01(reliability)
    a = _RELIABILITY_WEIGHT
    return _clip01((ec ** (1.0 - a)) * (r ** a))


def _backtest_reliability(
    history: Sequence[Record], min_edge: float, fee: float
) -> Optional[float]:
    """Empirical reliability of trading this signal, from `backtest.run_backtest`.

    Scores trading at the engine's own `min_edge` — the threshold `best_signal`
    actually acts on — so reliability measures the very policy being run. On a long
    enough history it does so out-of-time (later time slice only, via `run_backtest`'s
    train/test split at a single pinned threshold — no lookahead, and no risk of the
    multi-threshold sweep degenerating onto a zero-trade cutoff); on a short history
    it scores in-sample. Malformed/empty history degrades to None rather than
    raising, so `best_signal` stays robust to junk histories.
    """
    if not history:
        return None
    try:
        if len(history) >= _SWEEP_MIN_RECORDS:
            bt = run_backtest(history, fee=fee, train_frac=_SWEEP_TRAIN_FRAC,
                              thresholds=[min_edge])
        else:
            bt = run_backtest(history, threshold=min_edge, fee=fee)
    except ValueError:
        return None
    return _reliability_from_backtest(bt)


# ── the entry point ────────────────────────────────────────────────────────────

def best_signal(
    model_preds: Mapping[str, Pred],
    market_price: float,
    history: Optional[Sequence[Record]] = None,
    *,
    weights: Optional[StackWeights] = None,
    min_edge: float = MIN_EDGE,
    fee: float = FEE,
    market_id: Optional[str] = None,
) -> Signal:
    """Turn per-model predictions into one tradeable `Signal`.

    `model_preds` maps base-model name → prediction and is blended verbatim by
    `ensemble.blend` (a fitted `StackWeights` may be passed through). The blend's
    probability is compared against `market_price` to get an `edge`; the direction
    is BUY_YES when the edge clears `+min_edge`, BUY_NO when it clears `−min_edge`,
    HOLD otherwise (and always HOLD when no base model contributed). `confidence`
    is the ensemble's agreement-gated confidence, fused with the signal's
    backtested reliability when `history` (records of
    ``(pred, price, outcome, ts)``) is supplied.
    """
    price = _clip01(float(market_price))
    es: EnsembleSignal = blend(model_preds, weights)

    edge = es.prob - price
    reliability = _backtest_reliability(history or [], min_edge, fee)
    confidence = _fuse_confidence(es.confidence, reliability)

    # Direction: gated by the min-edge band, and never act without a base model.
    if not es.contributing:
        direction = HOLD
    elif edge > min_edge:
        direction = BUY_YES
    elif edge < -min_edge:
        direction = BUY_NO
    else:
        direction = HOLD

    # Risk-adjusted edge: only a live (non-HOLD) call is an opportunity to rank.
    score = 0.0 if direction == HOLD else abs(edge) * confidence

    return Signal(
        prob=es.prob,
        market_price=price,
        edge=edge,
        direction=direction,
        confidence=confidence,
        agreement=es.agreement,
        ensemble_confidence=es.confidence,
        reliability=reliability,
        score=score,
        contributing=list(es.contributing),
        market_id=market_id,
    )


def _coerce_signal(market: object) -> Signal:
    """Accept either a ready `Signal` or a market spec dict and return a `Signal`.

    A dict carries `model_preds` and `market_price` (required), plus optional
    `history`, `weights`, `min_edge`, `fee`, and an id under any of
    `market_id` / `id` / `name` / `slug`.
    """
    if isinstance(market, Signal):
        return market
    if not isinstance(market, Mapping):
        raise TypeError(f"rank_signals: expected Signal or mapping, got {type(market).__name__}")
    if "model_preds" not in market or "market_price" not in market:
        raise KeyError("rank_signals: market dict needs 'model_preds' and 'market_price'")
    # First key that is actually present (not the first *truthy* one) — an id of
    # 0 or "" is a valid identifier and must not be skipped by an `or` chain.
    mid = None
    for key in ("market_id", "id", "name", "slug"):
        if market.get(key) is not None:
            mid = market[key]
            break
    return best_signal(
        market["model_preds"],
        market["market_price"],
        market.get("history"),
        weights=market.get("weights"),
        min_edge=market.get("min_edge", MIN_EDGE),
        fee=market.get("fee", FEE),
        market_id=(str(mid) if mid is not None else None),
    )


def rank_signals(list_of_markets: Sequence[object]) -> List[Signal]:
    """Best opportunities first: build a `Signal` per market and sort by
    risk-adjusted edge (|edge| × confidence), descending.

    Each element is a `Signal` (used as-is) or a market spec dict (see
    `_coerce_signal`). HOLD calls score 0 and sink to the bottom; among live
    calls a bigger, better-trusted edge ranks higher.
    """
    signals = [_coerce_signal(m) for m in list_of_markets]
    return sorted(signals, key=lambda s: s.score, reverse=True)


# ── selfcheck ─────────────────────────────────────────────────────────────────

def _informed_history(n: int, seed: int, edge: float = 0.15) -> List[Record]:
    """Synthetic history where the model is genuinely informed: its probability
    sits `edge` on the correct side of the market price and outcomes follow the
    model. Trading it should be reliably profitable (hit-rate and Sharpe > flat)."""
    import random

    rng = random.Random(seed)
    out: List[Record] = []
    for t in range(n):
        price = rng.uniform(0.15, 0.85)
        # True prob leans away from price in a consistent direction the model sees.
        direction = 1.0 if rng.random() < 0.5 else -1.0
        true = min(0.95, max(0.05, price + direction * edge))
        prob = min(0.99, max(0.01, true + rng.gauss(0.0, 0.02)))
        outcome = 1.0 if rng.random() < true else 0.0
        out.append((prob, price, outcome, float(t)))
    return out


def _selfcheck() -> None:
    # 1. Strong agreement + big edge, backed by an informed history → BUY_YES with
    #    a genuinely high, empirically-boosted confidence and the right edge sign.
    hist = _informed_history(400, seed=7)
    strong = best_signal(
        {"resolve": 0.90, "flow": 0.88, "smart": 0.92},
        market_price=0.60,
        history=hist,
        market_id="strong-yes",
    )
    assert strong.direction == BUY_YES, strong.direction
    assert strong.edge > 0, strong.edge          # prob > price → positive edge
    assert 0.0 <= strong.confidence <= 1.0, strong.confidence
    assert strong.confidence > 0.5, strong.confidence
    assert strong.reliability is not None and strong.reliability > 0.5, strong.reliability
    assert strong.contributing == ensemble.MODELS, strong.contributing

    # Symmetric BUY_NO: unanimous low prob well under the price → sell YES.
    strong_no = best_signal(
        {"resolve": 0.10, "flow": 0.12, "smart": 0.08},
        market_price=0.40,
    )
    assert strong_no.direction == BUY_NO, strong_no.direction
    assert strong_no.edge < 0, strong_no.edge

    # 2. Base models at war → the ensemble crushes confidence → HOLD / low trust,
    #    even though the blended prob may sit far from the price.
    split = best_signal(
        {"resolve": 0.95, "flow": 0.05, "smart": 0.50},
        market_price=0.50,
    )
    assert split.agreement < 0.3, split.agreement
    assert split.confidence < 0.3, split.confidence
    assert split.confidence < strong.confidence, (split.confidence, strong.confidence)

    # A small edge inside the min-edge band → HOLD regardless of agreement.
    tiny = best_signal(
        {"resolve": 0.515, "flow": 0.515, "smart": 0.515},
        market_price=0.505,
    )
    assert abs(tiny.edge) < MIN_EDGE and tiny.direction == HOLD, (tiny.edge, tiny.direction)

    # No usable base models → explicit HOLD, zero confidence, never a crash.
    empty = best_signal({}, market_price=0.30)
    assert empty.direction == HOLD and empty.confidence == 0.0, empty

    # 3. Missing-history path: confidence is exactly the ensemble's own (no fusion).
    no_hist = best_signal(
        {"resolve": 0.90, "flow": 0.88, "smart": 0.92},
        market_price=0.60,
    )
    assert no_hist.reliability is None, no_hist.reliability
    assert abs(no_hist.confidence - no_hist.ensemble_confidence) < 1e-12, (
        no_hist.confidence, no_hist.ensemble_confidence)

    # 4. Ranking orders by edge × confidence, best first; HOLDs sink to the bottom.
    markets = [
        {"id": "weak-edge",
         "model_preds": {"resolve": 0.60, "flow": 0.58, "smart": 0.62},
         "market_price": 0.55},                       # small positive edge
        {"id": "strong-edge",
         "model_preds": {"resolve": 0.92, "flow": 0.90, "smart": 0.94},
         "market_price": 0.55},                       # big edge, high agreement
        {"id": "hold",
         "model_preds": {"resolve": 0.52, "flow": 0.52, "smart": 0.52},
         "market_price": 0.515},                      # sub-threshold → HOLD
    ]
    ranked = rank_signals(markets)
    ids = [s.market_id for s in ranked]
    assert ids[0] == "strong-edge", ids
    assert ids[-1] == "hold", ids
    assert ranked[-1].direction == HOLD and ranked[-1].score == 0.0
    # Monotone non-increasing risk-adjusted score.
    for i in range(1, len(ranked)):
        assert ranked[i - 1].score >= ranked[i].score - 1e-12, [s.score for s in ranked]
    # The ordering really is edge×confidence, not edge alone.
    top, mid = ranked[0], ranked[1]
    assert top.score == abs(top.edge) * top.confidence
    assert top.score > mid.score, (top.score, mid.score)

    # 5. Every confidence produced is a valid probability.
    for s in [strong, strong_no, split, tiny, empty, no_hist, *ranked]:
        assert 0.0 <= s.confidence <= 1.0, s
        assert 0.0 <= s.prob <= 1.0, s

    print(
        "signal_engine selfcheck ok "
        f"(strong={strong.direction}@conf={strong.confidence:.3f}"
        f"/rel={strong.reliability:.3f}, split={split.direction}@conf={split.confidence:.3f}, "
        f"ranked={ids})"
    )


if __name__ == "__main__":
    _selfcheck()
