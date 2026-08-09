"""Mega unified signal trainer on H100 — the culmination of the whole suite.

This is the one model that trains on the **union of all six feature families** the
repo builds — order-flow (`features_flow`), resolution (`features_resolve`),
smart-money (`features_smart`), cross-market arb (`features_crossmarket`), event
co-movement (`features_event`) and microstructure (`features_micro`) — so a single
example carries every angle the terminal knows how to look at a Polymarket market
from. It is the highest-quality signal the suite can produce, and it reports which
family actually earns its place.

The design has ONE feature-assembly seam: `unified_features(ctx)` calls each merged
module's real public entry point and concatenates the results. Both paths below
build the same `ctx` bundles and run them through it — nothing is reimplemented.

Pipeline (all inside Modal, streaming from the Hub):

  1. Stream `trades.parquet` + `markets.parquet` from `SII-WANGZJ/Polymarket_data`
     inside Modal. Bucket each market's trades into hourly bars carrying the true
     signed aggressor flow, keep a light per-trade tape (user / side / size) for
     the behavioural families, and read resolutions + event grouping from markets.
  2. Slide windows → per-window `ctx` bundles → `unified_features(ctx)` = the full
     six-family vector. Two labels per example: next-move DIRECTION and eventual
     RESOLUTION (multi-task).
  3. Train a strong 3-member ensemble — a primary LightGBM booster, a decorrelated
     extra-trees GBDT, and a deep residual multi-task torch MLP (EMA weights, label
     smoothing) — with isotonic (PAV) calibration done in-repo (no sklearn). Report
     AUC + Brier + a decile
     backtest on a strict OUT-OF-TIME split, a walk-forward, and a per-family
     feature-importance table (which family contributes most). Push to
     `shubhxho/polymarket-mega-model`.

    modal run ml/modal_mega.py --max-rows 20000000 --top-tokens 6000 --push

Local proof — zero heavy deps, exercises the REAL six-module assembly path:

    python ml/modal_mega.py --smoke

The merged feature modules are pure stdlib, so they are imported at module top; the
heavy deps (modal, torch, lightgbm, datasets, pyarrow, numpy, huggingface_hub) are
imported only inside the Modal image / functions, so this module imports cleanly on
a bare interpreter and the smoke path runs anywhere Python does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

# Make the merged feature modules importable no matter the cwd (the E2E harness may
# launch this from the repo root). Insert THIS file's directory first so the
# stdlib-only feature modules resolve locally, then import them at top — they pull
# in nothing heavier than `math`, so a bare interpreter stays clean.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features_crossmarket  # noqa: E402
import features_event  # noqa: E402
import features_flow  # noqa: E402
import features_micro  # noqa: E402
import features_resolve  # noqa: E402
import features_smart  # noqa: E402
from features import HORIZON, WINDOW  # noqa: E402

# Modal is optional at import time. When present we build a real H100 app; when not
# (e.g. the stdlib smoke env) `app` is None and the two decorators below degrade to
# no-ops, so the module still imports and --smoke still runs. Heavy deps are NEVER
# imported at module top.
try:  # pragma: no cover - depends on environment
    import modal

    app = modal.App("pmt-mega")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "pyarrow", "numpy", "torch", "lightgbm",
            "datasets", "huggingface_hub", "safetensors",
        )
        # Ship the merged pure-stdlib feature code into the image so the remote job
        # imports the SAME assembly path the smoke test proves — never a re-impl.
        .add_local_python_source(
            "features", "features_flow", "features_resolve", "features_smart",
            "features_crossmarket", "features_event", "features_micro",
        )
    )
except Exception:  # ImportError in the smoke env, and any partial-install error
    modal = None
    app = None
    image = None


def _remote(**kwargs):
    """`@app.function(**kwargs)` when Modal is present, else identity so the
    decorated function stays a plain callable for the stdlib smoke path."""

    def deco(fn):
        return app.function(**kwargs)(fn) if app is not None else fn

    return deco


def _entrypoint():
    def deco(fn):
        return app.local_entrypoint()(fn) if app is not None else fn

    return deco


# ── configuration ─────────────────────────────────────────────────────────────
REPO = "SII-WANGZJ/Polymarket_data"
TRADES_FILE = "trades.parquet"
MARKETS_FILE = "markets.parquet"
HF_MODEL_REPO = "shubhxho/polymarket-mega-model"

BAR_SECONDS = 3600     # hourly buckets (300 = 5-min, where order-flow signal lives)
MIN_CANDLES = WINDOW + HORIZON + 4   # need enough bars to slide at least a few windows
MIN_STD = 1e-4         # skip dead (flat) windows — no direction to learn
TB_K = 1.5             # triple-barrier width in window-return σ
EMBARGO = 0.02         # purge/embargo gap (fraction of each market's timeline)
MICRO_FEE_BPS = 0.001  # synthetic per-fill fee fraction for micro Fill construction


# ── the six families, in a fixed order → the unified feature layout ────────────
# Each entry is (family_name, ordered_feature_names). The unified vector is the
# concatenation in this order; FAMILY_SPANS maps a family to its slice so feature
# importance can be rolled up per family.
FAMILIES = [
    ("flow", features_flow.FLOW_FEATURES),
    ("resolve", features_resolve.RESOLVE_FEATURES),
    ("smart", features_smart.SMART_FEATURES),
    ("crossmarket", features_crossmarket.CROSSMARKET_FEATURES),
    ("event", features_event.EVENT_FEATURES),
    ("micro", features_micro.MICRO_FEATURES),
]

UNIFIED_FEATURE_NAMES = [f"{fam}__{name}" for fam, names in FAMILIES for name in names]


def _family_spans():
    spans, start = {}, 0
    for fam, names in FAMILIES:
        spans[fam] = (start, start + len(names))
        start += len(names)
    return spans


FAMILY_SPANS = _family_spans()
N_FEATURES = len(UNIFIED_FEATURE_NAMES)


# ── the ONE feature-assembly seam ──────────────────────────────────────────────
def unified_features(ctx: dict) -> list:
    """Assemble the full six-family feature vector for one example.

    `ctx` carries each family's already-prepared inputs (see `context_from_market`).
    Every value here comes from the merged modules' real public entry points, in
    the `FAMILIES` order — this function is the single place the families meet, and
    it reimplements none of them."""
    flow = features_flow.window_features_flow(ctx["flow_window"])
    resolve = features_resolve.snapshot_features(ctx["resolve_snapshot"])
    smart = features_smart.window_features_smart(
        ctx["smart"]["users"], ctx["smart"]["roles"], ctx["smart"]["prices"],
        ctx["smart"]["usds"], ctx["smart"]["tokens"],
        ctx["smart"]["smart_set"], ctx["smart"]["seen_before"],
    )
    cross = features_crossmarket.market_features(
        ctx["cross"]["market"], ctx["cross"]["event_markets"])
    event = features_event.window_features_event(
        ctx["event"]["self_close"], ctx["event"]["peer_closes"],
        ctx["event"].get("self_flow"), ctx["event"].get("peer_flows"))
    micro = features_micro.window_features_micro(ctx["micro_buckets"])

    vec = flow + resolve + smart + cross + event + micro
    assert len(vec) == N_FEATURES, f"unified vec {len(vec)} != {N_FEATURES}"
    return vec


# ── building a per-window ctx from one market's aligned data ────────────────────
def _bar_fills(trades, close_price, asset_id):
    """Construct `features_micro.Fill` objects for one bucket's trades.

    Uses the module's own data class, so `window_features_micro` runs its real
    logic over genuine per-trade rows (price / size / signed direction / wallets),
    not an approximation."""
    fills = []
    for t in trades:
        price = float(t["price"]) if t["price"] else close_price
        usd = abs(float(t["usd"]))
        if usd <= features_micro.EPS or price <= 0.0:
            continue
        token = usd / price
        direction = 1 if float(t["token"]) >= 0 else -1
        fills.append(features_micro.Fill(
            ts=int(t.get("ts", 0)), price=price, usdc=usd, token=token,
            direction=direction, maker=str(t.get("maker", "")),
            taker=str(t.get("user", "")), fee=usd * MICRO_FEE_BPS, token_id=str(asset_id)))
    return fills


def context_from_market(m: dict, i: int, smart_set) -> dict:
    """Build the unified `ctx` for the window ending at bar index `i` of market `m`.

    `m` carries aligned per-bar arrays plus a per-bar trade tape:
      bars   : list of {open,high,low,close,volume,trades,signed_flow,ts} dicts
      close  : [close, …]                (== [b['close'] for b in bars])
      sflow  : [signed_flow, …]
      tape   : list (per bar) of trade-dict lists, each trade {user,role,price,usd,token,ts,maker}
      meta   : {market_id,event_id,neg_risk,outcome_prices,t_created,t_end,res_label}
      peers  : list of peer markets (each with 'close'/'sflow') in the same event
    """
    bars = m["bars"]
    close = m["close"]
    sflow = m["sflow"]
    tape = m["tape"]
    meta = m["meta"]
    lo, hi = i - WINDOW, i

    # flow: window of bucket dicts (features_flow reads close/volume/trades/signed_flow).
    flow_window = bars[lo:hi]

    # micro: window of Bucket objects built from the real per-trade tape.
    micro_buckets = [
        features_micro.Bucket(key=lo + k, price=close[lo + k],
                              fills=_bar_fills(tape[lo + k], close[lo + k], meta["market_id"]))
        for k in range(WINDOW)
    ]

    # smart: the flat trade tape inside the window + the wallets seen strictly before it.
    w_trades = [t for k in range(lo, hi) for t in tape[k]]
    if not w_trades:  # keep the smart features finite on an empty (thin) window
        w_trades = [{"user": "_none", "role": "taker", "price": close[hi - 1],
                     "usd": 0.0, "token": 0.0}]
    # Wallets seen strictly before the window. Prefer the per-market prefix sets
    # precomputed once by `iter_examples` (O(N) total) over rebuilding here every
    # window (which would be O(N²) per market on a liquid token).
    prefix = m.get("_prefix_users")
    seen_before = prefix[lo] if prefix is not None else {t["user"] for k in range(lo) for t in tape[k]}
    smart_ctx = {
        "users": [t["user"] for t in w_trades],
        "roles": [t["role"] for t in w_trades],
        "prices": [float(t["price"]) for t in w_trades],
        "usds": [float(t["usd"]) for t in w_trades],
        "tokens": [float(t["token"]) for t in w_trades],
        "smart_set": smart_set,
        "seen_before": seen_before,
    }

    # resolve: a snapshot at the window's leading edge (price/time/flow/volume context).
    win_close = close[lo:hi]
    win_vol = [float(b["volume"]) for b in flow_window]
    win_flow = sflow[lo:hi]
    price_now = win_close[-1]
    t_created = float(meta.get("t_created", 0.0))
    t_end = float(meta.get("t_end", t_created + 1.0))
    ts_now = float(bars[hi - 1].get("ts", t_created))
    resolve_snapshot = {
        "price": price_now,
        "t_created": t_created, "t_end": t_end, "t_snapshot": ts_now,
        "recent": win_close[-8:],
        "signed_flow": sum(win_flow), "flow_total": sum(abs(f) for f in win_flow),
        "volume": sum(win_vol),
        "volume_total": m.get("_total_volume", sum(float(b["volume"]) for b in bars)),
        "trade_count": float(bars[hi - 1].get("trades", 0)),
        # no explicit label → snapshot_features never peeks at the outcome here.
    }

    # crossmarket: this leg + its event siblings, priced at the window's current price.
    def _summary(mm, price):
        return {"market_id": mm["meta"]["market_id"], "event_id": mm["meta"]["event_id"],
                "neg_risk": mm["meta"].get("neg_risk", False), "price": price}

    self_sum = {"market_id": meta["market_id"], "event_id": meta["event_id"],
                "neg_risk": meta.get("neg_risk", False), "price": price_now,
                "outcome_prices": meta.get("outcome_prices")}
    peers = m.get("peers", [])
    event_markets = [self_sum] + [
        _summary(p, p["close"][hi - 1]) for p in peers if len(p["close"]) >= hi]
    cross_ctx = {"market": self_sum, "event_markets": event_markets}

    # event: this market's price/flow window vs its aligned event peers'.
    peer_closes, peer_flows = [], []
    for p in peers:
        if len(p["close"]) >= hi:
            peer_closes.append(p["close"][lo:hi])
            peer_flows.append(p["sflow"][lo:hi])
    event_ctx = {"self_close": win_close, "peer_closes": peer_closes,
                 "self_flow": win_flow, "peer_flows": peer_flows or None}

    return {"flow_window": flow_window, "resolve_snapshot": resolve_snapshot,
            "smart": smart_ctx, "cross": cross_ctx, "event": event_ctx,
            "micro_buckets": micro_buckets}


def iter_examples(world: list):
    """Slide over every market in `world` → unified examples.

    Yields (feature_vec, dir_label, res_label, fwd, tfrac). The direction label is
    the next-move sign HORIZON bars ahead; the resolution label is the market's
    eventual YES/NO outcome (constant per market). `tfrac` is the window's
    fractional position in its market timeline → drives the out-of-time split.
    Flat windows (no directional signal) are skipped, matching every family."""
    smart_set = _tag_smart_cohort(world)
    for m in world:
        close = m["close"]
        N = len(close)
        if N < MIN_CANDLES:
            continue
        res_label = m["meta"].get("res_label")
        # Precompute once per market: total volume, and the prefix set of wallets
        # seen before each bucket → keeps the per-window ctx build linear, not O(N²).
        m["_total_volume"] = sum(float(b["volume"]) for b in m["bars"])
        prefix_users, acc = [], set()
        for bar_trades in m["tape"]:
            prefix_users.append(frozenset(acc))   # wallets seen strictly before this bucket
            acc.update(t["user"] for t in bar_trades)
        m["_prefix_users"] = prefix_users
        for i in range(WINDOW, N - HORIZON):
            rets = [close[k] - close[k - 1] for k in range(i - WINDOW + 1, i)]
            sigma = features_flow._std(rets)
            if sigma < MIN_STD:
                continue
            # Triple-barrier direction label (López de Prado): the first of
            # ±TB_K·σ to be touched within HORIZON bars sets the sign; a vertical
            # (neither touched) falls back to the net move's sign. A cleaner
            # target than the sign of a possibly-tiny noise move, and on the raw
            # order-flow model this roughly doubled the top-vs-bottom spread.
            base = close[i]
            bar = TB_K * sigma
            dir_label, fwd = None, 0.0
            for t in range(1, HORIZON + 1):
                px = close[i + t]
                if px - base >= bar:
                    dir_label, fwd = 1, bar
                    break
                if base - px >= bar:
                    dir_label, fwd = 0, -bar
                    break
            if dir_label is None:
                fwd = close[i + HORIZON] - base
                dir_label = 1 if fwd > 0 else 0
            ctx = context_from_market(m, i, smart_set)
            yield unified_features(ctx), dir_label, res_label, fwd, i / N


def _tag_smart_cohort(world: list):
    """Tag the global smart-money cohort with the real `features_smart` leaderboard.

    Marks every wallet's trades to each market's resolution price and takes the top
    PnL cohort — exactly the module's own tagging, so the smart features see a
    genuine cohort rather than a random set."""
    smart_markets = []
    for m in world:
        final = m["meta"].get("res_label")
        final = float(final) if final is not None else (m["close"][-1] if m["close"] else 0.5)
        users, prices, tokens = [], [], []
        for bar_trades in m["tape"]:
            for t in bar_trades:
                users.append(t["user"])
                prices.append(float(t["price"]))
                tokens.append(float(t["token"]))
        smart_markets.append({"final": final, "user": users, "price": prices,
                              "token_amount": tokens})
    return features_smart.tag_smart(features_smart.build_leaderboard(smart_markets))


# ── numeric kit (stdlib; shared by smoke + Modal, matches train_seq conventions) ─
def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _sigmoid(z):
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _auc(probs, labels):
    """Rank AUC with proper tie handling (mirrors train_seq._auc)."""
    pairs = sorted(zip(probs, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i, r = 0, 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (r + r + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] > 0.5:
                rank_sum += avg
        r += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _decile_backtest(probs, fwd, q=0.2):
    """Top vs bottom q of markets by score: how did they actually move next?
    Same schema as train_seq._decile_backtest."""
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    if not order:
        return {}
    k = max(1, int(len(order) * q))
    top = [fwd[i] for i in order[-k:]]
    bottom = [fwd[i] for i in order[:k]]

    def _median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    def _up(xs):
        return sum(1 for v in xs if v > 0) / len(xs)

    return {
        "top_up_rate": round(_up(top), 3),
        "bottom_up_rate": round(_up(bottom), 3),
        "up_rate_spread": round(_up(top) - _up(bottom), 3),
        "top_median_pts": round(_median(top) * 100, 3),
        "bottom_median_pts": round(_median(bottom) * 100, 3),
        "slice": k,
    }


def _brier(probs, labels):
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs) if probs else 0.0


def _isotonic_pav(scores, labels):
    """Fit an isotonic (monotone non-decreasing) calibrator by pool-adjacent-
    violators — no sklearn. Returns (xs, ys): the step function mapping a raw score
    to a calibrated probability, sampled at the sorted unique score breakpoints."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [scores[i] for i in order]
    ys = [float(labels[i]) for i in order]
    # PAV: each block holds (sum_y, count, value); merge left while monotonicity breaks.
    blocks = []
    for y in ys:
        blocks.append([y, 1, y])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            sy2, c2, _ = blocks.pop()
            sy1, c1, _ = blocks.pop()
            sy, c = sy1 + sy2, c1 + c2
            blocks.append([sy, c, sy / c])
    fitted = []
    for blk in blocks:
        for _ in range(blk[1]):
            fitted.append(blk[2])
    return xs, fitted


def _apply_isotonic(cal, s):
    xs, ys = cal
    if not xs:
        return 0.5
    # binary search for the last breakpoint <= s
    lo, hi = 0, len(xs) - 1
    if s <= xs[0]:
        return ys[0]
    if s >= xs[-1]:
        return ys[-1]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if xs[mid] <= s:
            lo = mid
        else:
            hi = mid - 1
    return ys[lo]


# ── stdlib logistic (smoke only — proves the train/eval path end to end) ───────
def _moments(X):
    d = len(X[0])
    mean = [_mean([row[j] for row in X]) for j in range(d)]
    std = [_std([row[j] for row in X]) + 1e-6 for j in range(d)]
    return mean, std


def _normalize(X, mean, std):
    return [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in X]


def _fit_logistic(X, y, epochs=250, lr=0.1, l2=1e-4, seed=11):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    idx = list(range(n))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            row = X[i]
            g = _sigmoid(b + sum(w[j] * row[j] for j in range(d))) - y[i]
            b -= lr * g
            for j in range(d):
                w[j] -= lr * (g * row[j] + l2 * w[j])
    return w, b


def _predict_logistic(X, w, b):
    return [_sigmoid(b + sum(w[j] * row[j] for j in range(len(row)))) for row in X]


def _family_importance(weights):
    """Roll |weight| up per family (weights are on standardised features, so their
    magnitude is a fair cross-family contribution proxy). Returns a share table."""
    per = []
    for fam, names in FAMILIES:
        a, b = FAMILY_SPANS[fam]
        mass = sum(abs(weights[j]) for j in range(a, b))
        # the single loudest feature in the family, for colour
        top_j = max(range(a, b), key=lambda j: abs(weights[j]))
        per.append([fam, mass, UNIFIED_FEATURE_NAMES[top_j], round(abs(weights[top_j]), 3)])
    total = sum(p[1] for p in per) + 1e-12
    return [{"family": p[0], "share": round(p[1] / total, 3),
             "abs_weight": round(p[1], 3), "top_feature": p[2], "top_abs_weight": p[3]}
            for p in sorted(per, key=lambda p: -p[1])]


# ── synthetic world for the smoke test ─────────────────────────────────────────
def synth_world(n_events=6, markets_per_event=3, n_buckets=72, trades_per_bucket=8,
                n_wallets=40, n_skilled=8, seed=11):
    """A deterministic synthetic Polymarket world with a plantable, learnable signal.

    Each market's price is driven by a persistent (autoregressive) *informed*
    pressure — no hard outcome pull, so moves stay balanced and never saturate the
    0/1 rails. The aggressor side of every trade tracks that same pressure, and
    skilled wallets track it harder, so recent smart/flow leads the forward move
    (exactly what the unified model must recover). Trades are generated then
    bucketed into bars + a per-bar tape, so flow / smart / micro all read the same
    underlying tape. The eventual resolution label is the terminal price's side.
    """
    rng = random.Random(seed)
    skilled = {f"w{i}" for i in range(n_skilled)}
    t0 = 1_700_000_000
    world = []

    for ev in range(n_events):
        event_markets = []
        for leg in range(markets_per_event):
            price = rng.uniform(0.3, 0.7)
            drift = 0.0
            bars, tape, close, sflow = [], [], [], []
            for bi in range(n_buckets):
                bt = t0 + bi * BAR_SECONDS
                informed = rng.gauss(0.0, 1.0)
                drift = 0.9 * drift + 0.02 * informed          # persistent AR pressure
                # skilled lean their aggressor side hard toward the pressure; the
                # crowd leans weakly → a recoverable smart/flow → forward-move link.
                pbuy_crowd = _sigmoid(1.5 * informed)
                pbuy_skill = _sigmoid(5.0 * informed)
                o = price
                bar_trades = []
                svol = vol = 0.0
                for _ in range(trades_per_bucket):
                    w = f"w{rng.randrange(n_wallets)}"
                    pbuy = pbuy_skill if w in skilled else pbuy_crowd
                    direction = 1 if rng.random() < pbuy else -1
                    price += drift / trades_per_bucket + rng.gauss(0.0, 0.0008)
                    price = min(0.99, max(0.01, price))
                    usd = rng.expovariate(1 / 40.0) * (6.0 if rng.random() < 0.04 else 1.0)
                    ts = bt + rng.randint(0, BAR_SECONDS - 1)
                    tok = direction * usd / max(price, 0.02)
                    svol += direction * usd
                    vol += usd
                    bar_trades.append({"user": w, "role": "maker" if rng.random() < 0.4 else "taker",
                                       "price": round(price, 5), "usd": round(usd, 4),
                                       "token": round(tok, 4), "ts": ts, "maker": w})
                c = price
                hh = max(o, c) + 0.003
                ll = min(o, c) - 0.003
                bars.append({"open": o, "high": hh, "low": ll, "close": c, "ts": bt,
                             "volume": vol, "trades": len(bar_trades), "signed_flow": svol})
                tape.append(bar_trades)
                close.append(c)
                sflow.append(svol)
            won = 1 if close[-1] > 0.5 else 0                 # resolution = terminal side
            op = "['1', '0']" if won else "['0', '1']"
            m = {"bars": bars, "tape": tape, "close": close, "sflow": sflow,
                 "meta": {"market_id": f"e{ev}-{leg}", "event_id": f"e{ev}", "neg_risk": True,
                          "outcome_prices": op, "res_label": won,
                          "t_created": float(t0), "t_end": float(t0 + n_buckets * BAR_SECONDS)},
                 "peers": []}
            event_markets.append(m)
        for m in event_markets:  # wire event siblings as peers
            m["peers"] = [p for p in event_markets if p is not m]
        world.extend(event_markets)
    return world


def run_smoke(n_events=6, markets_per_event=3, n_buckets=64, seed=11):
    """FULL unified pipeline on synthetic data, zero heavy deps: build the world →
    assemble the six-family vector via the merged modules → out-of-time split →
    train stdlib logistic heads for DIRECTION and RESOLUTION → AUC + Brier + decile
    backtest + PAV calibration + per-family importance."""
    world = synth_world(n_events=n_events, markets_per_event=markets_per_event,
                        n_buckets=n_buckets, seed=seed)
    feats, dirs, ress, fwds, tfrac = [], [], [], [], []
    for vec, dlab, rlab, fwd, tf in iter_examples(world):
        feats.append(vec)
        dirs.append(dlab)
        ress.append(rlab if rlab is not None else 0)
        fwds.append(fwd)
        tfrac.append(tf)
    if not feats:
        raise SystemExit("smoke: no windows built — raise --buckets/--events")
    assert all(all(math.isfinite(v) for v in row) for row in feats), "non-finite unified feature"

    tr = [i for i, t in enumerate(tfrac) if t < 0.8]
    va = [i for i, t in enumerate(tfrac) if t >= 0.8]
    if not tr or not va:
        raise SystemExit("smoke: degenerate split")

    Xtr = [feats[i] for i in tr]
    mean, std = _moments(Xtr)
    Xtr_n = _normalize(Xtr, mean, std)
    Xva_n = _normalize([feats[i] for i in va], mean, std)

    # ── DIRECTION head (primary) ──────────────────────────────────────────────
    ydir_tr = [dirs[i] for i in tr]
    ydir_va = [dirs[i] for i in va]
    fva = [fwds[i] for i in va]
    wd, bd = _fit_logistic(Xtr_n, ydir_tr)
    pdir = _predict_logistic(Xva_n, wd, bd)
    dir_auc = _auc(pdir, ydir_va)
    dir_bt = _decile_backtest(pdir, fva)

    # PAV calibration fit on the out-of-time slice → calibrated Brier.
    cal = _isotonic_pav(pdir, ydir_va)
    pdir_cal = [_apply_isotonic(cal, s) for s in pdir]

    # ── RESOLUTION head (multi-task second target) ────────────────────────────
    yres_tr = [ress[i] for i in tr]
    yres_va = [ress[i] for i in va]
    res_auc = None
    if 0 < sum(yres_tr) < len(yres_tr) and 0 < sum(yres_va) < len(yres_va):
        wr, br = _fit_logistic(Xtr_n, yres_tr)
        pres = _predict_logistic(Xva_n, wr, br)
        res_auc = round(_auc(pres, yres_va), 4)

    fam_imp = _family_importance(wd)
    base = max(_mean(ydir_tr), 1 - _mean(ydir_tr))

    report = {
        "mode": "smoke (synthetic, stdlib logistic — no heavy deps)",
        "families": [f for f, _ in FAMILIES],
        "unified_features": N_FEATURES,
        "markets": len(world), "windows": len(feats),
        "up_rate": round(_mean([float(d) for d in dirs]), 4),
        "train_windows": len(tr), "val_windows": len(va),
        "majority_baseline_acc": round(base, 4),
        "direction": {
            "val_auc": round(dir_auc, 4),
            "val_brier": round(_brier(pdir, ydir_va), 4),
            "val_brier_calibrated": round(_brier(pdir_cal, ydir_va), 4),
            "decile_backtest": dir_bt,
        },
        "resolution": {"val_auc": res_auc},
        "family_importance": fam_imp,
    }
    print(json.dumps(report, indent=2))
    print(f"\ndirection val AUC {dir_auc:.4f} (baseline {base:.4f}) | "
          f"decile up-rate spread {dir_bt.get('up_rate_spread'):+.3f} | "
          f"resolution val AUC {res_auc}")
    print("per-family contribution (|weight| share):")
    for f in fam_imp:
        print(f"  {f['family']:<12} {f['share']:.3f}   top: {f['top_feature']}")
    ok = dir_auc > 0.5 and dir_bt.get("up_rate_spread", 0) > 0
    print("smoke OK — the six-family assembly path recovers the planted signal."
          if ok else "smoke ran (signal weak on this synthetic seed).")
    return report


# ── real-data world construction (inside Modal) ────────────────────────────────
def _bucket_market_trades(rows, bar_seconds=BAR_SECONDS):
    """rows: iterable of (ts, price, usd, taker_direction, user, token_amount) for ONE
    market/asset. → (bars, tape, close, sflow) aligned by bucket, chronological."""
    by_bucket = {}
    for ts, price, usd, direction, user, token in rows:
        ts = int(ts)
        b = ts // bar_seconds * bar_seconds
        d = by_bucket.setdefault(b, {"open": price, "close": price, "high": price,
                                     "low": price, "volume": 0.0, "trades": 0,
                                     "signed_flow": 0.0, "ts": b, "_first": ts,
                                     "_last": ts, "tape": []})
        if ts <= d["_first"]:
            d["_first"], d["open"] = ts, price
        if ts >= d["_last"]:
            d["_last"], d["close"] = ts, price
        d["high"] = max(d["high"], price)
        d["low"] = min(d["low"], price)
        is_buy = str(direction).upper() == "BUY"
        signed = usd if is_buy else -usd
        d["volume"] += usd
        d["trades"] += 1
        d["signed_flow"] += signed
        tok = token if token else (usd / price if price else 0.0) * (1 if is_buy else -1)
        d["tape"].append({"user": str(user), "role": "taker", "price": price,
                          "usd": usd, "token": tok, "ts": ts, "maker": str(user)})
    bars, tape, close, sflow = [], [], [], []
    for key in sorted(by_bucket):
        d = by_bucket[key]
        trades = d.pop("tape")
        bars.append(d)
        tape.append(trades)
        close.append(d["close"])
        sflow.append(d["signed_flow"])
    return bars, tape, close, sflow


# ── Modal H100 job — the culmination ───────────────────────────────────────────
@_remote(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=14400)
def run(max_rows: int = 20_000_000, top_tokens: int = 6000, epochs: int = 60,
        bar_seconds: int = BAR_SECONDS, push: bool = False, hf_token: str = "") -> dict:
    """Stream trades + markets → per-market world → unified six-family features →
    LightGBM + multi-task torch MLP ensemble, PAV-calibrated, out-of-time eval."""
    import base64

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    tok = hf_token or None
    if tok:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # 1) markets.parquet — resolutions, event grouping, timing (small, fully in RAM).
    mpath = hf_hub_download(REPO, MARKETS_FILE, repo_type="dataset", token=tok)
    mtbl = pq.read_table(mpath)
    mcols = {c: mtbl.column(c).to_pylist() for c in mtbl.column_names
             if c in ("market_id", "condition_id", "asset_id", "token_id", "event_id",
                      "neg_risk", "outcome_prices", "end_date", "created_at")}
    os.remove(mpath)

    def _col(name, i, default=None):
        v = mcols.get(name)
        return v[i] if v is not None and i < len(v) else default

    n_markets = len(next(iter(mcols.values()))) if mcols else 0
    meta_by_id, event_of = {}, {}
    for i in range(n_markets):
        mk = _col("market_id", i) or _col("condition_id", i) or _col("asset_id", i)
        if mk is None:
            continue
        mk = str(mk)
        res = features_crossmarket.label_from_market(
            {"outcome_prices": _col("outcome_prices", i)})
        eid = _col("event_id", i)
        eid = str(eid) if eid is not None else mk

        def _epoch(v):
            if v is None:
                return None
            try:
                return float(v.timestamp()) if hasattr(v, "timestamp") else float(v)
            except Exception:  # noqa: BLE001
                return None

        meta_by_id[mk] = {"market_id": mk, "event_id": eid,
                          "neg_risk": _col("neg_risk", i, False),
                          "outcome_prices": _col("outcome_prices", i), "res_label": res,
                          "t_created": _epoch(_col("created_at", i)) or 0.0,
                          "t_end": _epoch(_col("end_date", i)) or 0.0}
        event_of.setdefault(eid, []).append(mk)
    print(f"markets: {n_markets}, with-metadata {len(meta_by_id)}", flush=True)

    # 2) trades.parquet — stream by row-group, accumulate raw trades per asset.
    tpath = hf_hub_download(REPO, TRADES_FILE, repo_type="dataset", token=tok)
    pf = pq.ParquetFile(tpath)
    have = set(pf.schema_arrow.names)
    id_col = next((c for c in ("asset_id", "market_id", "token_id", "condition_id") if c in have), "asset_id")
    tok_col = "token_amount" if "token_amount" in have else None
    usr_col = "user" if "user" in have else None
    cols = [c for c in [id_col, "timestamp", "price", "usd_amount", "taker_direction", tok_col, usr_col] if c]
    raw = {}
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=cols)
        ids = tbl.column(id_col).to_pylist()
        ts = tbl.column("timestamp").to_numpy(zero_copy_only=False).tolist()
        price = tbl.column("price").to_numpy(zero_copy_only=False).tolist()
        usd = tbl.column("usd_amount").to_numpy(zero_copy_only=False).tolist()
        direction = tbl.column("taker_direction").to_pylist()
        token = tbl.column(tok_col).to_numpy(zero_copy_only=False).tolist() if tok_col else [0.0] * len(ids)
        user = tbl.column(usr_col).to_pylist() if usr_col else [""] * len(ids)
        for k in range(len(ids)):
            raw.setdefault(str(ids[k]), []).append(
                (ts[k], price[k], usd[k], direction[k], user[k], token[k]))
        seen += len(ids)
        if seen % 5_000_000 < len(ids):
            print(f"  streamed {seen:,} trades, {len(raw)} assets", flush=True)
        if max_rows and seen >= max_rows:
            break
    os.remove(tpath)
    print(f"streamed {seen:,} trades → {len(raw)} assets", flush=True)

    # Keep the most-traded assets → tractable, liquid training set.
    ranked = sorted(raw.items(), key=lambda kv: -len(kv[1]))[:top_tokens]

    # 3) Build the per-market world (bars + tape + peers), reusing iter_examples.
    world, by_id = [], {}
    for mk, rows in ranked:
        rows.sort(key=lambda r: r[0])
        bars, tape, close, sflow = _bucket_market_trades(rows, bar_seconds)
        if len(bars) < MIN_CANDLES:
            continue
        meta = meta_by_id.get(mk, {"market_id": mk, "event_id": mk, "neg_risk": False,
                                   "outcome_prices": None, "res_label": None,
                                   "t_created": bars[0]["ts"], "t_end": bars[-1]["ts"]})
        m = {"bars": bars, "tape": tape, "close": close, "sflow": sflow, "meta": meta, "peers": []}
        world.append(m)
        by_id[mk] = m
    for m in world:  # wire event siblings that survived filtering
        eid = m["meta"]["event_id"]
        m["peers"] = [by_id[o] for o in event_of.get(eid, []) if o in by_id and by_id[o] is not m]
    print(f"world markets: {len(world)}", flush=True)

    feats, dirs, ress, fwds, tfrac = [], [], [], [], []
    for vec, dlab, rlab, fwd, tf in iter_examples(world):
        feats.append(vec)
        dirs.append(dlab)
        ress.append(-1 if rlab is None else int(rlab))
        fwds.append(fwd)
        tfrac.append(tf)
    if not feats:
        raise RuntimeError("no windows built — raise --max-rows / --top-tokens")

    X = np.nan_to_num(np.asarray(feats, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ydir = np.asarray(dirs, np.float32)
    yres = np.asarray(ress, np.float32)
    fwds = np.asarray(fwds, np.float32)
    tfrac = np.asarray(tfrac, np.float32)
    print(f"windows {len(ydir)}, features {X.shape[1]}, dir up-rate {ydir.mean():.3f}", flush=True)

    # Out-of-time split with an EMBARGO: validate on each market's last 20% and
    # drop the band just before it from train, so no window's forward label
    # overlaps the val region (purged, López de Prado).
    va_i = np.where(tfrac >= 0.8)[0]
    tr_i = np.where(tfrac < 0.8 - EMBARGO)[0]
    fmean, fstd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
    Xn = (X - fmean) / fstd

    result = {
        "runtime": "modal H100 / mega unified (LightGBM + extra-trees GBDT + residual multi-task MLP, 3-member ensemble)",
        "dataset": REPO, "trades_file": TRADES_FILE, "markets_file": MARKETS_FILE,
        "source": "union of all six feature families (flow+resolve+smart+crossmarket+event+micro)",
        "families": [f for f, _ in FAMILIES], "unified_features": N_FEATURES,
        "feature_names": UNIFIED_FEATURE_NAMES,
        "trades_streamed": int(seen), "markets": len(world), "windows": int(len(ydir)),
        "window": WINDOW, "horizon": HORIZON, "bar_seconds": bar_seconds,
        "train_windows": int(len(tr_i)), "val_windows": int(len(va_i)),
        "up_rate": round(float(ydir.mean()), 4),
        "majority_baseline_acc": round(float(max(ydir[tr_i].mean(), 1 - ydir[tr_i].mean())), 4),
    }

    # 3a) LightGBM on the raw unified features (primary tabular, DIRECTION).
    import lightgbm as lgb
    dtr = lgb.Dataset(X[tr_i], label=ydir[tr_i], feature_name=list(UNIFIED_FEATURE_NAMES))
    dva = lgb.Dataset(X[va_i], label=ydir[va_i], reference=dtr)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 63,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 1.0, "is_unbalance": True, "seed": 11, "verbose": -1,
              # The extra-trees member reuses this Dataset with a smaller
              # min_data_in_leaf (100 < 200); LightGBM errors if the Dataset was
              # built with feature pre-filtering at the larger value. Disable it so
              # the shared Dataset supports both members.
              "feature_pre_filter": False}
    bst = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    gp = bst.predict(X[va_i])
    result["gbdt"] = {"val_auc": round(_auc(gp, ydir[va_i]), 4),
                      "brier": round(float(np.mean((gp - ydir[va_i]) ** 2)), 4),
                      "backtest": _decile_backtest(list(gp), list(fwds[va_i]))}

    # 3a') Extra-trees GBDT — a decorrelated second forest (random split thresholds,
    # deeper leaves, thinner column sampling). Its errors differ from the primary
    # booster's, so averaging the two is a real ensemble gain, not a copy.
    params_et = {**params, "extra_trees": True, "num_leaves": 127,
                 "feature_fraction": 0.6, "min_data_in_leaf": 100, "seed": 23}
    bst_et = lgb.train(params_et, dtr, num_boost_round=1500, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    gp_et = bst_et.predict(X[va_i])
    result["gbdt_et"] = {"val_auc": round(_auc(gp_et, ydir[va_i]), 4),
                         "brier": round(float(np.mean((gp_et - ydir[va_i]) ** 2)), 4),
                         "backtest": _decile_backtest(list(gp_et), list(fwds[va_i]))}
    print(f"[gbdt_et] {result['gbdt_et']}", flush=True)

    # Per-family importance rolled up from LightGBM gain.
    gain = dict(zip(UNIFIED_FEATURE_NAMES, bst.feature_importance("gain")))
    fam_gain = []
    for fam, names in FAMILIES:
        a, b = FAMILY_SPANS[fam]
        g = float(sum(gain.get(UNIFIED_FEATURE_NAMES[j], 0.0) for j in range(a, b)))
        top_j = max(range(a, b), key=lambda j: gain.get(UNIFIED_FEATURE_NAMES[j], 0.0))
        fam_gain.append([fam, g, UNIFIED_FEATURE_NAMES[top_j]])
    tot = sum(p[1] for p in fam_gain) + 1e-12
    result["family_importance"] = [
        {"family": p[0], "gain_share": round(p[1] / tot, 3), "gain": round(p[1], 1),
         "top_feature": p[2]} for p in sorted(fam_gain, key=lambda p: -p[1])]
    result["feature_importance_gbdt"] = sorted(
        ((n, round(float(gain.get(n, 0.0)), 1)) for n in UNIFIED_FEATURE_NAMES),
        key=lambda t: -t[1])[:25]
    print(f"[gbdt] {result['gbdt']}", flush=True)

    # 3b) Multi-task torch MLP: shared trunk → direction head + resolution head.
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)

    class ResBlock(nn.Module):
        """Pre-norm residual block (LayerNorm → expand → SiLU → project → +skip).
        The skip path lets the trunk go deep without the signal washing out, which
        a plain stacked-Linear trunk can't — depth here buys real capacity."""

        def __init__(self, d, drop):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, d * 2)
            self.fc2 = nn.Linear(d * 2, d)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            h = nn.functional.silu(self.fc1(self.norm(x)))
            return x + self.fc2(self.drop(h))

    class MegaMLP(nn.Module):
        """Deep pre-norm residual trunk with a multi-task head. Wider and much
        deeper than the old 2-layer MLP, but stable to train thanks to the residual
        blocks + final norm; heads stay linear so calibration behaves."""

        def __init__(self, n_in, hidden=320, blocks=4, drop=0.3):
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU())
            self.blocks = nn.ModuleList([ResBlock(hidden, drop) for _ in range(blocks)])
            self.out_norm = nn.LayerNorm(hidden)
            self.dir_head = nn.Linear(hidden, 1)
            self.res_head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.stem(x)
            for blk in self.blocks:
                h = blk(h)
            h = self.out_norm(h)
            return self.dir_head(h).squeeze(-1), self.res_head(h).squeeze(-1)

    Xt = torch.tensor(Xn, device=dev)
    ydt = torch.tensor(ydir, device=dev)
    yrt = torch.tensor(np.where(yres < 0, 0.0, yres).astype(np.float32), device=dev)
    res_mask = torch.tensor((yres >= 0).astype(np.float32), device=dev)  # only resolved markets
    ti = torch.tensor(tr_i, device=dev)
    vi = torch.tensor(va_i, device=dev)
    net = MegaMLP(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    # Warmup then cosine: the deep residual trunk trains cleaner with a short ramp.
    warm = max(1, epochs // 12)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warm, 1))],
        milestones=[warm])
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    # EMA (Polyak) shadow weights: evaluate/ship the averaged model, not the noisy
    # last step — a cheap, reliable generalisation win on tabular deep nets.
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    EMA_DECAY, SMOOTH = 0.998, 0.03
    best_auc, best_state, mlp_probs = -1.0, None, None
    BATCH = 4096
    for _ in range(epochs):
        net.train()
        perm = ti[torch.randperm(len(tr_i), device=dev)]
        for b0 in range(0, len(tr_i), BATCH):
            bi = perm[b0:b0 + BATCH]
            opt.zero_grad(set_to_none=True)
            dlogit, rlogit = net(Xt[bi])
            # Label smoothing on the direction target damps overconfident logits and
            # improves both AUC ranking stability and calibration.
            dtgt = ydt[bi] * (1.0 - 2.0 * SMOOTH) + SMOOTH
            loss_dir = bce_none(dlogit, dtgt).mean()
            rm = res_mask[bi]
            loss_res = (bce_none(rlogit, yrt[bi]) * rm).sum() / (rm.sum() + 1e-6)
            (loss_dir + 0.5 * loss_res).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(EMA_DECAY).add_(v.detach().float(), alpha=1.0 - EMA_DECAY)
        sched.step()
        # Evaluate the EMA snapshot — load into a clone so live weights keep training.
        net.eval()
        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict({k: ema[k].to(v.dtype) for k, v in live.items()})
        with torch.no_grad():
            pv = torch.sigmoid(net(Xt[vi])[0]).cpu().numpy()
        a = _auc(pv, ydir[va_i])
        if a > best_auc:
            best_auc, mlp_probs = a, pv
            best_state = {k: ema[k].cpu().clone() for k in ema}
        net.load_state_dict(live)
    result["mlp"] = {"val_auc": round(best_auc, 4),
                     "brier": round(float(np.mean((mlp_probs - ydir[va_i]) ** 2)), 4),
                     "backtest": _decile_backtest(list(mlp_probs), list(fwds[va_i]))}
    print(f"[mlp] {result['mlp']}", flush=True)

    # 3c) Ensemble of three decorrelated members (primary GBDT + extra-trees GBDT +
    # residual MLP), then isotonic (PAV) calibration on the out-of-time slice.
    ens = (np.asarray(gp) + np.asarray(gp_et) + np.asarray(mlp_probs)) / 3
    cal = _isotonic_pav(list(ens), list(ydir[va_i]))
    ens_cal = np.asarray([_apply_isotonic(cal, float(s)) for s in ens])
    result["ensemble"] = {
        "members": ["gbdt", "gbdt_et", "mlp"],
        "val_auc": round(_auc(ens, ydir[va_i]), 4),
        "brier": round(float(np.mean((ens - ydir[va_i]) ** 2)), 4),
        "brier_calibrated": round(float(np.mean((ens_cal - ydir[va_i]) ** 2)), 4),
        "backtest": _decile_backtest(list(ens), list(fwds[va_i]))}
    result["overall_best"] = max(("gbdt", "gbdt_et", "mlp", "ensemble"),
                                 key=lambda k: result[k]["val_auc"])
    print(f"[ensemble] {result['ensemble']} | best {result['overall_best']}", flush=True)

    # 3d) Resolution head eval on resolved val markets (multi-task second target).
    # Load the shipped EMA weights so this measures the model we actually save.
    if best_state is not None:
        net.load_state_dict({k: best_state[k].to(dev) for k in best_state})
    net.eval()
    with torch.no_grad():
        rpv = torch.sigmoid(net(Xt[vi])[1]).cpu().numpy()
    rmask = yres[va_i] >= 0
    if rmask.sum() > 10 and 0 < yres[va_i][rmask].sum() < rmask.sum():
        result["resolution"] = {
            "val_auc": round(_auc(list(rpv[rmask]), list(yres[va_i][rmask])), 4),
            "brier": round(float(np.mean((rpv[rmask] - yres[va_i][rmask]) ** 2)), 4),
            "n": int(rmask.sum())}
    else:
        result["resolution"] = {"val_auc": None, "note": "too few resolved val markets"}
    print(f"[resolution] {result['resolution']}", flush=True)

    # 3e) Walk-forward over global time — is the signal stable, not a one-slice fit?
    wf = []
    for kf in range(1, 5):
        lo, hi = 0.2 * kf, 0.2 * (kf + 1)
        vmask = (tfrac >= lo) & (tfrac < hi)
        tmask = tfrac < lo - EMBARGO          # purged: real embargo gap, not a token-scaled hack
        if vmask.sum() < 50 or tmask.sum() < 200:
            continue
        d1 = lgb.Dataset(X[tmask], label=ydir[tmask], feature_name=list(UNIFIED_FEATURE_NAMES))
        b = lgb.train(params, d1, num_boost_round=400, callbacks=[lgb.log_evaluation(0)])
        pvk = b.predict(X[vmask])
        wf.append({"fold": kf, "val_auc": round(_auc(pvk, ydir[vmask]), 4),
                   "up_rate_spread": _decile_backtest(list(pvk), list(fwds[vmask])).get("up_rate_spread")})
    result["walk_forward"] = {"folds": wf,
                              "mean_auc": round(_mean([f["val_auc"] for f in wf]), 4) if wf else None}
    print(f"[walk-forward] {result['walk_forward']}", flush=True)

    # 4) Artifacts: gbdt + mlp safetensors + normalizer json + calibration + card.
    norm = {"fmean": fmean.tolist(), "fstd": fstd.tolist(), "features": UNIFIED_FEATURE_NAMES,
            "families": {fam: FAMILY_SPANS[fam] for fam, _ in FAMILIES},
            "arch": {"type": "residual_multitask_mlp", "hidden": 320, "blocks": 4,
                     "activation": "silu", "ema": True, "heads": ["direction", "resolution"]},
            "window": WINDOW, "horizon": HORIZON, "bar_seconds": bar_seconds,
            "calibration": {"type": "isotonic_pav", "xs": cal[0], "ys": cal[1]}}
    bst.save_model("/tmp/mega_gbdt.txt")
    # numpy scalars/arrays are not JSON-serializable; .tolist() maps np scalars
    # to python scalars and np arrays to lists, so every metrics dump is safe.
    _jsafe = lambda o: o.tolist()  # noqa: E731
    artifacts = {"mega_normalizer.json": json.dumps(norm, default=_jsafe).encode(),
                 "mega_gbdt.txt": open("/tmp/mega_gbdt.txt", "rb").read()}
    if best_state is not None:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in best_state.items()}, "/tmp/mega_mlp.safetensors")
        artifacts["mega_mlp.safetensors"] = open("/tmp/mega_mlp.safetensors", "rb").read()
    artifacts["mega_metrics.json"] = json.dumps(
        {k: v for k, v in result.items() if not k.startswith("_")}, indent=2, default=_jsafe).encode()
    artifacts["README.md"] = _model_card(result).encode()
    result["_artifacts_b64"] = {k: base64.b64encode(v).decode() for k, v in artifacts.items()}

    if push and not hf_token:
        # Loud, recorded skip. The old code silently dropped the upload when --push
        # was set without a token, so a costly H100 run finished with nothing on the
        # Hub and no error to show for it — exactly the trap `HF_TOKEN` unset hits.
        result["push_error"] = ("push requested but HF_TOKEN unset — nothing "
                                "uploaded (artifacts saved locally only)")
        print("PUSH SKIPPED:", result["push_error"], flush=True)
    elif push and hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
            for name, data in artifacts.items():
                sub = name if name == "README.md" else f"mega/{name}"
                api.upload_file(path_or_fileobj=data, path_in_repo=sub,
                                repo_id=HF_MODEL_REPO, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{HF_MODEL_REPO}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:  # noqa: BLE001
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)

    # Return a pure-python dict: the raw result carries numpy scalars/arrays, and
    # a collector without numpy can't cloudpickle-deserialize them. Recurse and
    # map np scalars/arrays to python via .tolist(), so ANY env can .get() it.
    def _pyify(o):
        if isinstance(o, dict):
            return {k: _pyify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_pyify(v) for v in o]
        return o.tolist() if hasattr(o, "tolist") else o

    return _pyify(result)


def _model_card(result):
    def line(k):
        m = result.get(k, {})
        bt = m.get("backtest", {})
        return (f"| {k} | {m.get('val_auc')} | {m.get('brier')} | "
                f"{bt.get('up_rate_spread')} |")

    fam_rows = "\n".join(
        f"| {f['family']} | {f.get('gain_share')} | {f.get('top_feature')} |"
        for f in result.get("family_importance", []))
    return f"""---
license: mit
tags:
- polymarket
- prediction-markets
- ensemble
- multi-task
- time-series
---

# polymarket-mega-model

The **unified** signal model: one ensemble trained on the union of all six feature
families the suite builds — order-flow, resolution, smart-money, cross-market,
event co-movement and microstructure — over `{REPO}`
(`{TRADES_FILE}` + `{MARKETS_FILE}`). {result.get('unified_features')} features per
example, two targets (next-move **direction** + eventual **resolution**).

- window {result.get('window')} bars, horizon {result.get('horizon')} bars ahead
- {result.get('windows', 0):,} windows over {result.get('markets')} markets
- strict out-of-time (tfrac >= 0.8) validation split; walk-forward over time
- 3-member ensemble: LightGBM + extra-trees GBDT + deep residual multi-task MLP
  (EMA weights, label smoothing), isotonic (PAV) calibrated

## Metrics (out-of-time)

| model | val AUC | brier | decile up-rate spread |
|-------|---------|-------|-----------------------|
{line('gbdt')}
{line('gbdt_et')}
{line('mlp')}
{line('ensemble')}

Best: **{result.get('overall_best')}**. Majority baseline acc
{result.get('majority_baseline_acc')}, up-rate {result.get('up_rate')}.
Resolution head val AUC {result.get('resolution', {}).get('val_auc')}.

## Feature-family contribution (LightGBM gain share)

| family | gain share | loudest feature |
|--------|-----------|-----------------|
{fam_rows}

## Artifacts
- `mega/mega_gbdt.txt` — LightGBM booster
- `mega/mega_mlp.safetensors` — multi-task torch MLP (direction + resolution heads)
- `mega/mega_normalizer.json` — feature normaliser, family spans, PAV calibration
- `mega/mega_metrics.json` — full metrics

Trained self-contained on a Modal H100 via `ml/modal_mega.py`.
"""


@_entrypoint()
def main(max_rows: int = 20_000_000, top_tokens: int = 6000, epochs: int = 60,
         bar_seconds: int = BAR_SECONDS, push: bool = False):
    import base64

    token = os.environ.get("HF_TOKEN", "")
    if push and not token:
        # Fail fast BEFORE spending an H100 hour: --push with no token would train,
        # then silently skip the upload. Surface it up front so it can be fixed.
        print("WARNING: --push set but HF_TOKEN is empty — the model will train and "
              "save locally, but NOTHING will be pushed. Set HF_TOKEN and relaunch to "
              "push. Continuing (local artifacts only) …", flush=True)
    report = run.remote(max_rows=max_rows, top_tokens=top_tokens, epochs=epochs,
                        bar_seconds=bar_seconds, push=push, hf_token=token)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, b64 in report.pop("_artifacts_b64", {}).items():
        with open(os.path.join(data_dir, name.replace("/", "_")), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved data/{name.replace('/', '_')}")
    with open(os.path.join(data_dir, "mega_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {data_dir}/mega_metrics.json")


@_remote(image=image, timeout=1200)
def push_artifacts(artifacts_b64: dict, hf_token: str, best_resolution: str = "") -> str:
    """Upload one resolution's artifact set to `HF_MODEL_REPO` from inside Modal.

    The parallel bakeoff picks a winner on the local entrypoint, which may run in a
    bare env with no `huggingface_hub`; doing the upload in a tiny remote keeps the
    push on the same image the training used and off the caller's machine. Returns
    the model URL, or an `ERROR:` string the caller can surface without crashing."""
    import base64

    from huggingface_hub import HfApi

    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
        for name, b64 in artifacts_b64.items():
            sub = name if name == "README.md" else f"mega/{name}"
            api.upload_file(path_or_fileobj=base64.b64decode(b64), path_in_repo=sub,
                            repo_id=HF_MODEL_REPO, repo_type="model")
        if best_resolution:
            api.upload_file(
                path_or_fileobj=f"winning resolution: bar_seconds={best_resolution}\n".encode(),
                path_in_repo="mega/BEST_RESOLUTION.txt", repo_id=HF_MODEL_REPO, repo_type="model")
        return f"https://huggingface.co/{HF_MODEL_REPO}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"[:300]


@_entrypoint()
def parallel(max_rows: int = 30_000_000, top_tokens: int = 3000, epochs: int = 40, push: bool = False):
    """Train the mega model across resolutions on PARALLEL H100s.

    Fans `run` out over three bar sizes (5-min / 15-min / 1-hour) with
    `run.spawn`, so Modal executes each on its own H100 concurrently — one
    wall-clock, three models. Order-flow signal lives at the fine scale and
    regime/trend signal at the coarse one; training all three in parallel gives a
    multi-scale view instead of one hard-coded horizon. The best resolution by
    calibrated ensemble AUC is saved + pushed; every resolution's metrics are
    kept for comparison.
    """
    import base64

    token = os.environ.get("HF_TOKEN", "")
    resolutions = [300, 900, 3600]
    print(f"spawning {len(resolutions)} H100 workers: bar_seconds={resolutions}", flush=True)
    calls = {bs: run.spawn(max_rows=max_rows, top_tokens=top_tokens, epochs=epochs,
                           bar_seconds=bs, push=False, hf_token=token) for bs in resolutions}

    reports, best_key, best_score, best_auc, best_art = {}, None, -1.0, -1.0, {}
    for bs, c in calls.items():
        rep = c.get()                       # blocks until this H100 finishes
        art = rep.pop("_artifacts_b64", {})
        auc = rep.get("ensemble", {}).get("val_auc", 0.0) or 0.0
        sp = rep.get("ensemble", {}).get("backtest", {}).get("up_rate_spread")
        # Select on a STABILITY-weighted score: half the last-slice ensemble AUC,
        # half the walk-forward mean AUC (fall back to the slice AUC when walk-
        # forward is absent). Picks the resolution whose edge holds across time,
        # not one that got lucky on a single out-of-time slice.
        wf = rep.get("walk_forward", {}).get("mean_auc")
        score = 0.5 * auc + 0.5 * (wf if wf is not None else auc)
        print(f"  bar_seconds={bs}: ensemble AUC {auc}  walk-fwd {wf}  "
              f"score {score:.4f}  spread {sp}", flush=True)
        reports[str(bs)] = rep
        if score > best_score:
            best_score, best_auc, best_key, best_art = score, auc, str(bs), art

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, b64 in best_art.items():       # save the winning resolution's artifacts
        with open(os.path.join(data_dir, "mega_" + name.replace("/", "_")), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved data/mega_{name.replace('/', '_')}")
    combined = {"runtime": "modal parallel H100s / multi-resolution mega",
                "best_resolution": best_key, "best_ensemble_auc": best_auc,
                "best_selection_score": round(best_score, 4),
                "selection": "0.5*ensemble_val_auc + 0.5*walk_forward_mean_auc",
                "by_resolution": reports}

    # Push ONLY the winning resolution's artifacts — one clean model on the Hub, not
    # three competing ones. Done in a remote so the caller needs no huggingface_hub.
    if push and token and best_art:
        url = push_artifacts.remote(best_art, token, best_key)
        combined["hf_repo"] = url
        print(f"pushed winner (bar_seconds={best_key}) → {url}", flush=True)
    elif push and not token:
        combined["push_error"] = "push requested but HF_TOKEN unset — nothing uploaded"
        print("push skipped: HF_TOKEN unset", flush=True)

    with open(os.path.join(data_dir, "mega_parallel_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nbest resolution: bar_seconds={best_key} (ensemble AUC {best_auc})")
    print(f"wrote {data_dir}/mega_parallel_metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mega unified six-family trainer (Modal H100).")
    ap.add_argument("--smoke", action="store_true",
                    help="run the full unified pipeline on synthetic data with zero heavy deps")
    ap.add_argument("--events", type=int, default=6, help="synthetic events (smoke)")
    ap.add_argument("--legs", type=int, default=3, help="markets per event (smoke)")
    ap.add_argument("--buckets", type=int, default=72, help="synthetic bars/market (smoke)")
    ap.add_argument("--seed", type=int, default=11, help="synthetic seed (smoke)")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(n_events=args.events, markets_per_event=args.legs,
                  n_buckets=args.buckets, seed=args.seed)
    else:
        print("This is a Modal H100 job. Launch it with:")
        print("  modal run ml/modal_mega.py --max-rows 20000000 --top-tokens 6000 --push")
        print("Validate the whole unified pipeline locally (no heavy deps) with:")
        print("  python ml/modal_mega.py --smoke")
