"""Smart-money signal trainer, on an H100 — a capability the repo has never had.

Every other model here reads *price and volume*. This one reads *who is trading*.
The `SII-WANGZJ/Polymarket_data` dataset ships a user-level tape (`users.parquet`)
— every fill tagged with the wallet behind it — so we can build a per-wallet PnL
leaderboard, tag the consistently-profitable "smart-money" cohort, and turn their
flow into a forward-return signal: when the wallets that have historically been
right lean one way, does the market follow?

Pipeline (all inside Modal, streamed from HF):

1. Stream `users.parquet` (+ `markets.parquet` for resolutions) via `hf_hub_download`.
2. Build a per-wallet PnL leaderboard, marking each trade to its market's final
   resolution (or last price). Tag the top cohort "smart", the bottom "dumb".
3. Per market, per prediction point, inline six smart-money features —
   smart_net_flow, smart_share, herding, crowding, whale_concentration,
   smart_minus_dumb. Label = forward YES-price move.
4. Train a GBDT (LightGBM) on a strict out-of-time split; report AUC + a decile
   backtest + walk-forward folds.
5. Push artifacts + normalizer + metrics + a model card to
   `shubhxho/polymarket-smartmoney-model`.

    modal run ml/modal_smart.py --max-trades 3000000 --push

── Anti-leakage / survivorship (read this) ───────────────────────────────────
The obvious trap here is *lookahead*: if a wallet is tagged "smart" using the
very trade we are trying to predict, the signal is circular. We avoid it with a
**temporal cohort cutoff**. Wallets are scored ONLY from trades strictly before a
cutoff timestamp (`cohort_cutoff_frac` of the timeline); prediction windows are
drawn ONLY from *after* the cutoff. So a wallet's smart/dumb status is fixed from
its past and never informed by the window being predicted — the cohort is frozen
before evaluation begins, exactly as it would be live. Survivorship is
acknowledged, not hidden: the leaderboard only contains wallets that traded in
the pre-cutoff era, and a `min_volume` floor drops wallets whose "skill" is one
lucky fill. These caveats are written into the model card too.

── Smoke test (no GPU, no heavy deps) ─────────────────────────────────────────
`python ml/modal_smart.py --smoke` runs the WHOLE pipeline — synthetic wallet
population with latent skill → PnL leaderboard → cohorts → features → labels →
a hand-rolled logistic baseline → AUC + decile backtest — on pure Python stdlib.
It shares the exact cohort/feature/eval code the H100 job uses, so the smoke test
exercises the real logic, just on toy data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

# NOTE: `modal`, `torch`, `lightgbm`, `datasets`, `pyarrow`, `numpy`,
# `huggingface_hub` are NEVER imported at module top — they live inside the
# functions / the Modal image below, so this module imports cleanly on a machine
# with none of them installed (the smoke path is pure stdlib). The Modal app is
# built only when `modal` is importable (i.e. under `modal run`).

HF_REPO = "SII-WANGZJ/Polymarket_data"
OUT_REPO = "shubhxho/polymarket-smartmoney-model"

WINDOW = 12                 # bars of tape per prediction window
HORIZON = 3                 # bars ahead for the forward-return label
BAR_SECONDS = 3600          # hourly bars
MIN_BARS = WINDOW + HORIZON + 2
COHORT_TOP_FRAC = 0.2       # top / bottom 20% of the leaderboard = smart / dumb
COHORT_MIN_VOLUME = 200.0   # USD floor to enter the leaderboard (anti-lucky-fill)
COHORT_CUTOFF_FRAC = 0.5    # wallets scored from trades before this time fraction

FEATURE_NAMES = [
    "smart_net_flow",       # signed $-flow of smart wallets, / their gross  ∈[-1,1]
    "smart_share",          # smart wallets' share of window volume           ∈[0,1]
    "herding",              # |net| / gross of ALL flow (directional consensus)∈[0,1]
    "crowding",             # fraction of wallets on the majority side         ∈[0,1]
    "whale_concentration",  # largest single wallet's share of volume          ∈[0,1]
    "smart_minus_dumb",     # smart_net_flow − dumb_net_flow                   ∈[-2,2]
]


# ══════════════════════════════════════════════════════════════════════════════
# Pure-stdlib core — shared verbatim by the H100 job and the smoke test.
# No numpy / lightgbm / pyarrow here, on purpose.
# ══════════════════════════════════════════════════════════════════════════════

def _parse_final_yes(outcome_prices):
    """Resolution price of the YES (token-1) leg from `outcome_prices`.

    `["0.99","0.01"]` → YES resolved true → 0.99 (≈1). The real dataset does NOT
    ship JSON here — it's a single-quoted Python-list repr like `"['1', '0']"`,
    so `json.loads` fails; we fall back to `ast.literal_eval`. Robust to str/list
    input; returns None when unparseable so the market is simply skipped."""
    arr = outcome_prices
    if isinstance(outcome_prices, str):
        try:
            arr = json.loads(outcome_prices)
        except (ValueError, TypeError):
            try:
                import ast
                arr = ast.literal_eval(outcome_prices)
            except (ValueError, SyntaxError, TypeError):
                return None
    try:
        if not arr:
            return None
        return float(arr[0])
    except (ValueError, TypeError, IndexError):
        return None


def _build_final_prices(market_rows):
    """{market_id: final_yes_price} from markets.parquet rows."""
    out = {}
    for m in market_rows:
        fy = _parse_final_yes(m.get("outcome_prices"))
        if fy is not None:
            out[m.get("market_id")] = fy
    return out


def _leaderboard(trades, final_prices):
    """Mark every pre-cutoff trade to its market's resolution → per-wallet PnL.

    PnL of a fill = signed_tokens · (final_yes − fill_price): buying YES cheap and
    resolving true pays off; the sign of `token_amount` (+buy / −sell) carries the
    side. Falls back to the market's last seen price when it never resolved."""
    last_price = {}
    for t in trades:
        last_price[t["market_id"]] = t["price"]
    pnl, vol = {}, {}
    for t in trades:
        mk = t["market_id"]
        mark = final_prices.get(mk, last_price.get(mk))
        if mark is None:
            continue
        u = t["user"]
        pnl[u] = pnl.get(u, 0.0) + t["token_amount"] * (mark - t["price"])
        vol[u] = vol.get(u, 0.0) + abs(t["usd_amount"])
    return pnl, vol


def _cohorts(pnl, vol, top_frac=COHORT_TOP_FRAC, min_volume=COHORT_MIN_VOLUME):
    """Top / bottom `top_frac` of qualifying wallets by realized PnL.

    The `min_volume` floor removes wallets whose PnL is a single lucky fill — a
    deliberate survivorship control (documented). Returns (smart:set, dumb:set)."""
    ranked = sorted(
        (u for u in pnl if vol.get(u, 0.0) >= min_volume),
        key=lambda u: pnl[u],
    )
    if len(ranked) < 5:
        return set(), set()
    # Cap k at just under half so the top and bottom slices never overlap even
    # for an aggressive top_frac (an overlap would put a wallet in both cohorts
    # and silently null out smart_minus_dumb).
    k = max(1, min(int(len(ranked) * top_frac), (len(ranked) - 1) // 2))
    dumb = set(ranked[:k])          # lowest PnL
    smart = set(ranked[-k:])        # highest PnL
    return smart, dumb


def _market_features(window_trades, smart, dumb):
    """Six smart-money features for one prediction window (a market's tape slice).

    All are bounded / scale-free so they transfer across markets of any size."""
    eps = 1e-9
    gross_all = eps
    net_all = 0.0
    smart_gross = eps
    smart_net = 0.0
    dumb_gross = eps
    dumb_net = 0.0
    wallet_gross = {}
    buy_wallets, sell_wallets, all_wallets = set(), set(), set()
    for t in window_trades:
        usd = abs(t["usd_amount"])
        sgn = 1.0 if t["token_amount"] > 0 else -1.0
        u = t["user"]
        gross_all += usd
        net_all += sgn * usd
        wallet_gross[u] = wallet_gross.get(u, 0.0) + usd
        all_wallets.add(u)
        (buy_wallets if sgn > 0 else sell_wallets).add(u)
        if u in smart:
            smart_gross += usd
            smart_net += sgn * usd
        if u in dumb:
            dumb_gross += usd
            dumb_net += sgn * usd
    smart_net_flow = smart_net / smart_gross
    dumb_net_flow = dumb_net / dumb_gross
    smart_share = (smart_gross - eps) / gross_all
    herding = abs(net_all) / gross_all
    crowding = max(len(buy_wallets), len(sell_wallets)) / max(len(all_wallets), 1)
    whale_concentration = (max(wallet_gross.values()) if wallet_gross else 0.0) / gross_all
    return [
        smart_net_flow,
        smart_share,
        herding,
        crowding,
        whale_concentration,
        smart_net_flow - dumb_net_flow,
    ]


def _group_by_market(trades):
    g = {}
    for t in trades:
        g.setdefault(t["market_id"], []).append(t)
    return g


def _build_dataset(trades, final_prices, window=WINDOW, horizon=HORIZON,
                   bar_seconds=BAR_SECONDS, cutoff_frac=COHORT_CUTOFF_FRAC,
                   top_frac=COHORT_TOP_FRAC, min_volume=COHORT_MIN_VOLUME):
    """Trades → (X, y, fwd, ts, meta). The whole leakage-safe pipeline in one place.

    1. cutoff = `cutoff_frac` quantile of trade timestamps.
    2. cohorts scored ONLY from trades before cutoff (marked to resolution).
    3. per market, hourly last-price bars → windows whose end is AFTER cutoff;
       features from the window's trades, label from the forward bar.
    """
    if not trades:
        return [], [], [], [], {}
    ts_sorted = sorted(t["timestamp"] for t in trades)
    cutoff = ts_sorted[min(len(ts_sorted) - 1, int(len(ts_sorted) * cutoff_frac))]

    pre = [t for t in trades if t["timestamp"] < cutoff]
    pnl, vol = _leaderboard(pre, final_prices)
    smart, dumb = _cohorts(pnl, vol, top_frac, min_volume)

    X, y, fwd, ts = [], [], [], []
    for mk, mkt_trades in _group_by_market(trades).items():
        mkt_trades.sort(key=lambda t: t["timestamp"])
        bars = {}
        for t in mkt_trades:
            bars.setdefault(t["timestamp"] // bar_seconds, []).append(t)
        idx = sorted(bars)
        if len(idx) < MIN_BARS:
            continue
        prices = [bars[b][-1]["price"] for b in idx]   # last trade price per bar
        for i in range(window, len(idx) - horizon):
            wbars = idx[i - window:i]
            wtrades = [t for b in wbars for t in bars[b]]
            w_start = min(t["timestamp"] for t in wtrades)
            w_end = max(t["timestamp"] for t in wtrades)
            # Gate on the window START: the ENTIRE feature window must post-date
            # the cohort cutoff, so no trade that scored the smart/dumb cohorts can
            # also feed the features here (strict no-lookahead, not just partial).
            if w_start < cutoff:
                continue
            wprices = prices[i - window:i]
            if _std(wprices) < 1e-4:                    # skip dead windows
                continue
            move = prices[i + horizon] - prices[i]
            X.append(_market_features(wtrades, smart, dumb))
            y.append(1.0 if move > 0 else 0.0)
            fwd.append(move)
            ts.append(w_end)
    meta = {
        "cohort_cutoff_ts": int(cutoff),
        "smart_wallets": len(smart),
        "dumb_wallets": len(dumb),
        "leaderboard_wallets": len(pnl),
        "qualifying_wallets": sum(1 for u in pnl if vol.get(u, 0.0) >= min_volume),
    }
    return X, y, fwd, ts, meta


# ── stats / eval (pure python; mirrors train_seq._auc & _decile_backtest) ──────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _auc(probs, labels):
    """Rank AUC with tie handling — identical to train_seq._auc."""
    pairs = sorted(zip(probs, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, yy in pairs if yy > 0.5)
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
    """Top/bottom-`q` slice by score vs how markets actually moved — matches
    train_seq._decile_backtest (up-rate headline + median forward points)."""
    if not probs:
        return {}
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    k = max(1, int(len(order) * q))
    top = [fwd[i] for i in order[-k:]]
    bottom = [fwd[i] for i in order[:k]]

    def _median(xs):
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    up = lambda xs: sum(1 for v in xs if v > 0) / len(xs)  # noqa: E731
    return {
        "top_up_rate": round(up(top), 3),
        "bottom_up_rate": round(up(bottom), 3),
        "up_rate_spread": round(up(top) - up(bottom), 3),
        "top_median_pts": round(_median(top) * 100, 3),
        "bottom_median_pts": round(_median(bottom) * 100, 3),
        "slice": k,
    }


# ── hand-rolled logistic regression (smoke-only baseline, zero deps) ───────────

def _standardize_fit(X):
    d = len(X[0])
    mean = [_mean([row[j] for row in X]) for j in range(d)]
    std = [(_std([row[j] for row in X]) or 1.0) for j in range(d)]
    return mean, std


def _standardize_apply(X, mean, std):
    return [[(row[j] - mean[j]) / std[j] for j in range(len(mean))] for row in X]


def _fit_logreg(X, y, epochs=400, lr=0.3, l2=1e-3):
    """Class-weighted logistic regression by full-batch gradient descent."""
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    pos = _mean(y)
    w_pos, w_neg = 0.5 / max(pos, 1e-3), 0.5 / max(1 - pos, 1e-3)
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for row, yi in zip(X, y):
            z = b + sum(w[j] * row[j] for j in range(d))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            cw = w_pos if yi > 0.5 else w_neg
            err = (p - yi) * cw
            for j in range(d):
                gw[j] += err * row[j]
            gb += err
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def _predict_logreg(w, b, X):
    out = []
    for row in X:
        z = max(-30.0, min(30.0, b + sum(w[j] * row[j] for j in range(len(w)))))
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def _oot_split(ts, val_frac=0.2):
    """Indices split strictly by time: latest `val_frac` = validation."""
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    cut = int(len(order) * (1 - val_frac))
    return set(order[:cut]), set(order[cut:])


# ══════════════════════════════════════════════════════════════════════════════
# Smoke pipeline — synthetic wallets w/ latent skill, run end-to-end on stdlib.
# ══════════════════════════════════════════════════════════════════════════════

def _synth(seed=11, n_wallets=80, n_markets=48, bars_per_market=44,
           trades_per_bar=5):
    """Synthetic tape where some wallets genuinely have edge.

    Each wallet gets a latent `skill`; skilled wallets trade toward each market's
    eventual resolution, so (a) they earn PnL in the pre-cutoff era → the
    leaderboard finds them, and (b) their net flow leads the forward move in the
    post-cutoff era → the features carry signal. Markets are laid out sequentially
    in time, so the pre-cutoff (cohort) markets and post-cutoff (prediction)
    markets are disjoint — the cleanest possible anti-leakage layout."""
    rng = random.Random(seed)
    wallets = [f"0x{i:040x}" for i in range(n_wallets)]
    skill = {w: rng.gauss(0.0, 1.0) for w in wallets}          # latent edge
    trades, markets = [], []
    base = 1_600_000_000
    span = bars_per_market * BAR_SECONDS
    for m in range(n_markets):
        mk = f"m{m}"
        drift = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.8)
        start = base + m * span
        final_yes = 1.0 if drift > 0 else 0.0
        for step in range(bars_per_market):
            frac = step / bars_per_market
            price = 0.5 + drift * frac + rng.gauss(0.0, 0.03)
            price = min(0.98, max(0.02, price))
            ts = start + step * BAR_SECONDS + rng.randint(0, BAR_SECONDS - 1)
            for _ in range(trades_per_bar):
                u = rng.choice(wallets)
                # skilled wallets lean toward `drift`; unskilled are noise
                bias = math.tanh(skill[u] * 1.5)
                p_buy = 0.5 + 0.45 * bias * (1.0 if drift > 0 else -1.0)
                side = 1.0 if rng.random() < p_buy else -1.0
                size = rng.uniform(50.0, 500.0)
                trades.append({
                    "timestamp": ts,
                    "market_id": mk,
                    "user": u,
                    "price": price,
                    "usd_amount": size * price,
                    "token_amount": side * size,
                })
        markets.append({"market_id": mk,
                        "outcome_prices": json.dumps([str(final_yes), str(1.0 - final_yes)])})
    return trades, markets


def _run_smoke():
    trades, market_rows = _synth()
    final_prices = _build_final_prices(market_rows)
    X, y, fwd, ts, meta = _build_dataset(
        trades, final_prices, min_volume=100.0)
    print(f"synthetic tape: {len(trades)} trades, {len(market_rows)} markets")
    print(f"leaderboard {meta['leaderboard_wallets']} wallets "
          f"({meta['qualifying_wallets']} qualifying) → "
          f"smart {meta['smart_wallets']} / dumb {meta['dumb_wallets']}")
    if len(y) < 40:
        raise SystemExit(f"too few windows built ({len(y)}) — synth params too thin")

    tr_i, va_i = _oot_split(ts, val_frac=0.25)
    Xtr = [X[i] for i in sorted(tr_i)]
    ytr = [y[i] for i in sorted(tr_i)]
    Xva = [X[i] for i in sorted(va_i)]
    yva = [y[i] for i in sorted(va_i)]
    fva = [fwd[i] for i in sorted(va_i)]

    mean, std = _standardize_fit(Xtr)
    w, b = _fit_logreg(_standardize_apply(Xtr, mean, std), ytr)
    pv = _predict_logreg(w, b, _standardize_apply(Xva, mean, std))

    auc = _auc(pv, yva)
    bt = _decile_backtest(pv, fva)
    base_acc = max(_mean(ytr), 1 - _mean(ytr))
    print(f"windows: train {len(ytr)} / val {len(yva)}  (val up-rate {_mean(yva):.3f})")
    print(f"majority baseline acc: {base_acc:.3f}")
    print(f"\n== smart-money signal (hand-rolled logistic, out-of-time) ==")
    print(f"val AUC        : {auc:.4f}")
    print(f"decile backtest: {json.dumps(bt)}")
    coefs = sorted(zip(FEATURE_NAMES, w), key=lambda t: -abs(t[1]))
    print(f"feature coefs  : {[(n, round(c, 3)) for n, c in coefs]}")
    if auc <= 0.5:
        raise SystemExit(f"smoke FAILED: AUC {auc:.4f} ≤ 0.5 — signal did not learn")
    print(f"\nsmoke OK — AUC {auc:.4f} > 0.5, spread {bt.get('up_rate_spread')}")
    return {"val_auc": round(auc, 4), "backtest": bt}


# ══════════════════════════════════════════════════════════════════════════════
# H100 training core (imports numpy / lightgbm INSIDE — never at module top).
# ══════════════════════════════════════════════════════════════════════════════

def _model_card(result):
    m = result.get("gbdt", {})
    return f"""---
license: mit
tags:
- polymarket
- prediction-markets
- smart-money
- order-flow
- lightgbm
---

# Polymarket Smart-Money Signal

A forward-return signal built from the **smart-money cohort's order flow**, trained
on the user-level tape of `{HF_REPO}` (`users.parquet`) on an H100 via Modal.

Unlike the other models in this suite (which read price/volume), this one reads
*who* is trading. It builds a per-wallet PnL leaderboard, freezes a top-cohort of
"smart" wallets from history, and turns their net flow into six features
({', '.join(FEATURE_NAMES)}).

## Headline (out-of-time)
- **val AUC**: `{m.get('val_auc')}`
- **decile backtest**: `{json.dumps(m.get('backtest', {}))}`
- windows: {result.get('train_windows')} train / {result.get('val_windows')} val
- smart wallets: {result.get('cohort', {}).get('smart_wallets')} / dumb: {result.get('cohort', {}).get('dumb_wallets')}

## Anti-leakage & survivorship
Cohorts are scored ONLY from trades **before** a temporal cutoff
(`cohort_cutoff_frac={COHORT_CUTOFF_FRAC}`); prediction windows come only from
**after** it, so a wallet's smart/dumb status is fixed from its past and never
informed by the window being predicted. A `min_volume` floor drops lucky-one-fill
wallets. The leaderboard is, by construction, survivorship-limited to wallets
active in the pre-cutoff era — reported, not hidden.

## Files
- `smartmoney_gbdt.txt` — LightGBM model
- `smartmoney_normalizer.json` — feature mean/std + config
- `metrics/smartmoney_metrics.json` — full report

*Signal for research, not financial advice.*
"""


def _train_gbdt(X, y, fwd, ts, meta):
    """LightGBM on the smart-money features with a strict out-of-time split."""
    import numpy as np

    Xa = np.asarray(X, dtype=np.float32)
    ya = np.asarray(y, dtype=np.float32)
    fa = np.asarray(fwd, dtype=np.float32)
    ta = np.asarray(ts, dtype=np.float64)

    order = np.argsort(ta, kind="mergesort")
    cut = int(len(order) * 0.8)
    tr_i, va_i = order[:cut], order[cut:]
    fmean = Xa[tr_i].mean(0)
    fstd = Xa[tr_i].std(0) + 1e-6

    result = {
        "runtime": "modal H100 / smart-money",
        "dataset": HF_REPO,
        "source": "users.parquet user-level tape → per-wallet PnL cohorts → smart-money flow",
        "features": FEATURE_NAMES,
        "window": WINDOW, "horizon": HORIZON, "bar_seconds": BAR_SECONDS,
        "windows": int(len(ya)),
        "train_windows": int(len(tr_i)), "val_windows": int(len(va_i)),
        "cohort": meta,
        "majority_baseline_acc": round(float(max(ya[tr_i].mean(), 1 - ya[tr_i].mean())), 4),
    }

    import lightgbm as lgb
    dtr = lgb.Dataset(Xa[tr_i], label=ya[tr_i], feature_name=list(FEATURE_NAMES))
    dva = lgb.Dataset(Xa[va_i], label=ya[va_i], reference=dtr)
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.02,
              "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "is_unbalance": True, "seed": 11, "verbose": -1}
    bst = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    gp = bst.predict(Xa[va_i])
    result["gbdt"] = {
        "val_auc": round(_auc(list(gp), list(ya[va_i])), 4),
        "brier": round(float(np.mean((gp - ya[va_i]) ** 2)), 4),
        "backtest": _decile_backtest(list(gp), list(fa[va_i])),
    }
    result["feature_importance_gbdt"] = sorted(
        zip(FEATURE_NAMES, [round(float(x), 1) for x in bst.feature_importance("gain")]),
        key=lambda t: -t[1])
    print(f"[gbdt] {result['gbdt']}", flush=True)

    # Walk-forward over global time — is the smart-money edge stable, not a fluke?
    wf = []
    for kf in range(1, 5):
        lo, hi = 0.2 * kf, 0.2 * (kf + 1)
        n = len(order)
        vmask = order[int(n * lo):int(n * hi)]
        tmask = order[:int(n * lo)]
        if len(vmask) < 40 or len(tmask) < 120:
            continue
        b = lgb.train(params, lgb.Dataset(Xa[tmask], label=ya[tmask]),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
        pv = b.predict(Xa[vmask])
        wf.append({"fold": kf, "val_auc": round(_auc(list(pv), list(ya[vmask])), 4),
                   "up_rate_spread": _decile_backtest(list(pv), list(fa[vmask])).get("up_rate_spread")})
    result["walk_forward"] = {"folds": wf,
                              "mean_auc": round(float(np.mean([f["val_auc"] for f in wf])), 4) if wf else None}
    print(f"[walk-forward] {result['walk_forward']}", flush=True)

    normalizer = {"fmean": fmean.tolist(), "fstd": fstd.tolist(),
                  "features": FEATURE_NAMES, "window": WINDOW, "horizon": HORIZON,
                  "bar_seconds": BAR_SECONDS, "cohort_top_frac": COHORT_TOP_FRAC,
                  "cohort_min_volume": COHORT_MIN_VOLUME,
                  "cohort_cutoff_frac": COHORT_CUTOFF_FRAC}
    return result, normalizer, bst


def _run_h100(max_trades: int, max_markets: int, push: bool, hf_token: str) -> dict:
    """The body of the Modal function — also importable for local orchestration."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    def _read(fname, columns):
        path = hf_hub_download(HF_REPO, fname, repo_type="dataset")
        tbl = pq.read_table(path, columns=columns)
        rows = tbl.to_pylist()
        os.remove(path)
        return rows

    market_rows = _read("markets.parquet", ["market_id", "outcome_prices"])
    if max_markets and len(market_rows) > max_markets:
        market_rows = market_rows[:max_markets]
    final_prices = _build_final_prices(market_rows)
    keep_markets = set(final_prices)
    print(f"markets: {len(market_rows)} rows, {len(final_prices)} resolved", flush=True)

    user_rows = _read("users.parquet",
                      ["timestamp", "market_id", "user", "price", "usd_amount", "token_amount"])
    print(f"user tape: {len(user_rows)} raw fills", flush=True)
    trades = []
    for r in user_rows:
        mk = r["market_id"]
        if keep_markets and mk not in keep_markets:
            continue
        try:
            trades.append({
                "timestamp": int(r["timestamp"]),
                "market_id": mk,
                "user": r["user"],
                "price": float(r["price"]),
                "usd_amount": float(r["usd_amount"]),
                "token_amount": float(r["token_amount"]),
            })
        except (TypeError, ValueError):
            continue
        if max_trades and len(trades) >= max_trades:
            break
    trades.sort(key=lambda t: t["timestamp"])
    print(f"usable trades: {len(trades)}", flush=True)

    X, y, fwd, ts, meta = _build_dataset(trades, final_prices)
    print(f"built {len(y)} windows; cohort meta {meta}", flush=True)
    if len(y) < 200:
        return {"error": f"too few windows ({len(y)}) — raise max_trades/max_markets", "cohort": meta}

    result, normalizer, bst = _train_gbdt(X, y, fwd, ts, meta)

    # Serialize artifacts and return them base64 for a local push (mirrors bigdata).
    import base64
    bst.save_model("/tmp/smartmoney_gbdt.txt")
    artifacts = {
        "smartmoney_gbdt.txt": open("/tmp/smartmoney_gbdt.txt", "rb").read(),
        "smartmoney_normalizer.json": json.dumps(normalizer, indent=2).encode(),
        "smartmoney_metrics.json": json.dumps(result, indent=2).encode(),
        "README.md": _model_card(result).encode(),
    }
    result["_artifacts_b64"] = {k: base64.b64encode(v).decode() for k, v in artifacts.items()}

    if push and hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=OUT_REPO, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=artifacts["smartmoney_gbdt.txt"],
                            path_in_repo="smartmoney_gbdt.txt", repo_id=OUT_REPO, repo_type="model")
            api.upload_file(path_or_fileobj=artifacts["smartmoney_normalizer.json"],
                            path_in_repo="smartmoney_normalizer.json", repo_id=OUT_REPO, repo_type="model")
            api.upload_file(path_or_fileobj=artifacts["smartmoney_metrics.json"],
                            path_in_repo="metrics/smartmoney_metrics.json", repo_id=OUT_REPO, repo_type="model")
            api.upload_file(path_or_fileobj=artifacts["README.md"],
                            path_in_repo="README.md", repo_id=OUT_REPO, repo_type="model")
            result["hf_repo"] = f"https://huggingface.co/{OUT_REPO}"
            print(f"pushed → {result['hf_repo']}", flush=True)
        except Exception as e:
            result["push_error"] = f"{type(e).__name__}: {e}"[:300]
            print("push failed:", result["push_error"], flush=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Modal app — built ONLY when modal is importable, so the module stays import-clean
# on a bare machine (recipe b). Matches modal_bigdata.py's structure.
# ══════════════════════════════════════════════════════════════════════════════
try:
    import modal  # noqa: F401
    _HAS_MODAL = True
except Exception:
    _HAS_MODAL = False

if _HAS_MODAL:
    app = modal.App("pmt-smartmoney")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("pyarrow", "numpy", "lightgbm", "huggingface_hub")
    )

    @app.function(image=image, gpu="H100", cpu=16.0, memory=131072, timeout=9000)
    def run(max_trades: int = 3_000_000, max_markets: int = 0,
            push: bool = False, hf_token: str = "") -> dict:
        return _run_h100(max_trades, max_markets, push, hf_token)

    @app.local_entrypoint()
    def main(max_trades: int = 3_000_000, max_markets: int = 0, push: bool = False):
        import base64
        token = os.environ.get("HF_TOKEN", "")
        report = run.remote(max_trades=max_trades, max_markets=max_markets,
                            push=push, hf_token=token)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        for name, b64 in report.pop("_artifacts_b64", {}).items():
            out_name = "smartmoney_card.md" if name == "README.md" else name
            with open(os.path.join(data_dir, out_name), "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"saved data/{out_name}")
        with open(os.path.join(data_dir, "smartmoney_metrics.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nwrote {data_dir}/smartmoney_metrics.json")


# ── CLI: --smoke runs the full pipeline on stdlib; no modal needed ─────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smart-money signal trainer")
    ap.add_argument("--smoke", action="store_true",
                    help="run the full synthetic pipeline on pure stdlib (no GPU/deps)")
    args = ap.parse_args()
    if args.smoke:
        _run_smoke()
    else:
        raise SystemExit(
            "This is a Modal job. Run the real training with:\n"
            "    modal run ml/modal_smart.py --push\n"
            "or the dependency-free smoke test with:\n"
            "    python ml/modal_smart.py --smoke")
