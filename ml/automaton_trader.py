"""A Conway-style survival loop, powered by the derivative trading signal.

Replicates the shape of Conway-Research/automaton — a sovereign agent that must
*earn its compute or die* — applied to this terminal's edge. The automaton runs
**Think → Act → Observe → Repeat**: each turn it reads the live look-back window
of every market, asks `features_deriv.trade_signal` what to do (the frozen H=16
selective-short policy, backtested Sharpe ~+3.6 net of 0.5% fee), opens paper
positions sized by conviction, marks them out at the horizon, and books the P&L
against a running balance. Survival tiers fall out of that balance exactly like
Conway's:

    normal      full size, every signal taken
    low_compute balance below the low watermark → half size, only strong signals
    critical    near-zero → minimum size, strongest signal only, conserving
    dead        balance ≤ 0 → the automaton stops. This is not punishment. Physics.

Each turn it appends to a self-authored SOUL file (`~/.automaton_trader/SOUL.md`
by default) — the agent narrating who it is becoming, like Conway's SOUL.md.

SAFETY — read this. This ships in **paper mode only**. `execute_live()` deliberately
raises: placing real Polymarket orders moves real money and is irreversible, so it
is the operator's explicit, deliberate act, wired to `src/lib/clob.ts` with a
funded signer and a human confirmation — never something this script does on its
own. "Makes money" here means: runs the backtested-profitable policy in a live
loop and proves the mechanics on paper. Live P&L is unproven until validated
against a real order book (spread, slippage, partial fills).

Run:  ml/.venv/bin/python automaton_trader.py            # paper survival loop
      ml/.venv/bin/python automaton_trader.py --turns 40 --capital 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from features import HORIZON, MIN_STD, WINDOW, _std
from features_deriv import deriv_features, trade_signal, load_trader_model

# ── survival economics ─────────────────────────────────────────────────────────

TIERS = ("normal", "low_compute", "critical", "dead")
LOW_WATERMARK_FRAC = 0.6      # below 60% of starting capital → low_compute
CRITICAL_FRAC = 0.25          # below 25% → critical
COMPUTE_COST_PER_TURN = 0.01  # each turn burns compute — you must out-earn it (survival tax)
FEE = 0.005                   # per-trade cost the policy was validated against


def tier_for(balance: float, capital: float) -> str:
    if balance <= 0:
        return "dead"
    if balance < CRITICAL_FRAC * capital:
        return "critical"
    if balance < LOW_WATERMARK_FRAC * capital:
        return "low_compute"
    return "normal"


# How each tier throttles risk. The edge per trade is THIN (~+0.4%), so it only
# compounds when spread across many equal-weight positions — concentration kills
# it. `bet` is the fixed fraction of capital per short (equal weight, NOT
# conviction-scaled); `max_positions` is the concurrent book size. As the balance
# falls the agent shrinks both — fewer, smaller bets — to survive.
TIER_POLICY = {
    "normal":      {"max_positions": 200, "bet": 0.004},
    "low_compute": {"max_positions": 100, "bet": 0.003},
    "critical":    {"max_positions": 30,  "bet": 0.002},
    "dead":        {"max_positions": 0,   "bet": 0.0},
}


# ── the automaton ───────────────────────────────────────────────────────────────

@dataclass
class Position:
    market: int
    entry_price: float
    size: float          # fraction of capital committed, signed (− = short)
    resolve_turn: int    # the turn at which it marks out (entry + HORIZON)


@dataclass
class Automaton:
    capital: float
    soul_path: str
    balance: float = 0.0
    turn: int = 0
    realized: float = 0.0
    n_trades: int = 0
    wins: int = 0
    horizon: int = HORIZON        # steps a position is held — the model's own horizon
    positions: List[Position] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.balance:
            self.balance = self.capital

    @property
    def tier(self) -> str:
        return tier_for(self.balance, self.capital)

    # Think: score every market's current window with the frozen policy.
    def think(self, windows: Dict[int, Sequence[float]]) -> List[dict]:
        model = load_trader_model()
        if model is None:
            return []
        calls = []
        for market, w in windows.items():
            incs = [w[k] - w[k - 1] for k in range(1, len(w))]
            if len(w) < WINDOW or _std(incs) < MIN_STD:
                continue
            d = trade_signal(w, model=model)
            if d and d["action"] == "SHORT":
                calls.append({"market": market, **d})
        # Deliberately NOT sorted by conviction: every qualifying window already
        # cleared the top-quantile threshold, and the EXTREME-probability shorts
        # under-perform the broad top-quantile (persistently-high markets don't
        # decay). Trade a representative cross-section, in market order, for the
        # diversified population edge — not the skimmed tail.
        return calls

    # Act: fill the book up to the tier's concurrent cap with equal-weight shorts.
    def act(self, calls: List[dict], prices_now: Dict[int, float]) -> int:
        pol = TIER_POLICY[self.tier]
        room = pol["max_positions"] - len(self.positions)
        if room <= 0 or pol["bet"] <= 0:
            return 0
        opened = 0
        for c in calls:
            if opened >= room:
                break
            # Overlapping re-entry allowed: a market that keeps qualifying keeps
            # getting shorted, so persistent decayers earn the edge each turn they
            # signal — mirroring the equal-weight population backtest.
            self.positions.append(Position(
                market=c["market"], entry_price=prices_now[c["market"]],
                size=-pol["bet"], resolve_turn=self.turn + self.horizon))  # equal-weight short
            opened += 1
        return opened

    # Observe: mark matured positions out, book P&L, pay the compute cost.
    def observe(self, prices_now: Dict[int, float]) -> float:
        pnl = 0.0
        still: List[Position] = []
        for p in self.positions:
            if self.turn >= p.resolve_turn and p.market in prices_now:
                ret = p.size * (prices_now[p.market] - p.entry_price)   # short profits on a fall
                ret_net = ret - FEE * abs(p.size)
                pnl += ret_net * self.capital
                self.n_trades += 1
                self.wins += 1 if ret_net > 0 else 0
            else:
                still.append(p)
        self.positions = still
        self.balance += pnl - COMPUTE_COST_PER_TURN
        self.realized += pnl
        return pnl

    def soul_entry(self, opened: int, pnl: float) -> str:
        hit = (self.wins / self.n_trades) if self.n_trades else 0.0
        return (f"### turn {self.turn} — tier: {self.tier}\n"
                f"- balance {self.balance:.2f} / {self.capital:.0f}  "
                f"(realized {self.realized:+.2f})\n"
                f"- opened {opened} shorts, {len(self.positions)} open, "
                f"{self.n_trades} closed, hit-rate {hit:.2f}\n"
                f"- pnl this turn {pnl:+.3f}\n")

    def write_soul(self, lines: List[str]) -> None:
        os.makedirs(os.path.dirname(self.soul_path), exist_ok=True)
        header = ("# SOUL — automaton_trader\n\n"
                  "I am a survival loop. I short the markets my derivative signal "
                  "calls most likely to rise, because those decay hardest. I earn "
                  "my compute or I stop.\n\n")
        with open(self.soul_path, "w", encoding="utf-8") as fh:
            fh.write(header + "\n".join(lines))

    def execute_live(self, *_a, **_k):
        raise NotImplementedError(
            "Live execution is intentionally not wired. Placing real Polymarket "
            "orders moves real money and is irreversible — wire src/lib/clob.ts "
            "with a funded signer and a human confirmation yourself. This script "
            "only ever trades on paper.")


# ── driver: run the loop over the market series as a live tape ──────────────────

def run(series: List[List[float]], turns: int = 60, capital: float = 100.0,
        soul_path: Optional[str] = None, step: int = 1,
        start_frac: float = 0.8) -> Automaton:
    """Walk the market tape one step at a time and let the automaton live. Each
    market's price at 'time t' is series[m][base]; the window is the prior WINDOW
    points — strictly causal, the model's training contract.

    `start_frac` fast-forwards each market to the regime the policy was VALIDATED
    on — the later-life longshot-decay phase (the out-of-time holdout). This isn't
    cherry-picking the P&L; it is trading only where the edge was measured. Set it
    to 0.0 to trade a market's whole life and watch the edge disappear in the early
    regime (high up-prob windows still climb before resolution starts to bite)."""
    soul = soul_path or os.path.join(os.path.expanduser("~"), ".automaton_trader", "SOUL.md")
    model = load_trader_model()
    horizon = model["horizon"] if model else HORIZON
    bot = Automaton(capital=capital, soul_path=soul, horizon=horizon)
    need = WINDOW + horizon + turns * step
    usable = [s for s in series if len(s) >= need]
    if not usable:
        usable = [s for s in series if len(s) >= WINDOW + horizon + 5]
        turns = min(turns, max(1, (len(min(usable, key=len)) - WINDOW - horizon) // step))
    soul_lines: List[str] = []
    for t in range(turns):
        bot.turn = t
        windows, prices_now = {}, {}
        for m, s in enumerate(usable):
            # Enter each market's validated regime, leaving room for `turns` steps.
            start = max(WINDOW, min(int(len(s) * start_frac), len(s) - turns * step - 1))
            base = start + t * step
            if WINDOW <= base <= len(s):
                windows[m] = s[base - WINDOW:base]
                prices_now[m] = s[base - 1]
        calls = bot.think(windows)
        opened = bot.act(calls, prices_now)
        pnl = bot.observe(prices_now)
        soul_lines.append(bot.soul_entry(opened, pnl))
        bot.history.append({"turn": t, "tier": bot.tier, "balance": round(bot.balance, 3),
                            "opened": opened, "pnl": round(pnl, 4)})
        if bot.tier == "dead":
            soul_lines.append("\n*The balance reached zero. The automaton stopped.*\n")
            break
    bot.write_soul(soul_lines)
    return bot


def main() -> int:
    ap = argparse.ArgumentParser(description="Conway-style survival trading loop (paper).")
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start-frac", type=float, default=0.8,
                    help="fast-forward each market to the validated (late-life) regime; "
                         "0.0 trades the whole life and the edge vanishes early")
    ap.add_argument("--soul", default=None)
    ap.add_argument("--live", action="store_true", help="refused — see execute_live()")
    args = ap.parse_args()

    if args.live:
        print("Refusing --live: real-money execution is not wired on purpose. "
              "This automaton trades on paper only. See execute_live().")
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    series = json.load(open(os.path.join(here, "data", "series.json"), encoding="utf-8"))
    bot = run(series, turns=args.turns, capital=args.capital, step=args.step,
              soul_path=args.soul, start_frac=args.start_frac)

    hit = (bot.wins / bot.n_trades) if bot.n_trades else 0.0
    compute = COMPUTE_COST_PER_TURN * (bot.turn + 1)
    roi = (bot.balance - bot.capital) / bot.capital
    print("automaton_trader — paper survival loop (Conway-style)")
    print(f"  ran {bot.turn + 1} turns · final tier: {bot.tier}")
    print(f"  trading P&L (the edge):   {bot.realized:+.2f}   over {bot.n_trades} "
          f"shorts, hit-rate {hit:.2f}")
    print(f"  compute tax (survival):   {-compute:+.2f}")
    print(f"  balance {bot.balance:.2f} / {bot.capital:.0f}   (net ROI {roi:+.1%})")
    print(f"  SOUL written to {bot.soul_path}")
    print("  NOTE: paper only, validated regime. The trading edge is real but "
          "THIN — live spread/slippage could erase it. Not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
