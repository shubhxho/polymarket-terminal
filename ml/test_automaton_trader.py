"""Stdlib tests for the Conway-style survival trading loop — no GPU, no network.

Run:  python ml/test_automaton_trader.py

Covers the survival economics (tiers by balance, risk throttled as death nears),
the short-P&L bookkeeping (a short profits on a fall, fees always subtracted), and
the safety guarantee that live execution is refused.
"""

from __future__ import annotations

from automaton_trader import (
    Automaton,
    Position,
    TIER_POLICY,
    tier_for,
)


def test_tiers_by_balance():
    cap = 100.0
    assert tier_for(90, cap) == "normal"
    assert tier_for(50, cap) == "low_compute"     # below 60%
    assert tier_for(20, cap) == "critical"        # below 25%
    assert tier_for(0, cap) == "dead"
    assert tier_for(-5, cap) == "dead"


def test_risk_shrinks_toward_death():
    order = ["normal", "low_compute", "critical", "dead"]
    caps = [TIER_POLICY[t]["max_positions"] for t in order]
    bets = [TIER_POLICY[t]["bet"] for t in order]
    assert caps == sorted(caps, reverse=True)      # fewer positions as it worsens
    assert bets == sorted(bets, reverse=True)       # smaller bets as it worsens
    assert TIER_POLICY["dead"]["max_positions"] == 0


def test_short_pnl_sign_and_fee():
    bot = Automaton(capital=100.0, soul_path="/tmp/_at_test_soul.md", horizon=1)
    # Open a short at 0.60; next turn price falls to 0.50 → the short wins.
    bot.turn = 0
    bot.positions = [Position(market=0, entry_price=0.60, size=-0.1, resolve_turn=1)]
    bot.turn = 1
    pnl = bot.observe({0: 0.50})
    # gross = -0.1 * (0.50 - 0.60) = +0.01 ; net = 0.01 - 0.005*0.1 = 0.0095 ; ×capital
    assert abs(pnl - 0.0095 * 100.0) < 1e-9, pnl
    assert bot.wins == 1 and bot.n_trades == 1
    # A rise would have lost.
    bot2 = Automaton(capital=100.0, soul_path="/tmp/_at_test_soul.md", horizon=1)
    bot2.positions = [Position(market=0, entry_price=0.60, size=-0.1, resolve_turn=1)]
    bot2.turn = 1
    assert bot2.observe({0: 0.70}) < 0


def test_dead_automaton_takes_no_risk():
    bot = Automaton(capital=100.0, soul_path="/tmp/_at_test_soul.md")
    bot.balance = 0.0                              # ran out of compute mid-loop
    assert bot.tier == "dead"
    opened = bot.act([{"market": 0, "action": "SHORT", "size": 0.9}], {0: 0.5})
    assert opened == 0 and bot.positions == []


def test_live_execution_is_refused():
    bot = Automaton(capital=100.0, soul_path="/tmp/_at_test_soul.md")
    raised = False
    try:
        bot.execute_live()
    except NotImplementedError:
        raised = True
    assert raised, "live execution must refuse"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
