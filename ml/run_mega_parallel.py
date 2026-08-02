"""Robust parallel launcher for the mega trainer on parallel H100s.

The `modal run …::parallel` path dies when the local client's heartbeat drops
(ConnectionError: Deadline exceeded) — an ephemeral app is torn down with the
client, cancelling the spawned workers. This script instead spawns against the
already-**deployed** `pmt-mega` app (`modal deploy ml/modal_mega.py` first), so
the workers run fully server-side and survive any client disconnect.

It spawns one worker per bar resolution (5-min / 15-min / 1-hour), writes their
FunctionCall ids to `data/mega_call_ids.json` immediately (so a dead poller can
re-attach), then polls each `.get()`, saves the best resolution's artifacts and a
combined metrics file.

    ~/.local/share/uv/tools/modal/bin/python ml/run_mega_parallel.py

Re-attach after a poller death:  python ml/run_mega_parallel.py --collect
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

import modal

# Network errors that mean "the client's stream dropped" — retry, don't die. My
# local connection to Modal blips (StreamTerminatedError / Connection lost); the
# workers run server-side and results stay retrievable, so reconnecting is always
# safe. Matched by class-name substring so grpclib need not be importable.
_NET = ("StreamTerminated", "Connection", "Cancelled", "Broken", "Unavailable", "GOAWAY")


def _robust_get(cid, poll=30, backoff=5):
    """Block until this FunctionCall finishes, surviving client network drops by
    reconnecting. TimeoutError = still running; a network error = reconnect;
    anything else (a real worker exception) propagates."""
    while True:
        try:
            return modal.FunctionCall.from_id(cid).get(timeout=poll)
        except TimeoutError:
            continue
        except (ConnectionError, OSError):
            print(f"  net drop; reconnecting in {backoff}s…", flush=True)
            time.sleep(backoff)
        except Exception as e:  # noqa: BLE001
            if any(s in type(e).__name__ for s in _NET):
                print(f"  net drop ({type(e).__name__}); reconnecting in {backoff}s…", flush=True)
                time.sleep(backoff)
                continue
            raise

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IDS = os.path.join(DATA, "mega_call_ids.json")
APP, FN = "pmt-mega", "run"
# Probe the finer frontier — 5-min beat 15-min beat 1-hour monotonically, so test
# 1-min too. Finer bars explode the window count, so cap tokens tighter the finer
# we go (keeps each worker's in-memory dataset bounded).
RESOLUTIONS = [60, 300, 900]
TOP_TOKENS = {60: 1200, 300: 2500, 900: 3500}
KW = dict(max_rows=30_000_000, epochs=40)


def _collect(ids: dict) -> None:
    reports, best, best_score, best_auc, best_art = {}, None, -1.0, -1.0, {}
    for bs, cid in ids.items():
        # One crashed worker must not sink the whole run: a real worker exception
        # (including one whose remote exception can't even be deserialized locally)
        # is caught here so the surviving resolutions are still collected, saved and
        # ranked. _robust_get already retries transient network drops; only genuine
        # failures reach this except.
        try:
            rep = _robust_get(cid)
        except Exception as e:  # noqa: BLE001
            reports[str(bs)] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  bar_seconds={bs}: FAILED — {type(e).__name__} (skipped)", flush=True)
            continue
        art = rep.pop("_artifacts_b64", {})
        auc = rep.get("ensemble", {}).get("val_auc", 0.0) or 0.0
        sp = rep.get("ensemble", {}).get("backtest", {}).get("up_rate_spread")
        # Rank on a stability-weighted score (matches modal_mega.parallel): half the
        # slice ensemble AUC, half the walk-forward mean, so the shipped resolution
        # holds up across time rather than winning one lucky slice.
        wf = rep.get("walk_forward", {}).get("mean_auc")
        score = 0.5 * auc + 0.5 * (wf if wf is not None else auc)
        print(f"  bar_seconds={bs}: ensemble AUC {auc}  walk-fwd {wf}  "
              f"score {score:.4f}  spread {sp}", flush=True)
        reports[str(bs)] = rep
        if score > best_score:
            best_score, best_auc, best, best_art = score, auc, str(bs), art
    if best is None:
        print("all resolutions failed — nothing to save", flush=True)
    os.makedirs(DATA, exist_ok=True)
    for name, b64 in best_art.items():
        with open(os.path.join(DATA, "mega_" + name.replace("/", "_")), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved data/mega_{name.replace('/', '_')}", flush=True)
    with open(os.path.join(DATA, "mega_parallel_metrics.json"), "w") as f:
        json.dump({"runtime": "modal parallel H100s / multi-resolution mega (deployed)",
                   "best_resolution": best, "best_ensemble_auc": best_auc,
                   "best_selection_score": round(best_score, 4) if best is not None else None,
                   "selection": "0.5*ensemble_val_auc + 0.5*walk_forward_mean_auc",
                   "failed_resolutions": [k for k, v in reports.items() if "error" in v],
                   "by_resolution": reports}, f, indent=2, default=lambda o: o.tolist())
    print(f"\nbest resolution: bar_seconds={best} (ensemble AUC {best_auc})", flush=True)
    print(f"wrote {DATA}/mega_parallel_metrics.json", flush=True)


def main() -> None:
    fn = modal.Function.from_name(APP, FN)
    if "--collect" in sys.argv and os.path.exists(IDS):
        ids = json.load(open(IDS))
        print(f"re-attached to {ids}", flush=True)
    else:
        hf = os.environ.get("HF_TOKEN", "")
        calls = {bs: fn.spawn(bar_seconds=bs, top_tokens=TOP_TOKENS[bs], push=False, hf_token=hf, **KW)
                 for bs in RESOLUTIONS}
        ids = {str(bs): c.object_id for bs, c in calls.items()}
        os.makedirs(DATA, exist_ok=True)
        json.dump(ids, open(IDS, "w"))
        print(f"spawned {len(calls)} H100 workers (deployed app), ids → {IDS}: {ids}", flush=True)
    _collect(ids)


if __name__ == "__main__":
    main()
