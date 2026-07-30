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

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IDS = os.path.join(DATA, "mega_call_ids.json")
APP, FN = "pmt-mega", "run"
RESOLUTIONS = [300, 900, 3600]
KW = dict(max_rows=30_000_000, top_tokens=3000, epochs=40)


def _collect(calls: dict) -> None:
    reports, best, best_auc, best_art = {}, None, -1.0, {}
    for bs, call in calls.items():
        rep = call.get()                          # blocks; server-side, survives client blips
        art = rep.pop("_artifacts_b64", {})
        auc = rep.get("ensemble", {}).get("val_auc", 0.0)
        sp = rep.get("ensemble", {}).get("backtest", {}).get("up_rate_spread")
        print(f"  bar_seconds={bs}: ensemble AUC {auc}  spread {sp}", flush=True)
        reports[str(bs)] = rep
        if auc > best_auc:
            best_auc, best, best_art = auc, str(bs), art
    os.makedirs(DATA, exist_ok=True)
    for name, b64 in best_art.items():
        with open(os.path.join(DATA, "mega_" + name.replace("/", "_")), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved data/mega_{name.replace('/', '_')}", flush=True)
    with open(os.path.join(DATA, "mega_parallel_metrics.json"), "w") as f:
        json.dump({"runtime": "modal parallel H100s / multi-resolution mega (deployed)",
                   "best_resolution": best, "best_ensemble_auc": best_auc,
                   "by_resolution": reports}, f, indent=2)
    print(f"\nbest resolution: bar_seconds={best} (ensemble AUC {best_auc})", flush=True)
    print(f"wrote {DATA}/mega_parallel_metrics.json", flush=True)


def main() -> None:
    fn = modal.Function.from_name(APP, FN)
    if "--collect" in sys.argv and os.path.exists(IDS):
        ids = json.load(open(IDS))
        calls = {int(bs): modal.FunctionCall.from_id(cid) for bs, cid in ids.items()}
        print(f"re-attached to {ids}", flush=True)
    else:
        hf = os.environ.get("HF_TOKEN", "")
        calls = {bs: fn.spawn(bar_seconds=bs, push=False, hf_token=hf, **KW) for bs in RESOLUTIONS}
        ids = {str(bs): c.object_id for bs, c in calls.items()}
        os.makedirs(DATA, exist_ok=True)
        json.dump(ids, open(IDS, "w"))
        print(f"spawned {len(calls)} H100 workers (deployed app), ids → {IDS}: {ids}", flush=True)
    _collect(calls)


if __name__ == "__main__":
    main()
