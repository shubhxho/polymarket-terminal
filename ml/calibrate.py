"""Calibrate the shipped model's probabilities, then bake the result into the
web bundle's `mlModel.json`.

A 0.65-AUC ranker can still be badly *calibrated*: it may say 80% when the true
rate is 60%, which is exactly the number the terminal shows a trader. This fits a
**temperature** T on the strictly out-of-time validation split — the same
`_split` the model was validated on — by minimising validation NLL, then divides
every logit by T before the sigmoid.

Temperature scaling is deliberate over the more flexible Platt (scale + shift):
with shift fixed at zero the 0.5 decision boundary is preserved, so calibration
never flips the model's direction or changes its AUC — it only makes the
confidence honest. Reliability (ECE) and Brier are reported before and after so
the improvement is auditable, not asserted.

Run:  ml/.venv/bin/python ml/calibrate.py
"""

from __future__ import annotations

import json
import math
import os

import mlx.core as mx
from mlx.utils import tree_unflatten

from features import FEATURE_NAMES
from train_seq import SEED, FeatureMLP, _split, _tensors

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(os.path.dirname(HERE), "src", "lib", "mlModel.json")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _nll(logits: list[float], labels: list[float], t: float) -> float:
    """Mean binary cross-entropy of the temperature-scaled logits."""
    total = 0.0
    for z, y in zip(logits, labels):
        p = min(max(_sigmoid(z / t), 1e-7), 1 - 1e-7)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(logits)


def _fit_temperature(logits: list[float], labels: list[float]) -> float:
    """Golden-section search for the T that minimises validation NLL.

    NLL in T is unimodal for a fixed set of logits, so a bracketed 1-D search is
    exact to tolerance and needs no gradients.
    """
    lo, hi = 0.25, 6.0
    gr = (math.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = _nll(logits, labels, c), _nll(logits, labels, d)
    for _ in range(80):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = _nll(logits, labels, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = _nll(logits, labels, d)
    return round((lo + hi) / 2, 5)


def _ece(probs: list[float], labels: list[float], bins: int = 10) -> float:
    """Expected calibration error: mean |confidence − accuracy| over equal-width
    probability bins, weighted by bin population."""
    tot = [0] * bins
    hit = [0.0] * bins
    conf = [0.0] * bins
    for p, y in zip(probs, labels):
        b = min(bins - 1, int(p * bins))
        tot[b] += 1
        hit[b] += y
        conf[b] += p
    n = len(probs)
    e = 0.0
    for b in range(bins):
        if tot[b]:
            acc = hit[b] / tot[b]
            cf = conf[b] / tot[b]
            e += (tot[b] / n) * abs(cf - acc)
    return e


def _brier(probs: list[float], labels: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def _reliability_curve(probs: list[float], labels: list[float], bins: int = 10) -> list[dict]:
    """Per-bin (mean confidence, empirical accuracy, count) — the points a
    reliability diagram plots against the diagonal. Only populated bins are
    emitted so the web side can render without null-handling."""
    acc = [[0.0, 0.0, 0] for _ in range(bins)]  # sum(conf), sum(label), n
    for p, y in zip(probs, labels):
        b = min(bins - 1, int(p * bins))
        acc[b][0] += p
        acc[b][1] += y
        acc[b][2] += 1
    return [
        {"conf": round(s / n, 4), "acc": round(h / n, 4), "n": n}
        for s, h, n in acc
        if n > 0
    ]


def main() -> None:
    mx.random.seed(SEED)
    with open(os.path.join(DATA, "series.json")) as f:
        series = json.load(f)
    _, va = _split(series)
    if not va:
        raise SystemExit("empty validation split")

    norm = json.load(open(os.path.join(DATA, "seq_normalizer.json")))
    fmean = mx.array(norm["fmean"])
    fstd = mx.array(norm["fstd"])
    rstd = float(norm["rstd"])

    model = FeatureMLP(len(FEATURE_NAMES))
    model.update(tree_unflatten(list(mx.load(os.path.join(DATA, "seq_model.safetensors")).items())))
    model.eval()

    seq, feat, y = _tensors(va, fmean, fstd, rstd)
    logits = model(seq, feat).reshape(-1).tolist()
    labels = y.tolist()

    # Deployed temperature: fit on all of validation (1 parameter, so no
    # meaningful overfit from re-using the same set).
    t = _fit_temperature(logits, labels)
    raw = [_sigmoid(z) for z in logits]
    cal = [_sigmoid(z / t) for z in logits]

    # Honest held-out ECE: fit the temperature on the first time-half of
    # validation and score it on the second. `ece_after` on the full set can
    # only flatter the fit; this is the number to trust — and it is reported
    # alongside so the two can be compared.
    mid = len(logits) // 2
    t_held = _fit_temperature(logits[:mid], labels[:mid])
    ece_heldout = _ece([_sigmoid(z / t_held) for z in logits[mid:]], labels[mid:])

    report = {
        "temperature": t,
        "val_n": len(labels),
        "ece_before": round(_ece(raw, labels), 5),
        "ece_after": round(_ece(cal, labels), 5),
        "ece_heldout": round(ece_heldout, 5),
        "brier_before": round(_brier(raw, labels), 5),
        "brier_after": round(_brier(cal, labels), 5),
        "reliability": _reliability_curve(cal, labels),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "reliability"}, indent=2))
    print(f"reliability bins: {len(report['reliability'])}")

    # Merge temperature + report into the exported model the web bundle loads.
    doc = json.load(open(OUT))
    doc["temperature"] = t
    doc["calibration"] = report
    with open(OUT, "w") as f:
        json.dump(doc, f)
    print(f"wrote temperature {t} to {OUT}")


if __name__ == "__main__":
    main()
