"""Stdlib tests for the end-to-end pipeline orchestrator (`run_all.py`).

No pytest, no numpy — just `unittest`. Covers:

  * the orchestration ORDER is exactly data → features → train → evaluate → signal;
  * `--dry-run` prints all five stages and performs NO side effects;
  * a stage degrades gracefully (returns SKIPPED, never raises) when its
    dependency / artifact is absent;
  * a `--sample` smoke run exits 0 and produces the expected artifact set — or,
    on a box without mlx / network, honestly logs the skip for the missing piece.

Run:  python ml/test_run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_all  # noqa: E402

RUN_ALL = os.path.join(HERE, "run_all.py")
EXPECTED_ORDER = ["data", "features", "train", "evaluate", "signal"]


def _dir_snapshot(path):
    """Map basename -> mtime_ns for every file in `path` (or None if absent)."""
    if not os.path.isdir(path):
        return None
    return {n: os.stat(os.path.join(path, n)).st_mtime_ns for n in sorted(os.listdir(path))}


class TestOrder(unittest.TestCase):
    def test_stage_names_and_order(self):
        self.assertEqual(run_all.STAGES, EXPECTED_ORDER)

    def test_pipeline_registry_aligns(self):
        # PIPELINE is the single source of truth; its names must match STAGES in
        # order, and each entry must expose (name, headline, runner, artifacts).
        names = [row[0] for row in run_all.PIPELINE]
        self.assertEqual(names, EXPECTED_ORDER)
        for name, headline, runner, arts in run_all.PIPELINE:
            self.assertTrue(callable(runner), f"{name} runner not callable")
            self.assertIsInstance(headline, str)
            self.assertIsInstance(arts, list)


class TestDryRun(unittest.TestCase):
    def test_prints_all_five_stages_no_side_effects(self):
        data_dir = os.path.join(HERE, "data")
        before = _dir_snapshot(data_dir)

        proc = subprocess.run(
            [sys.executable, RUN_ALL, "--dry-run"],
            cwd=HERE, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout

        # Every stage is named in the printed plan, in order. Match the numbered
        # plan lines ("1. data", …) so the word "signal" in the header banner
        # can't be mistaken for the stage entry.
        last = -1
        for i, stage in enumerate(EXPECTED_ORDER, 1):
            idx = out.find(f"{i}. {stage}")
            self.assertGreater(idx, last, f"stage '{stage}' missing/out-of-order in plan")
            last = idx
        self.assertIn("no side effects", out.lower())

        # No side effects: the data dir is byte-for-byte unchanged (or still absent).
        self.assertEqual(_dir_snapshot(data_dir), before, "dry-run touched the data dir")


class TestGracefulSkip(unittest.TestCase):
    def test_train_skips_when_mlx_absent(self):
        # A bogus interpreter makes the mlx import-probe fail — the train stage
        # must skip gracefully, not raise or crash the run.
        ctx = run_all.Ctx(python="/nonexistent/python/interpreter")
        res = run_all.run_train(ctx)
        self.assertEqual(res.status, run_all.SKIPPED)
        self.assertIn("mlx", res.detail.lower())

    def test_stage_skips_when_script_missing(self):
        # Point the run at an empty dir with none of the merged modules present.
        with tempfile.TemporaryDirectory() as tmp:
            ctx = run_all.Ctx(here=tmp)
            for runner, label in (
                (run_all.run_data, "data"),
                (run_all.run_train, "train"),
                (run_all.run_evaluate, "evaluate"),
            ):
                res = runner(ctx)
                self.assertEqual(res.status, run_all.SKIPPED, f"{label} should skip")
                self.assertEqual(res.artifacts, [])

    def test_pipeline_skip_flag(self):
        # --skip removes a stage from the plan without erroring.
        proc = subprocess.run(
            [sys.executable, RUN_ALL, "--dry-run", "--skip", "train"],
            cwd=HERE, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skipped by --skip", proc.stdout)

    def test_unknown_skip_is_rejected(self):
        proc = subprocess.run(
            [sys.executable, RUN_ALL, "--dry-run", "--skip", "bogus"],
            cwd=HERE, capture_output=True, text=True, timeout=120,
        )
        self.assertNotEqual(proc.returncode, 0)


class TestSampleSmoke(unittest.TestCase):
    def test_sample_run_produces_artifacts_or_logs_skip(self):
        proc = subprocess.run(
            [sys.executable, RUN_ALL, "--sample"],
            cwd=HERE, capture_output=True, text=True, timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = proc.stdout

        # All five stage banners were emitted, in order.
        last = -1
        for stage in EXPECTED_ORDER:
            idx = out.find(stage.upper())
            self.assertGreater(idx, last, f"stage banner '{stage}' missing/out-of-order")
            last = idx

        self.assertIn("PIPELINE SUMMARY", out)
        self.assertIn("Artifacts produced", out)
        # The signal stage prints a tradeable-call table.
        self.assertIn("top", out.lower())
        self.assertTrue(
            any(call in out for call in ("BUY_YES", "BUY_NO", "HOLD")),
            "signal stage printed no BUY/HOLD calls",
        )

        # Each artifact-producing stage either wrote its file OR honestly logged a
        # skip/failure for it — never silently vanished.
        data_dir = os.path.join(HERE, "data")
        stage_status = self._parse_summary(out)
        for name, _headline, _runner, arts in run_all.PIPELINE:
            if not arts:
                continue
            status = stage_status.get(name)
            self.assertIsNotNone(status, f"stage '{name}' absent from summary")
            if status == run_all.RAN:
                present = [a for a in arts if os.path.exists(os.path.join(data_dir, a))]
                self.assertTrue(
                    present, f"stage '{name}' ran but produced none of {arts}",
                )
            else:
                # A skip/failure is acceptable (no mlx / no network) as long as it
                # was reported — which the summary line we just read proves.
                self.assertIn(status, (run_all.SKIPPED, run_all.FAILED))

    @staticmethod
    def _parse_summary(out):
        """Extract {stage: status} from the PIPELINE SUMMARY block."""
        status = {}
        in_summary = False
        for line in out.splitlines():
            if "PIPELINE SUMMARY" in line:
                in_summary = True
                continue
            if in_summary:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in EXPECTED_ORDER:
                    status[parts[0]] = parts[1]
        return status


if __name__ == "__main__":
    unittest.main(verbosity=2)
