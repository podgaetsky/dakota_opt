"""Opt-in local Dakota 6.23 workflow checks (no Slurm needed)."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dakota_checks import check_dakota

ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(os.environ.get("DAKOTA_INTEGRATION") == "1",
                     "Set DAKOTA_INTEGRATION=1 and DAKOTA to enable local Dakota checks")
class DakotaIntegrationTests(unittest.TestCase):
    def test_file_benchmark_optimization(self):
        dakota = os.environ.get("DAKOTA", "dakota")
        with tempfile.TemporaryDirectory(prefix="dakota-opt-integration-") as directory:
            subprocess.run([sys.executable, str(ROOT / "tool.py"), "opt",
                            str(ROOT / "templates/file_benchmark.json"), "--dakota", dakota,
                            "--workdir", directory], cwd=ROOT, check=True, capture_output=True, text=True)
            runs = list((Path(directory) / "runs").glob("*_opt"))
            self.assertEqual(len(runs), 1)
            run = runs[0]
            with (run / "evaluations.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            best = json.loads((run / "best.json").read_text())
            self.assertAlmostEqual(best["objective"], .02921, delta=.005)
            for name in ("dakota.rst", "report.html", "provenance.json"):
                self.assertTrue((run / name).is_file(), name)

    def test_queso_method_is_checked_not_assumed(self):
        result = check_dakota(os.environ.get("DAKOTA", "dakota"), os.environ.copy(), sys.executable)
        self.assertIn("Dakota version 6.23", result["version"])
        self.assertIn(result["queso"], ("available", "unavailable"), result["queso_detail"])
        if os.environ.get("DAKOTA_EXPECT_QUESO") == "1":
            self.assertEqual(result["queso"], "available", result["queso_detail"])
