import tempfile
import unittest
import json
import subprocess
import sys
import csv
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from driver import curve, parse_parameters, rms, run_job
from plot_results import fit_diagnostic_gp, main as plot_main
from run import input_text


class WorkflowTests(unittest.TestCase):
    def test_physical_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "params.in"
            path.write_text("2 variables\n0.5 x_p1\n0 x_p2\n1 functions\n1 ASV_1\n")
            specs = {"p1": {"lower": 0.1, "upper": 3.0},
                     "p2": {"lower": 0.01, "upper": 2.0}}
            result = parse_parameters(path, specs)
            self.assertAlmostEqual(result["p1"], 1.55)
            self.assertAlmostEqual(result["p2"], .01)
            path.write_text("2 variables\nnan x_p1\n0 x_p2\n")
            with self.assertRaises(ValueError):
                parse_parameters(path, specs)

    def test_curve_rejects_mismatched_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "curve.csv"
            path.write_text("x,y\n0,1\n1,2\n")
            self.assertAlmostEqual(rms(curve(path), curve(path)), 0)
            with self.assertRaises(ValueError):
                rms(curve(path), [(0, 1), (2, 2)])
            path.write_text("x,y\n0,1\n1,nan\n")
            with self.assertRaises(ValueError):
                curve(path)

    def test_dakota_budget(self):
        self.assertIn("max_iterations = 80", input_text("bo", Path("/tmp/example")))
        deck = input_text("opt", Path("/tmp/example"))
        self.assertIn("evaluation_concurrency = 8", deck)
        self.assertIn(str(Path(__file__).resolve().parent.parent / "app" / "driver.py"), deck)

    def test_failed_evaluation_returns_penalty_and_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            work = run / "params" / "eval.1"
            work.mkdir(parents=True)
            (run / "logs").mkdir()
            (run / "config.json").write_text(json.dumps({
                "parameters": {"p1": {"lower": 0.1, "upper": 3.0}},
                "backend": "local", "failure_penalty": 123456, "kind": "opt",
                "bo_initial_samples": 2}))
            (work / "params.in").write_text("1 variables\n0.5 x_p1\n1 functions\n1 ASV_1\n")
            # Missing reference.csv must be penalized, not silently accepted.
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "app" / "driver.py"),
                            "params.in", "results.out"], cwd=work, check=True)
            self.assertEqual((work / "results.out").read_text().strip(), "123456")
            self.assertTrue((run / "logs" / "eval.1.error.txt").exists())
            self.assertEqual((work / "diagnostic.json").exists(), True)

    def test_allocation_step_requests_one_evaluation_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"SLURM_JOB_ID": "42"}):
                with patch("driver.subprocess.run") as launch:
                    job = run_job(Path(temporary), {"backend": "allocation",
                                   "cpus_per_evaluation": 4, "job_timeout_seconds": 100})
            command = launch.call_args.args[0]
            self.assertEqual(job, "42 (allocation step)")
            self.assertIn("--ntasks=1", command)
            self.assertIn("--cpus-per-task=4", command)
            self.assertIn("--exclusive", command)

    def test_deterministic_diagnostic_gp_is_not_flat(self):
        grid = np.array([[i / 4, j / 4] for i in range(5) for j in range(5)])
        values = np.sin(2 * grid[:, 0]) + grid[:, 1] ** 2
        gp = fit_diagnostic_gp(grid, values)
        predicted, uncertainty = gp.predict(grid, return_std=True)
        self.assertLess(np.max(np.abs(predicted - values)), .02)
        self.assertLess(np.max(uncertainty), .02)
        self.assertGreater(np.ptp(predicted), 1.)

    def test_two_parameter_bo_plots_and_failed_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "results").mkdir()
            (run / "config.json").write_text(json.dumps({
                "kind": "bo", "parameters": {
                    "p1": {"lower": 0., "upper": 1.},
                    "p2": {"lower": 0., "upper": 1.}}}))
            (run / "reference.csv").write_text("x,y\n0,0\n1,0\n")
            with (run / "evaluations.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["status", "phase", "objective", "p1", "p2", "curve_file"])
                writer.writeheader()
                for i in range(10):
                    p1, p2 = i / 10, (i * 3 % 10) / 10
                    loss = math.hypot(p1 - .5, p2 - .4)
                    path = f"results/eval.{i}.csv"
                    (run / path).write_text(f"x,y\n0,{loss}\n1,{loss}\n")
                    writer.writerow({"status": "ok", "phase": "initial" if i < 6 else "acquisition",
                                     "objective": loss, "p1": p1, "p2": p2, "curve_file": path})
                writer.writerow({"status": "failed", "phase": "acquisition", "objective": 1e6,
                                 "p1": .3, "p2": .3, "curve_file": ""})
            with patch.object(sys, "argv", ["plot_results.py", str(run)]):
                plot_main()
            for filename in ("convergence.png", "parameters.png", "landscape.png",
                             "curve_fit.png", "surrogate_diagnostics.png", "landscape_2d_gp.png"):
                path = run / "plots" / filename
                self.assertTrue(path.exists(), filename)
                self.assertGreater(path.stat().st_size, 10_000, filename)
            (run / "results" / "eval.5.csv").write_text("x,y\n0,5\n1,5\n")
            with patch.object(sys, "argv", ["plot_results.py", str(run)]):
                with self.assertRaisesRegex(ValueError, "disagrees with curves"):
                    plot_main()


if __name__ == "__main__":
    unittest.main()