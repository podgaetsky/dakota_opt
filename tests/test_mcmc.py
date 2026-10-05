"""Regression checks for Dakota MCMC curve calibration and posterior diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import kstest

from mcmc_analyze import analytic_linear_posterior, analyze, read_chains, read_queso_chain
from mcmc_run import BENCHMARK, calibration_record, input_text
from dakota_checks import check_dakota
from benchmark_samplers import compare
from report import audit
from sbc_benchmark import run_sbc


class BayesianTests(unittest.TestCase):
    def test_queso_method_check_distinguishes_missing_build_from_bad_deck(self):
        import subprocess
        version = subprocess.CompletedProcess([], 0, "Dakota version 6.23\n", "")
        missing = subprocess.CompletedProcess([], -11, "Error: QUESO Bayesian calibration method unavailable.\n", "")
        with patch("dakota_checks.subprocess.run", side_effect=(version, missing)) as launch:
            result = check_dakota("dakota", {}, "/usr/bin/python3")
        self.assertEqual(result["queso"], "unavailable")
        self.assertEqual(result["queso_exit_code"], -11)
        self.assertEqual(launch.call_args.args[0][-3:], ["-i", "dakota.in", "-check"])
        invalid = subprocess.CompletedProcess([], 1, "Error: invalid calibration input", "")
        with patch("dakota_checks.subprocess.run", side_effect=(version, invalid)):
            self.assertEqual(check_dakota("dakota", {}, "/usr/bin/python3")["queso"], "unknown")

    def test_sampler_comparison_rejects_different_references(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = iter(("dream", "queso"))

            def fake_launch(*_args, **_kwargs):
                sampler = next(calls)
                run = root / "runs" / sampler
                run.mkdir(parents=True)
                (run / "posterior_summary.json").write_text(json.dumps({"posterior": {"a": {"median": 1}}}))
                (run / "config.json").write_text(json.dumps({"parameters": BENCHMARK,
                    "sigma": .05, "seed": 491, "build_samples": 8,
                    "likelihood_data_selection": {"selected_ordinates": 9}}))
                (run / "reference.csv").write_text(f"x,y\n0,{sampler}\n")
                return subprocess.CompletedProcess([], 0, str(run) + "\n", "")

            with patch("benchmark_samplers.check_dakota", return_value={"queso": "available", "version": "Dakota 6.23"}), patch(
                    "benchmark_samplers.subprocess.run", side_effect=fake_launch):
                with self.assertRaisesRegex(RuntimeError, "Sampler inputs differ"):
                    compare(root / "template.json", "/usr/bin/dakota", root, samples=100, build_samples=8)

    def test_exact_posterior_for_orthogonal_linear_design(self):
        xs = np.linspace(-1, 1, 9)
        mean, covariance = analytic_linear_posterior(xs, 1.2 - .3 * xs, .05)
        np.testing.assert_allclose(mean, [1.2, -.3], atol=1e-13)
        np.testing.assert_allclose(np.diag(covariance), [.05 ** 2 / 9, .05 ** 2 / sum(xs ** 2)])

    def test_dakota_input_stores_variance_and_returns_curve_vector(self):
        text = input_text(Path("/tmp/benchmark"), BENCHMARK, 9, 24000, 48, 123)
        self.assertIn("bayes_calibration dream", text)
        self.assertIn("chains = 4", text)
        self.assertIn("calibration_terms = 9", text)
        self.assertIn("variance_type = 'scalar'", text)
        self.assertIn("asynchronous evaluation_concurrency", text)
        row = calibration_record([{"y": "1.5"}, {"y": "2.0"}], .05).split()
        np.testing.assert_allclose([float(item) for item in row], [1.5, 2.0, .0025, .0025])

    def test_chain_reader_requires_four_column_dream_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(4):
                (root / f"dakota_dream_chain{i}.txt").write_text(
                    f"DREAM chain {i}\n0 -5 0.25 0.5\n1 -4 0.3 0.55\n")
            x, logp = read_chains(root, ("a", "b"))
            self.assertEqual(x.shape, (4, 2, 2))
            self.assertEqual(logp.shape, (4, 2))
            self.assertEqual(x[0, 1, 0], .3)

    def test_queso_deck_and_annotated_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = input_text(root, BENCHMARK, 9, 501, 48, 123, "queso")
            self.assertIn("bayes_calibration queso", text)
            self.assertIn("dram", text)
            self.assertIn("export_chain_points_file 'chain.dat'", text)
            self.assertNotIn("chains = 4", text)
            self.assertIn("calibration_terms = 9", text)
            (root / "chain.dat").write_text(
                "%mcmc_id interface x_a x_b response\n"
                "1 APPROX_INTERFACE_1 0.2 0.3 1\n"
                "2 APPROX_INTERFACE_1 0.21 0.31 1\n"
                "3 APPROX_INTERFACE_1 0.22 0.32 1\n"
                "4 APPROX_INTERFACE_1 0.23 0.33 1\n")
            np.testing.assert_allclose(read_queso_chain(root, ("a", "b"))[0, 0], [.2, .3])
            (root / "chain.dat").write_text("%mcmc_id x_a x_b\n1 nan 0.2\n2 0.1 0.2\n3 0.2 0.2\n4 0.3 0.2\n")
            with self.assertRaisesRegex(ValueError, "finite"):
                read_queso_chain(root, ("a", "b"))

    def test_queso_report_never_passes_inference_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"mcmc_backend": "queso", "mode": "curve",
                "parameters": {"a": BENCHMARK["a"]}}))
            (root / "posterior_summary.json").write_text(json.dumps({"posterior": {"a": {"median": 1.0}}}))
            result = audit(root)
            self.assertFalse(result["metrics"]["numerical_checks_passed"])
            self.assertTrue(any(c["state"] == "missing" for c in result["checks"]))

    def test_queso_single_chain_analysis_is_exploratory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"mcmc_backend": "queso", "mode": "curve",
                "parameters": {"a": BENCHMARK["a"]}, "sigma": .05}))
            (root / "reference.csv").write_text("x,y\n0,1\n1,2\n")
            draws = np.random.default_rng(123).uniform(.3, .7, 300)
            (root / "chain.dat").write_text("%mcmc_id interface x_a least_sq_term_1_1\n" +
                "".join(f"{i} GP {v:.8f} 1.0\n" for i, v in enumerate(draws, 1)))
            summary = analyze(root)
            self.assertEqual(summary["retained_draws"], 150)
            self.assertEqual(summary["rhat"], {})
            self.assertIn("EXPLORATORY", summary["posterior_interpretation"])
            self.assertFalse(audit(root)["metrics"]["numerical_checks_passed"])

    def test_independent_sbc_pit_is_calibrated(self):
        pit, results = run_sbc(500, .05, 2748)
        self.assertEqual(pit.shape, (500, 2))
        for name, column in zip(("a", "b"), pit.T):
            self.assertGreater(results[name]["p_value"], .01)
            self.assertLess(results[name]["D"], .08)
            # An intentionally misspecified diagnostic must be detectably wrong.
            self.assertLess(kstest(column ** 3, "uniform").pvalue, 1e-5)

    def test_verified_dream_benchmark_if_available(self):
        root = Path(__file__).parent / "runs"
        eligible = [p for p in root.glob("*_mcmc_benchmark") if (p / "validation.json").exists()]
        if not eligible:
            self.skipTest("Run mcmc_run.py --mode benchmark first")
        run = max(eligible, key=lambda p: p.stat().st_mtime)
        validated = json.loads((run / "validation.json").read_text())
        self.assertTrue(validated["inference_ready"])
        summary = json.loads((run / "posterior_summary.json").read_text())
        for index, name in enumerate(("a", "b")):
            expected = summary["analytic_gaussian_reference"]["mean"][index]
            self.assertLess(abs(summary["posterior"][name]["median"] - expected), .01)
            self.assertLess(summary["descriptive_marginal_ks_D"][name], .08)
        for file in ("corner.png", "trace.png", "benchmark_marginals.png", "posterior_predictive.png"):
            self.assertGreater((run / "plots" / file).stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()