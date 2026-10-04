"""Regression checks for Dakota MCMC curve calibration and posterior diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from mcmc_analyze import analytic_linear_posterior, read_chains
from mcmc_run import BENCHMARK, calibration_record, input_text
from sbc_benchmark import run_sbc


class BayesianTests(unittest.TestCase):
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