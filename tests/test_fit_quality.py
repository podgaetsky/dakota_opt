"""Synthetic controls for standalone inference; no Dakota installation required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_distribution
from scipy import stats

from fit_quality.core import analyze_curve
from fit_quality.inference import identifiability, finite_difference_jacobian, posterior_predictive
from fit_quality.residuals import diagnose
from fit_quality.compare import compare
from fit_quality.profile import profile_objective, save_profile_figure
from fit_quality.surrogate import validate as validate_surrogate


class ScientificTests(unittest.TestCase):
    def test_gaussian_likelihood_and_eligibility(self):
        x = np.arange(5.)
        ref = np.column_stack((x, [1., 2., 3., 4., 5.]))
        pred = ref.copy()
        pred[:, 1] += 1
        result, data = analyze_curve(ref, pred, noise={"sigma": 2.}, parameter_count=1)
        self.assertAlmostEqual(result["sse"], 5.)
        self.assertAlmostEqual(result["chi2"], 1.25)
        self.assertAlmostEqual(result["log_likelihood"], -.5 * (1.25 + 5 * np.log(2 * np.pi * 4)))
        self.assertIsNone(result["chi2_p_value"])
        eligible, _ = analyze_curve(ref, pred, noise={"sigma": 2.}, parameter_count=1, chi2_test_justified=True)
        self.assertAlmostEqual(eligible["chi2_p_value"], chi2_distribution.sf(1.25, 4))
        self.assertAlmostEqual(result["aic"], 2 - 2 * result["log_likelihood"])
        self.assertEqual(data["standardized"].shape, (5,))

    def test_heteroscedastic_and_correlated_noise(self):
        ref = np.column_stack((np.arange(3.), np.zeros(3)))
        pred = ref.copy(); pred[:, 1] = [1., 1., 0.]
        het, _ = analyze_curve(ref, pred, noise={"sigma_i": [1., 2., 3.]})
        self.assertAlmostEqual(het["chi2"], 1.25)
        cov = [[1., .5, 0.], [.5, 1., 0.], [0., 0., 1.]]
        corr, data = analyze_curve(ref, pred, noise={"covariance": cov})
        self.assertAlmostEqual(corr["chi2"], 4 / 3)
        self.assertEqual(data["standardized"].shape, (3,))
        for noise in ({"sigma": 0}, {"sigma": 1, "covariance": cov}, {"sigma": 1, "unrecognized": 2},
                  {"covariance": [[1, 2], [2, 1]]}):
            with self.assertRaises(ValueError):
                analyze_curve(ref, pred, noise=noise)

    def test_sampling_density_and_zero_scale(self):
        x = np.array([0., .01, .02, 1.])
        ref = np.column_stack((x, np.zeros(len(x))))
        pred = ref.copy(); pred[:, 1] = x
        result, _ = analyze_curve(ref, pred)
        self.assertNotAlmostEqual(result["rmse"], result["x_integrated_rmse"])
        self.assertIsNone(result["relative_l2_error"])
        self.assertAlmostEqual(result["x_integrated_rmse"], np.sqrt(np.trapezoid(x ** 2, x)))
        with self.assertRaisesRegex(ValueError, "grids differ"):
            analyze_curve(ref, np.column_stack((x + .2, x)))

    def test_ordered_correlation_warns(self):
        x = np.arange(120.)
        ref = np.column_stack((x, np.zeros(len(x))))
        pred = np.column_stack((x, np.sin(x / 15)))
        _, data = analyze_curve(ref, pred, noise={"sigma": 1})
        result = diagnose(data)
        self.assertEqual(result["independence_status"], "WARN")
        self.assertIn("questionable", result["independence_note"])

    def test_local_identifiability_and_finite_differences(self):
        x = np.linspace(0, 1, 11)
        fn = lambda p: p[0] + p[1] * x
        jac = finite_difference_jacobian(fn, [1., .5], [0, 0], [2, 1])
        np.testing.assert_allclose(jac, np.column_stack((2 * np.ones(len(x)), x)), atol=1e-9)
        strong = identifiability(jac, {"sigma": .1})
        self.assertEqual(strong["numerical_rank"], 2)
        self.assertIn("local_correlation", strong)
        weak = identifiability(np.column_stack((x, x)), {"sigma": .1})
        self.assertEqual(weak["numerical_rank"], 1)
        self.assertNotIn("local_covariance_normalized", weak)

    def test_predictive_curves_keep_noise_separate(self):
        ref = np.column_stack((np.arange(4.), np.ones(4)))
        curves = np.array([[1., 1., 1., 1.], [1.1, 1.1, 1.1, 1.1], [.9, .9, .9, .9]])
        result, _, obs = posterior_predictive(ref, curves, {"sigma": .1})
        self.assertEqual(result["noise_model"], "iid Gaussian")
        self.assertEqual(obs.shape, (3, 4))
        self.assertEqual(len(result["rms_distribution"]), 3)
        no_noise, _, _ = posterior_predictive(ref, curves)
        self.assertIsNone(no_noise["observation_predictive_interval_95"])

    def test_model_comparison_rejects_incompatible_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.csv").write_text("x,y\n0,1\n1,2\n")
            (root / "b.csv").write_text("x,y\n0,2\n1,3\n")
            (root / "noise.json").write_text(json.dumps({"sigma": .1}))
            specs = [{"model": "a", "reference": str(root / "a.csv"), "prediction": str(root / "a.csv"),
                      "noise_model": str(root / "noise.json"), "parameter_count": 1},
                     {"model": "b", "reference": str(root / "b.csv"), "prediction": str(root / "b.csv"),
                      "noise_model": str(root / "noise.json"), "parameter_count": 1}]
            with self.assertRaisesRegex(ValueError, "identical observations"):
                compare(specs)

    def test_nonlinear_and_parameter_bound_controls(self):
        x = np.linspace(0, 1, 30)
        true = np.exp(-2 * x)
        ref = np.column_stack((x, true))
        pred = np.column_stack((x, np.exp(-1.5 * x)))
        metrics, _ = analyze_curve(ref, pred)
        self.assertGreater(metrics["sse"], 0)
        local = finite_difference_jacobian(lambda p: np.exp(-p[0] * x), [0.], [0.], [4.])
        np.testing.assert_allclose(local[:, 0], -4 * x, atol=.001)  # one-sided at bound

    def test_multimodal_profile_reoptimizes_other_parameters(self):
        objective = lambda p: (p[0] ** 2 - 1) ** 2 + (p[1] - p[0]) ** 2
        result = profile_objective(objective, [1, 1], [[-2, 2], [-2, 2]], grid_size=9, starts=2)
        self.assertLess(result["profiles"][0][2]["best_profile_objective"], .01)  # p0 = -1, p1 reoptimized
        self.assertLess(result["profiles"][0][6]["best_profile_objective"], .01)  # p0 = +1
        self.assertGreater(result["profiles"][0][4]["best_profile_objective"], .5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.png"
            save_profile_figure(result, path, ("p0", "p1"), "RMS")
            self.assertGreater(path.stat().st_size, 1000)

    def test_deliberately_misspecified_linear_fit_has_correlated_residuals(self):
        x = np.linspace(0, 6, 120)
        ref = np.column_stack((x, x ** 2))
        fit = np.polyval(np.polyfit(x, ref[:, 1], 1), x)
        _, inputs = analyze_curve(ref, np.column_stack((x, fit)), noise={"sigma": 1})
        self.assertEqual(diagnose(inputs)["independence_status"], "WARN")

    def test_analytic_linear_90_and_95_percent_coverage(self):
        # Coverage under independent datasets and a correctly specified known-noise model.
        rng = np.random.default_rng(18)
        x = np.linspace(-1, 1, 9)
        design = np.column_stack((np.ones(len(x)), x))
        sd = .05 / np.sqrt(len(x))
        draws = 1000
        y = 1.15 + .4 * x + rng.normal(0, .05, size=(draws, len(x)))
        means = np.linalg.lstsq(design, y.T, rcond=None)[0][0]
        for level in (.90, .95):
            z = stats.norm.ppf((1 + level) / 2)
            coverage = np.mean(np.abs(means - 1.15) <= z * sd)
            self.assertLess(abs(coverage - level), .035)

    def test_heteroscedastic_and_correlated_interval_coverage(self):
        rng = np.random.default_rng(33)
        x = np.linspace(0, 1, 10)
        sd = .04 + .08 * x
        for covariance in (np.diag(sd ** 2),
                           np.outer(sd, sd) * .5 ** np.abs(np.subtract.outer(np.arange(10), np.arange(10)))):
            inverse = np.linalg.inv(covariance)
            variance = 1 / np.sum(inverse)
            noise = rng.multivariate_normal(np.zeros(10), covariance, size=1400)
            means = (noise @ inverse @ np.ones(10)) * variance
            for level in (.90, .95):
                coverage = np.mean(np.abs(means) <= stats.norm.ppf((1 + level) / 2) * np.sqrt(variance))
                self.assertLess(abs(coverage - level), .035)

    def test_surrogate_holdout_predictive_sd_not_observation_noise(self):
        x = np.arange(5.)
        exact = np.column_stack((x, x))
        approx = np.column_stack((x, x + .1))
        result = validate_surrogate(exact, approx, np.full(5, .2))
        self.assertAlmostEqual(result["rmse"], .1)
        self.assertAlmostEqual(result["standardized_prediction_error_rmse"], .5)
        self.assertEqual(result["predictive_95_coverage"], 1.)
        with self.assertRaises(ValueError):
            validate_surrogate(exact, approx, np.zeros(5))

    def test_dakota_free_cli_exposes_missing_validation(self):
        root = Path(__file__).resolve().parent.parent / "templates/linear_benchmark"
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, "-m", "fit_quality", "--reference", str(root / "reference.csv"),
                       "--prediction", str(root / "example_prediction.csv"),
                       "--parameters", str(root / "parameters.json"), "--noise-model", str(root / "noise.json"),
                       "--output", temp]
            subprocess.run(command, check=True, capture_output=True)
            result = json.loads((Path(temp) / "report.json").read_text())
            self.assertIsNone(result["fit_metrics"]["chi2_p_value"])
            self.assertEqual(result["validation"]["status"], "UNKNOWN")
            self.assertIn("Scientific assumption audit", (Path(temp) / "report.html").read_text())

    def test_multiple_held_out_curves_are_not_training_accuracy(self):
        root = Path(__file__).resolve().parent.parent / "templates/linear_benchmark"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = [{"name": "future batch A", "reference": str(root / "reference.csv"),
                         "prediction": str(root / "example_prediction.csv")},
                        {"name": "future batch B", "reference": str(root / "reference.csv"),
                         "prediction": str(root / "example_prediction.csv") }]
            (directory / "validation.json").write_text(json.dumps(manifest))
            subprocess.run([sys.executable, "-m", "fit_quality", "--reference", str(root / "reference.csv"),
                            "--prediction", str(root / "reference.csv"), "--validation-manifest", str(directory / "validation.json"),
                            "--output", str(directory / "out")], check=True, capture_output=True)
            report = json.loads((directory / "out/report.json").read_text())
            self.assertEqual(report["fit_metrics"]["rms"], 0)
            self.assertGreater(report["validation"]["validation_rms"], 0)
            self.assertEqual(len(report["validation"]["curves"]), 2)


if __name__ == "__main__":
    unittest.main()
