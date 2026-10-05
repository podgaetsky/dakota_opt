"""Fast fixture-based checks for the editable templates and offline run audit."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import numpy as np
from file_analysis import analyze_files, write_analysis
from driver import evaluate_loss
from mcmc_validate import validate
from project import init_project, validate_model
from report import audit, render
from resources import write_scripts
from settings import CONFIG_KEYS, load_settings

ROOT = Path(__file__).resolve().parent.parent


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.saved = {attr: getattr(config, attr) for attr in CONFIG_KEYS.values()}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: [setattr(config, attr, value) for attr, value in self.saved.items()])
        self.folder = Path(self.tmp.name)
        self.template = json.loads((ROOT / "templates/demo.json").read_text())

    def write(self, data):
        path = self.folder / "settings.json"
        path.write_text(json.dumps(data))
        return path

    def test_demo_template_and_bounds(self):
        user = load_settings(self.write(self.template))
        self.assertEqual(user["mode"], "demo")
        self.assertEqual(config.CONCURRENCY, 8)
        self.template["parameters"]["p1"]["initial"] = 99
        with self.assertRaisesRegex(ValueError, "bounds"):
            load_settings(self.write(self.template))

    def test_missing_measured_reference_and_override(self):
        self.template["mode"] = "measured"
        self.template["reference"] = "NOT_A_FILE.csv"
        path = self.write(self.template)
        with self.assertRaisesRegex(FileNotFoundError, "Measured reference"):
            load_settings(path)
        reference = self.folder / "real.csv"
        reference.write_text("x,y\n0,1\n1,2\n")
        self.assertEqual(load_settings(path, reference)["reference"], str(reference))

    def test_rejects_bad_budget_and_unknown_key(self):
        self.template["bo_batch_size"] = 9
        with self.assertRaisesRegex(ValueError, "exceed concurrency"):
            load_settings(self.write(self.template))
        self.template["bo_batch_size"] = 8
        self.template["typo"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown template keys"):
            load_settings(self.write(self.template))

    def test_file_benchmark_template_selects_custom_forward_model(self):
        benchmark = ROOT / "templates/file_benchmark.json"
        user = load_settings(benchmark)
        self.assertEqual(user["mode"], "measured")
        self.assertEqual(Path(user["simulation_script"]), ROOT / "templates/linear_benchmark/model.py")
        self.assertEqual(Path(user["reference"]), ROOT / "templates/linear_benchmark/reference.csv")
        self.assertEqual(list(config.PARAMETERS), ["a", "b"])

    def test_mcmc_backends_are_explicit_in_templates(self):
        for name in ("demo", "file_benchmark", "physical_curve"):
            data = json.loads((ROOT / "templates" / f"{name}.json").read_text())
            self.assertEqual(data["mcmc"]["backend"], "dream")
        queso = ROOT / "templates/queso_file_benchmark.json"
        user = load_settings(queso)
        self.assertEqual(user["mcmc"]["backend"], "queso")
        self.assertEqual(user["mcmc"]["validate_samples"], 0)
        self.assertEqual(Path(user["reference"]), ROOT / "templates/linear_benchmark/reference.csv")
        self.assertEqual(Path(user["simulation_script"]), ROOT / "templates/linear_benchmark/model.py")
        self.assertEqual(config.BACKEND, "local")
        self.template["mcmc"]["backend"] = "typo"
        with self.assertRaisesRegex(ValueError, "mcmc.backend"):
            load_settings(self.write(self.template))

    def test_new_and_legacy_file_model_entry_points(self):
        reference = ROOT / "templates/linear_benchmark/reference.csv"
        for model in (ROOT / "templates/linear_benchmark/model.py",
                      ROOT / "templates/linear_file_model.py"):
            work = self.folder / model.stem
            work.mkdir()
            (work / "physical_params.json").write_text(json.dumps({"a": 1.15, "b": .4}))
            (work / "reference.csv").write_bytes(reference.read_bytes())
            subprocess.run([sys.executable, str(model), str(work)], check=True, capture_output=True)
            with (work / "curve.csv").open() as stream:
                values = list(csv.DictReader(stream))
            self.assertEqual(len(values), 9)
            self.assertAlmostEqual(float(values[0]["y"]), .75)

    def test_missing_custom_script_fails_before_run(self):
        self.template["simulation_script"] = "missing_model.py"
        with self.assertRaisesRegex(FileNotFoundError, "Simulation Python script"):
            load_settings(self.write(self.template))

    def test_resource_block_generates_matching_slurm_scripts(self):
        for key in ("partition", "cpus_per_evaluation", "memory", "time_limit",
                    "concurrency", "job_timeout_seconds"):
            del self.template[key]
        self.template["resources"] = {"partition": "compute", "cpus": 3, "mem": "12G",
                                      "time": "01:30:00", "concurrency": 4, "timeout": 600}
        self.template["bo_batch_size"] = 4
        load_settings(self.write(self.template))
        self.assertEqual((config.CONCURRENCY, config.CPUS_PER_EVALUATION), (4, 3))
        values = {"partition": config.PARTITION, "cpus_per_evaluation": config.CPUS_PER_EVALUATION,
                  "memory": config.MEMORY, "time_limit": config.TIME_LIMIT,
                  "concurrency": config.CONCURRENCY, "python_executable": sys.executable}
        run = self.folder / "scripts"
        write_scripts(run, values, mode="bo", template=self.folder / "settings.json",
                      workdir=self.folder, dakota="/usr/bin/dakota")
        self.assertIn("#SBATCH --cpus-per-task=3", (run / "run_slurm.sh").read_text())
        allocation = (run / "submit_allocation.sh").read_text()
        self.assertIn("#SBATCH --ntasks=4", allocation)
        self.assertIn("#SBATCH --partition=compute", allocation)
        self.assertIn("--workdir", allocation)
        self.assertIn(str(ROOT / "app" / "tool.py"), allocation)

    def test_scaffold_and_validate_midpoint_without_dakota(self):
        project = init_project(self.folder / "project")
        self.assertTrue((project / "runs/.gitkeep").is_file())
        reference = project / "reference.csv"
        reference.write_text("x,y\n0,0\n1,1\n")
        outcome = validate_model(project / "template.json", workdir=project)
        self.assertTrue(np.isfinite(outcome["midpoint_rms"]))
        self.assertEqual(outcome["job_id"], "local")
        self.assertFalse(list((project / "runs").glob("validate-model-*")))

    def test_execution_sampler_aliases_and_conflicts(self):
        self.template["execution"] = self.template.pop("backend")
        self.template["mcmc"]["sampler"] = self.template["mcmc"].pop("backend")
        user = load_settings(self.write(self.template))
        self.assertEqual((config.BACKEND, user["mcmc"]["backend"]), ("local", "dream"))
        self.template["backend"] = "slurm"
        with self.assertRaisesRegex(ValueError, "execution instead of backend"):
            load_settings(self.write(self.template))

    def test_schema_reports_invalid_timeout_and_sigma(self):
        self.template["job_timeout_seconds"] = -1
        with self.assertRaisesRegex(ValueError, "job_timeout_seconds must be a positive"):
            load_settings(self.write(self.template))
        self.template["job_timeout_seconds"] = 600
        self.template["mcmc"]["sigma"] = -0.1
        with self.assertRaisesRegex(ValueError, "mcmc.sigma must be a positive"):
            load_settings(self.write(self.template))

    def test_simulation_command_uses_template_paths_and_checks_grid(self):
        reference = self.folder / "reference.csv"
        reference.write_text("x,y\n0,1\n1,2\n")
        model = self.folder / "model.py"
        model.write_text("""import csv, json, sys
params = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'w') as stream:
    stream.write('x,y\\n0,1\\n1,' + str(params['p1']) + '\\n')
""")
        self.template.update(mode="measured", reference="reference.csv",
                             simulation={"command": "python model.py {params} {curve}", "workdir": "."})
        user = load_settings(self.write(self.template))
        run = self.folder / "run"
        work = run / "params" / "eval.1"
        work.mkdir(parents=True)
        (run / "results").mkdir()
        (run / "reference.csv").write_bytes(reference.read_bytes())
        settings = {"backend": "local", "simulation": user["simulation"],
                    "python_executable": sys.executable, "job_timeout_seconds": 30}
        value, job, _ = evaluate_loss({"p1": 2}, work, run, settings)
        self.assertEqual((value, job), (0, "local"))
        model.write_text(model.read_text().replace("0,1", "0.5,1"))
        with self.assertRaisesRegex(ValueError, "x grids differ"):
            evaluate_loss({"p1": 2}, work, run, settings)


class FileAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.reference = self.folder / "reference.csv"
        self.prediction = self.folder / "prediction.csv"
        self.reference.write_text("x,y\n0,1\n1,2\n2,3\n")
        self.prediction.write_text("x,y\n0,0.9\n1,2.1\n2,3\n")

    def test_known_chi_squared_reduced_and_saved_figure(self):
        metrics, rows = analyze_files(self.reference, self.prediction, .1, fitted_parameters=1)
        self.assertAlmostEqual(metrics["chi_squared"], 2)
        self.assertAlmostEqual(metrics["reduced_chi_squared"], 1)
        self.assertAlmostEqual(metrics["rms"], (2 * .01 / 3) ** .5)
        self.assertAlmostEqual(rows[0]["standardized_residual"], 1)
        write_analysis(self.reference, self.prediction, .1, self.folder / "out", 1)
        self.assertTrue((self.folder / "out/chi_squared.png").is_file())
        with (self.folder / "out/residuals.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_shipped_file_benchmark_known_chi_squared(self):
        result, _ = analyze_files(ROOT / "templates/linear_benchmark/reference.csv",
                      ROOT / "templates/linear_benchmark/example_prediction.csv", .05, 2)
        self.assertAlmostEqual(result["chi_squared"], 3.09, places=9)
        self.assertAlmostEqual(result["reduced_chi_squared"], 3.09 / 7, places=9)

    def test_invalid_sigma_grid_and_dof(self):
        for sigma in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "sigma"):
                analyze_files(self.reference, self.prediction, sigma)
        result, _ = analyze_files(self.reference, self.prediction, .1, 3)
        self.assertIsNone(result["reduced_chi_squared"])
        self.prediction.write_text("x,y\n0,1\n1.5,2\n2,3\n")
        with self.assertRaisesRegex(ValueError, "x grids differ"):
            analyze_files(self.reference, self.prediction, .1)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        (self.run / "plots").mkdir()

    def save(self, name, data):
        (self.run / name).write_text(json.dumps(data))

    def mcmc(self, rhat=1.003, ess=1000, exact=0.01, validated=16):
        self.save("config.json", {"parameters": {"a": {"lower": 0, "upper": 2}},
                                  "mode": "benchmark", "mcmc_backend": "dream"})
        self.save("posterior_summary.json", {"retained_draws": 12000, "rhat": {"a": rhat},
                    "ess_bulk": {"a": ess}, "ess_tail": {"a": ess},
                    "mcse_mean_fraction_of_posterior_sd": {"a": .03},
                    "posterior": {"a": {"median": 1.0, "equal_tailed_95": [.9, 1.1]}},
                    "descriptive_marginal_ks_D": {"a": .02}})
        if validated is not None:
            self.save("validation.json", {"validated_samples": validated,
                                          "log_density_error_p90": exact})

    def test_good_numerics_do_not_certify_science(self):
        self.mcmc()
        result = audit(self.run)
        self.assertTrue(result["metrics"]["numerical_checks_passed"])
        self.assertIn("scientific validity unverified", result["headline"])
        self.assertTrue(any(c["state"] == "unknown" for c in result["checks"]))
        (self.run / "plots" / "rank.png").write_bytes(b"image")
        page = render(self.run, result)
        self.assertIn('src="plots/rank.png"', page.read_text())
        self.assertEqual(json.loads((self.run / "report.json").read_text())["metrics"]["retained_draws"], 12000)

    def test_bad_chains_surrogate_and_missing_validation(self):
        self.mcmc(rhat=1.12, ess=25, exact=3)
        self.assertFalse(audit(self.run)["metrics"]["numerical_checks_passed"])
        (self.run / "validation.json").unlink()
        result = audit(self.run)
        self.assertFalse(result["metrics"]["numerical_checks_passed"])
        self.assertTrue(any(c["state"] == "missing" for c in result["checks"]))

    def test_incomplete_diagnostics_fail_closed(self):
        self.mcmc(validated=4)
        summary = json.loads((self.run / "posterior_summary.json").read_text())
        summary["rhat"] = {}
        summary["ess_tail"] = {}
        self.save("posterior_summary.json", summary)
        result = audit(self.run)
        self.assertFalse(result["metrics"]["numerical_checks_passed"])
        self.assertIsNone(result["metrics"]["max_rhat"])
        self.assertIsNone(result["metrics"]["min_tail_ess"])
        self.assertEqual(next(c["state"] for c in result["checks"]
                              if c["label"].startswith("Chain mixing")), "warn")
        render(self.run, result)

    def test_failed_benchmark_reference_blocks_numerical_headline(self):
        self.mcmc()
        summary = json.loads((self.run / "posterior_summary.json").read_text())
        summary["descriptive_marginal_ks_D"]["a"] = .25
        self.save("posterior_summary.json", summary)
        self.assertFalse(audit(self.run)["metrics"]["numerical_checks_passed"])

    def test_exact_validation_needs_sufficient_draws(self):
        self.save("config.json", {"parameters": {"a": {"lower": 0, "upper": 2}},
                                  "simulation_script": "simulate.py", "sigma": .05})
        self.save("posterior_summary.json", {"rhat": {"a": 1.001},
                                            "ess_bulk": {"a": 900}, "ess_tail": {"a": 800}})
        (self.run / "reference.csv").write_text("x,y\n0,1\n1,1\n")
        (self.run / "params").mkdir()
        chains = np.full((4, 30, 1), .5)
        logp = np.zeros((4, 30))

        def exact_curve(work, *_):
            (work / "curve.csv").write_text("x,y\n0,1\n1,1\n")

        with patch("mcmc_validate.read_chains", return_value=(chains, logp)), patch(
                "mcmc_validate.run_job", side_effect=exact_curve):
            self.assertFalse(validate(self.run, 4)["inference_ready"])
            self.assertTrue(validate(self.run, 16)["inference_ready"])

    def test_optimization_never_certifies_global_optimum(self):
        self.save("config.json", {"kind": "bo", "parameters": {"a": {"lower": 0, "upper": 2}}})
        with (self.run / "evaluations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["status", "objective", "a"])
            writer.writeheader()
            writer.writerows([{"status": "ok", "objective": 4, "a": 1},
                              {"status": "ok", "objective": 2, "a": 2},
                              {"status": "failed", "objective": 1e6, "a": 0}])
        result = audit(self.run)
        self.assertEqual(result["metrics"]["failed"], 1)
        self.assertEqual(result["metrics"]["best_rms"], 2)
        self.assertIn("not certified", result["headline"])
        self.assertTrue(any(c["state"] == "warn" for c in result["checks"]))


if __name__ == "__main__":
    unittest.main()
