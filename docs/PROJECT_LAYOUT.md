# Project layout and relocation notes

Run commands from the repository root. Only [README.md](../README.md) and
[requirements.txt](../requirements.txt) are visible root-level files. Git's
hidden [.gitignore](../.gitignore) must remain here to exclude local results,
cache files, and machine-specific cluster profiles.

| Directory | What belongs there |
| --- | --- |
| [app/](../app/) | All standalone Python controllers, Dakota drivers, model helpers, diagnostics, reports, and configuration. Run `python app/tool.py ...`; internal scripts import each other by their module names. |
| [tests/](../tests/) | Standard-library regression tests and opt-in local Dakota integration. Run `python -m unittest discover -s tests -t . -v`. |
| [fit_quality/](../fit_quality/) | Independent Dakota-free scientific audit package. From the root, run `python -m fit_quality ...`; it is intentionally root-level to preserve its module CLI. |
| [templates/](../templates/) | Editable synthetic and measured starting templates; JSON reference/model paths resolve relative to their JSON file. |
| [clusters/](../clusters/) | Sample profile; ignored `local.json` may contain your own installation and library paths. |
| [scripts/](../scripts/) | Optional Slurm shell wrappers and the QUESO-enabled source-build script. New runs get frozen per-run copies of their evaluation scripts. |
| [docs/](../docs/) | Build, migration, scientific audit, and project-layout guides. |
| [runs/](../runs/) | Generated Dakota runs, plots, audit artifacts and benchmark comparisons. Git ignores outputs except the directory marker. |

## Python entry points

The main CLI is [app/tool.py](../app/tool.py). Its optimization launch path is
[app/run.py](../app/run.py) → [app/driver.py](../app/driver.py) → either the
built-in [app/simulate.py](../app/simulate.py) or a file-backed model selected
by the template. Its MCMC path uses [app/mcmc_run.py](../app/mcmc_run.py),
[app/mcmc_driver.py](../app/mcmc_driver.py),
[app/mcmc_analyze.py](../app/mcmc_analyze.py), and optionally
[app/mcmc_validate.py](../app/mcmc_validate.py). The sampler comparison is
[app/benchmark_samplers.py](../app/benchmark_samplers.py).

The optimization report and diagnostic entry points are
[app/analyze.py](../app/analyze.py),
[app/plot_results.py](../app/plot_results.py),
[app/file_analysis.py](../app/file_analysis.py), and
[app/report.py](../app/report.py). Defaults and template validation are in
[app/config.py](../app/config.py) and [app/settings.py](../app/settings.py);
[app/resources.py](../app/resources.py) writes per-run Slurm scripts, while
[app/provenance.py](../app/provenance.py) records frozen input hashes.

Other app modules: [app/dakota_checks.py](../app/dakota_checks.py) probes the
actual Dakota binary for QUESO support;
[app/mcmc_simulate.py](../app/mcmc_simulate.py) implements the synthetic
Gaussian benchmark and [app/sbc_benchmark.py](../app/sbc_benchmark.py) checks
its analytic calibration; [app/uncertainty.py](../app/uncertainty.py) offers
exploratory optimization-candidate screening;
[app/project.py](../app/project.py) scaffolds and validates an external model.
These tools are optional and do not turn a numerical fit into validated
parameter inference.

## Path and migration behavior

The app modules are deliberately flat standalone scripts rather than an
installed package: invoking `python app/tool.py` places app/ on the script
import path. The repository root remains the default work directory; new
results are still placed under `runs/`, and templates and profiles stay in
their original directories. Dakota input files use absolute paths to the
current app drivers, allowing evaluations started in per-run work directories.

**Previously generated** Dakota input decks, restart files and Slurm jobs may
refer to former root-level scripts by absolute pathname. Re-run the launcher
to create new input files before continuing those workflows. Existing
diagnostics/reports can still be viewed or regenerated without restarting the
old Dakota deck. In particular, do not expect `resume` to repair historical
driver paths automatically. Existing `runs/` files are not rewritten by this
move. The original root-level `python tool.py` and `python run.py` commands are
replaced with `python app/tool.py` and `python app/run.py`.

For reproducible installation and verification see the
[quick start](../README.md#install-and-check), [config migration](CONFIG_MIGRATION.md),
[scientific audit](SCIENTIFIC_AUDIT.md), and [QUESO build](QUESO_BUILD.md) guides.