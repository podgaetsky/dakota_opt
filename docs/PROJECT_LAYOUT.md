# Project layout

Run commands from the repository root. The root-level Python and shell entry points stay in place: generated Dakota inputs and saved runs call them by path. Moving them into a package would invalidate existing run commands and job launch paths.

| Location | Purpose |
| --- | --- |
| `tool.py` | Beginner CLI: checks, demo, opt, BO, MCMC, reporting and listing. |
| `run.py`, `driver.py`, `analyze.py`, `plot_results.py` | Optimization input, model evaluation, best result and figures. |
| `mcmc_run.py`, `mcmc_driver.py`, `mcmc_analyze.py`, `mcmc_validate.py` | Full-curve DREAM calibration, diagnostics and exact-model validation. |
| `report.py`, `file_analysis.py`, `uncertainty.py`, `sbc_benchmark.py` | Offline reports, CSV χ² analysis, optional screening and analytic likelihood checks. |
| `config.py`, `settings.py` | Default settings and user-editable JSON template loader. JSON templates take precedence for a templated run. |
| `simulate.py`, `mcmc_simulate.py` | Synthetic built-in example forward models; not real physics. |
| `run_slurm.sh`, `submit_allocation.sh` | Evaluation wrapper and optional Slurm allocation. |
| `templates/demo.json` | Four-parameter synthetic demonstration. |
| `templates/physical_curve.json` | Real-data starting point; intentionally contains an invalid reference placeholder and no noise SD. |
| `templates/file_benchmark.json`, `templates/linear_benchmark/` | Two-parameter, file-backed runnable benchmark: reference CSV, editable model, example prediction CSV. Paths inside JSON are relative to the JSON file. |
| `templates/linear_file_model.py` | Compatibility wrapper for older saved runs; edit the grouped model instead. |
| `test_*.py` | Standard-library `unittest` regression suite; kept at root so existing test commands work. |
| `runs/` | Generated run data, diagnostics and HTML reports; ignored by Git except the directory marker. |

## Getting started

Install [Python dependencies](../requirements.txt) and Dakota 6.23, then follow the [quick start](../README.md#start-here-editable-templates-and-automatic-reports). Run `python tool.py check` before starting a new run. For a zero-Dakota CSV example, see [the file-based benchmark](../README.md#file-based-benchmark-you-can-edit).

The `.gitignore` excludes generated reports, temporary analysis output, caches and local environments. Reference CSVs in `templates/` are **inputs**, not results. Do not put observational measurements or private credentials in tracked templates.

## Validation

Run `python -m unittest discover -s . -p 'test_*.py'`. The suite tests the file benchmark, expected χ² arithmetic, diagnostics and plotting; some tests inspect saved benchmark runs if available. For a clean checkout, execute `python tool.py demo` to generate full-size real Dakota runs. Slurm backends require a cluster and cannot be verified by local unit tests.
