# Dakota curve optimization and Bayesian calibration

Run Dakota 6.23 curve fitting (parallel pattern search or native GP efficient
global optimization), DREAM or QUESO Bayesian calibration, and Dakota-free
scientific curve audits. The included model and observations are **synthetic**;
replace them and justify the noise model before making physical claims.

The repository root contains only this README and [requirements.txt](requirements.txt)
as visible files. Python entry points live in [app/](app/), shell helpers in
[scripts/](scripts/), tests in [tests/](tests/), model templates in
[templates/](templates/), and explanations in [docs/](docs/). The standard
hidden [.gitignore](.gitignore) remains at root so Git can ignore generated
results and private profiles. See the [layout and migration guide](docs/PROJECT_LAYOUT.md).

## Install and check

Use Python 3.10+ and install the [requirements](requirements.txt) in your
environment. Install Dakota 6.23 separately; it must be on `PATH`, selected by
`DAKOTA`, or passed using `--dakota /path/to/dakota`. Start in the repository
root so relative template and output paths work:

    python -m pip install -r requirements.txt
    python app/tool.py doctor --dakota /path/to/dakota
    python -m unittest discover -s tests -t . -v

The upstream prebuilt Dakota CLI used here lacks QUESO. To use QUESO, build it
from official source with [scripts/build_queso.sh](scripts/build_queso.sh) and
follow [docs/QUESO_BUILD.md](docs/QUESO_BUILD.md). A QUESO build also needs its
runtime libraries accessible through `LD_LIBRARY_PATH` or an equivalent loader
configuration. A private `clusters/local.json` profile can record the Dakota
binary, Python interpreter, modules, and loader directories; copy the shape of
[clusters/example.json](clusters/example.json) and check it with
`python app/tool.py doctor --cluster local`. This machine's isolated build is
under `/tmp` and must be rebuilt if that directory is cleared. Slurm has not
been verified on this machine.

## Quick start (no Slurm required)

    python app/tool.py opt templates/file_benchmark.json --dakota /path/to/dakota
    python app/tool.py bo templates/file_benchmark.json --dakota /path/to/dakota
    python app/tool.py list

The [file benchmark](templates/file_benchmark.json) points to a nine-ordinate
[synthetic reference](templates/linear_benchmark/reference.csv) and an editable
[forward model](templates/linear_benchmark/model.py). Each optimization writes
an isolated run under `runs/` with its Dakota input, settings snapshot,
per-evaluation curves and diagnostics, best observed RMS, plots, provenance,
HTML report and (where applicable) a separate scientific audit. Optimization
does **not** infer parameter intervals or certify a global optimum. For the
four-parameter built-in demo and analytic MCMC check, use
`python app/tool.py demo --dakota /path/to/dakota`; this runs several examples
and uses substantially more evaluations.

To start your own file-backed project:

    python app/tool.py init /path/to/new-project
    # Replace the placeholder reference.csv and model.py; edit template.json.
    python app/tool.py validate-model /path/to/new-project/template.json --workdir /path/to/new-project
    python app/tool.py opt /path/to/new-project/template.json --workdir /path/to/new-project --dakota /path/to/dakota

The scaffold's `reference.csv` is deliberately invalid until replaced. Supply
a strictly increasing `x,y` reference CSV, a model that produces `x,y` on the
same grid, physical parameter bounds and realistic resource budgets. Paths to
the reference and simulator inside a JSON template resolve relative to that
template, not the working directory. [docs/CONFIG_MIGRATION.md](docs/CONFIG_MIGRATION.md)
explains the template schema, aliases and independent `execution` (local,
Slurm, allocation) versus `mcmc.sampler` (DREAM, QUESO) choices.

## Bayesian calibration and sampler comparison

    python app/mcmc_run.py --mode benchmark --dakota /path/to/dakota --samples 24000 --build-samples 48
    python app/tool.py mcmc templates/file_benchmark.json --sampler dream --dakota /path/to/dakota
    python app/tool.py mcmc templates/queso_file_benchmark.json --sampler queso --dakota /path/to/queso-enabled-dakota
    python app/benchmark_samplers.py --dakota /path/to/queso-enabled-dakota --samples 4000

DREAM uses four chains; its `chain_samples` value is **total** samples, not
samples per chain. QUESO uses one DRAM chain and requires a live method
availability check (parsing the keyword is insufficient). Both fit the
*curve ordinates*, not optimized RMS, with bounded uniform priors and a native
Dakota GP emulator. Templates specify an observation **standard deviation**;
the Dakota calibration file correctly stores its **variance**. The sampler
comparison uses identical reference, bounds, sigma, seed, training budget and
total requested draws; its comparison JSON and run outputs are under `runs/`.
The [measured local comparison and build details](docs/QUESO_BUILD.md#verified-local-results-2026-10-05)
describe the results and their limitations. QUESO's single chain cannot supply
multichain R-hat, and the quick comparison skips exact-vs-GP validation; its
intervals are exploratory. A small RMS or a green numerical report cannot
validate a likelihood, prior, observational covariance or physical model.

## Dakota-free diagnostics and tests

    python -m fit_quality --reference templates/linear_benchmark/reference.csv --prediction templates/linear_benchmark/example_prediction.csv --parameters templates/linear_benchmark/parameters.json --parameter-definitions templates/file_benchmark.json --parameter-count 2 --noise-model templates/linear_benchmark/noise.json --output example_analysis
    python app/file_analysis.py --reference templates/linear_benchmark/reference.csv --prediction templates/linear_benchmark/example_prediction.csv --sigma 0.05 --fitted-parameters 2 --output example_analysis
    python -m unittest discover -s tests -t . -v

For the **opt-in real Dakota integration** tests, set
`DAKOTA_INTEGRATION=1`, `DAKOTA=/path/to/dakota`, the required loader paths,
and `DAKOTA_EXPECT_QUESO=1` if the binary supports QUESO, then run the same
unittest command. Without the flag these two tests are skipped. The scientific
audit's separate goodness-of-fit, identifiability and validation evidence is
described in [docs/SCIENTIFIC_AUDIT.md](docs/SCIENTIFIC_AUDIT.md). Generated
`runs/` contents and `example_analysis/` are ignored by Git.

## Existing runs and cluster use

Existing run snapshots may contain absolute paths to the **old root-level**
driver scripts in their `dakota.in` and frozen Slurm commands. Their saved
reports and output remain readable, but restart/submission from those snapshots
requires regenerating them with the new layout (or restoring the original
scripts at those paths); a restart is not automatically migrated. New runs
write paths to [app/](app/) and per-run Slurm scripts. An optional allocation
example lives in [scripts/submit_allocation.sh](scripts/submit_allocation.sh);
edit resource requests to match the simulation and your cluster before use.
For a lightweight single-evaluation wrapper see
[scripts/run_slurm.sh](scripts/run_slurm.sh). Neither script substitutes for
cluster-specific validation.