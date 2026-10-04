# Dakota 6.23: parallel curve fitting and native batch EGO

## Start here: editable templates and automatic reports

Install Dakota 6.23 and Python packages NumPy, SciPy, Matplotlib, scikit-learn,
ArviZ and corner (see installation below). From this directory, using the Python
environment with those packages:

    python tool.py check
    python tool.py demo
    python tool.py list

`demo` runs a local synthetic optimization, native GP batch EGO, a four-chain
analytic DREAM benchmark, and a separate simulation-based calibration (SBC)
exercise. New `runs/` directories contain `report.html`, `report.json` and
`plots/`. Open the HTML **locally** in a browser; keep its adjacent `plots/`
directory when copying the report. This is a working example, **not a fit to
your physical data**. On this development machine the tested Python executable
is `/home/roypo/miniconda3/envs/scientific/bin/python`; on other machines use
your own Python. `check` detects Dakota 6.23 on PATH or under the local 6.23
installation; override with `--dakota /path/to/dakota`. The RHEL8 CLI on this
machine needs extra private runtime libraries, discussed below.

To adjust budgets, bounds or resources, copy and edit `templates/demo.json`,
then run `python tool.py opt templates/demo.json`,
`python tool.py bo templates/demo.json`, or
`python tool.py mcmc templates/demo.json`. The latter fits a synthetic curve,
**not** the analytic benchmark in `demo`; it may have poor mixing and should
not be used for credible-interval claims unless all diagnostics pass. Use
`python tool.py report runs/YOUR_RUN` to regenerate an audit after further
analysis. `tool.py list` shows recent run statuses. The `opt`/`bo` controllers
store all evaluated curves and best-so-far diagnostics.

### File-based benchmark you can edit

The runnable [file benchmark template](templates/file_benchmark.json) uses the
[nine-point CSV](templates/linear_reference.csv) as its **synthetic** observations
and the editable [forward model](templates/linear_file_model.py). That model reads
each evaluation's `physical_params.json` and `reference.csv`, then writes
`curve.csv` with columns `x,y` on the identical x grid. It uses the line
$m(x)=a+bx$; replace that function with your own file parser/simulator and
update `parameters` in the JSON. `simulation_script` and `reference` paths are
resolved relative to the JSON template. A custom model must work on compute
nodes too. The example is local and runs without any external measured data:

    python tool.py opt templates/file_benchmark.json
    python tool.py bo templates/file_benchmark.json
    python tool.py mcmc templates/file_benchmark.json
    python tool.py list

Optimization **still minimizes RMS**; DREAM **still calibrates all curve
ordinates** using the configured noise SD. Separately, the file-analysis
function in [file_analysis.py](file_analysis.py) reads the observed and predicted
CSV files, checks their x grids, and calculates
$\chi^2=\sum_i[(y_i-m_i)/\sigma]^2$ and
$\chi^2_\mathrm{reduced}=\chi^2/(N-k)$ if $N>k$ (where $k$ is the provided
number of fitted parameters). When a measured-style optimization template has
a non-null `mcmc.sigma`, `tool.py` automatically analyzes its saved best curve
and includes `plots/chi_squared.json`, `plots/residuals.csv`,
`plots/chi_squared.png` and the χ² metrics/plot in its HTML report. To redo an
optimization analysis with a different **independently justified** SD:

    python tool.py analyze runs/YOUR_OPT_RUN --sigma 0.05

To compare two arbitrary x,y CSVs without Dakota, use:

    python file_analysis.py --reference path/to/observed.csv --prediction path/to/model.csv --sigma 0.05 --fitted-parameters 2 --output path/to/figures

For a zero-setup numerical check, compare the included reference to the
[example prediction](templates/linear_example_prediction.csv):

    python file_analysis.py --reference templates/linear_reference.csv --prediction templates/linear_example_prediction.csv --sigma 0.05 --fitted-parameters 2 --output example_analysis

This yields χ² ≈ 3.09 and reduced χ² ≈ 0.4414, with a residual CSV, JSON and
PNG under `example_analysis/`. Change `--sigma` only when independently
justified; it is not estimated from the same residuals here.

The benchmark opt run returned best RMS ≈ 0.02921, χ² ≈ 3.073, reduced χ² ≈
0.439 (nine points and two fitted parameters); EGO returned RMS ≈ 0.02918 and
χ² ≈ 3.065. The file-based four-chain DREAM run passed the *numerical* checks
(max R̂ ≈ 1.0034, minimum bulk ESS ≈ 1148, GP/exact log-density p90 ≈
0.000058). These are reproducible software examples, **not** evidence that an
independent-noise Gaussian model or a physical forward model is valid. Reduced
χ² and the residual plot are descriptive, not a p-value or a certificate of
fit; correlated errors require a covariance-aware likelihood and analysis.

For your own x,y data, edit `templates/physical_curve.json`: point `reference`
to an existing CSV with header `x,y` (relative paths in the template are
relative to that template), replace `parameters` with **your model's** names
and physical bounds, set the resource requests and `backend` to `local`,
`slurm`, or `allocation`, and set `mcmc.sigma` to an independently justified
positive *per-ordinate standard deviation* (not variance). You may instead
pass `--reference /path/to/data.csv` and `--sigma 0.05` to the CLI, e.g.:

    python tool.py opt templates/physical_curve.json --reference /path/to/data.csv
    python tool.py mcmc templates/physical_curve.json --reference /path/to/data.csv --sigma 0.05

The placeholder reference is deliberately absent so that a beginner cannot
silently mistake synthetic data for observations. **You must replace the demo
forward model in `simulate.py` (or `run_slurm.sh`) before fitting real data**;
it consumes physical parameter names and produces the model x,y curve on the
same grid as the observations. The provided measured template uses `slurm` and
Slurm was **not** available for testing on this machine: switch to `local` to
smoke-test one model evaluation and verify your simulator/grid before
submitting. For allocation mode match `submit_allocation.sh` to the template's
resources. `sigma: null` in the measured template requires explicit user input
for MCMC; optimizing RMS does not require sigma.

Audit interpretation: **pass** means only that a stated *numerical* criterion
was met; **warn** identifies a failed/weak numerical check; **missing** means
validation has not run; **unknown** means the available data cannot establish
the claim. The run report lists best RMS, failed evaluations, late objective
gain and boundary proximity (none certifies a global optimum). For MCMC it
checks each parameter's rank-normalized R̂ < 1.01, bulk and tail ESS ≥ 400,
reports MCSE(mean)/posterior SD, and requires ≥ 16 exact posterior simulator
comparisons with heuristic 90th-percentile |Δ log density| < 1. Review trace,
rank and autocorrelation plots, exact-vs-GP plots and residuals; a small sample
of exact draws cannot guarantee surrogate fidelity globally. **Even a green
numerical report cannot establish that the likelihood, priors or physical
model are scientifically correct**. For published claims assess correlated
measurement errors, discrepancy, identifiability, prior sensitivity and
held-out observables. The analytic benchmark and independent-data SBC tests
check software behavior under their assumptions, not your actual experiment.

For automated regression checks run `python -m unittest -v test_workflow
test_mcmc test_user_tool` (set `LD_LIBRARY_PATH` as in Installation if using
the RHEL8 dependencies on this machine).

### Example outputs (tested synthetic runs, not measured EOS data)

| Run | Result | What the report means |
| --- | --- | --- |
| Local pattern search | 87 successful evaluations; best RMS 0.125 | Best *observed* fit; objective still improved ~41% in its last fifth of evaluations, so a larger budget is warranted. |
| Native batch EGO | 104 successful evaluations; best RMS 0.0594 | Better observed fit here, but this is neither a global-optimum certificate nor a parameter uncertainty interval. |
| Analytic two-parameter DREAM benchmark | 12,000 retained draws; max R̂ 1.0034; min bulk/tail ESS 1148/1547; GP log-density error p90 0.000058 | Numerical checks passed; scientific validity of other models remains unknown. |
| Four-parameter curve DREAM | 12,000 retained draws; max R̂ 1.035; min bulk/tail ESS 124/396; GP log-density error p90 17.4 | **Not inference-ready**: neither its corner plot nor its 95% intervals support parameter claims. |

The retained benchmark and curve reports are under `runs/` (use `python tool.py
list` to find the latest). Different seeds, priors, data or budget can change
these values. A good-looking curve fit cannot override chain or emulator checks.

An executable **local demonstration** and a Slurm-backed Dakota workflow. All
expensive evaluations go through the same interface: Dakota standard parameters
→ `driver.py` → physical dictionary → `sbatch`/local simulation → curve CSV →
RMS → one-number Dakota results file. The built-in simulator is **only a demo**.

## Hydrodynamic/EOS calibration: tested native DREAM MCMC

An additional, **actually executed with Dakota 6.23** Bayesian calibration
pipeline is in `mcmc_run.py`, `mcmc_driver.py`, `mcmc_analyze.py`,
`mcmc_validate.py`, `mcmc_simulate.py`, `sbc_benchmark.py`. The installed binary
rejects `bayes_calibration queso` because QUESO is not compiled in; we use
Dakota's available `bayes_calibration dream`, four parallel Markov chains and
native `gaussian_process dakota` **curve-output** emulation instead. `chain_samples`
is the total across four chains, **not** samples per chain. The 48 model-building
simulations can run at `CONCURRENCY`; surrogate MCMC is cheap and should not
submit a simulation per draw. In an expensive cluster run start with 48–128
training points for 2–4 active parameters and 24,000 total chain samples;
validate surrogate error and iterate rather than assuming these fixed counts
work for 20D models. DOE + native EGO is still useful for optimization; chain
calibration consumes the **full curve**, not the optimized scalar RMS.

For a self-contained Gaussian linear benchmark with a known posterior, run from
the project directory with Dakota and a Python environment with NumPy, SciPy,
Matplotlib, ArviZ, and corner:

    python mcmc_run.py --mode benchmark --samples 24000 --build-samples 48
    python sbc_benchmark.py --repetitions 500
    python -m unittest -v test_mcmc test_workflow

For the existing example four-parameter curve model, run:

    python mcmc_run.py --mode curve --reference runs/2026-10-04_121212_bo/reference.csv --max-ordinates 9 --sigma 0.05 --samples 24000 --build-samples 128

Replace the reference path with **measured** hydrodynamics data and supply a
*measured, defensible* per-ordinate observation SD via `--sigma`. Dakota's
`variance_type = 'scalar'` calibration data records the **variance** σ², not σ;
`mcmc_run.py` squares `--sigma`. An incorrectly supplied SD in place of variance
made a test posterior >4× too wide, and the analytic benchmark caught it.
The curve mode evenly selects up to `--max-ordinates` reference locations (default
15) to keep separate GP models tractable; this changes the likelihood. The
simulator is `simulate.py` until replaced; for benchmark mode it is an explicit
two-parameter linear curve with a closed-form Gaussian posterior. Backend
`BACKEND="slurm"` submits model-building and exact-validation evaluations;
`BACKEND="allocation"` uses exclusive `srun` steps in a shared allocation.
`--validate-samples 16` (default) pays for 16 *additional exact simulations*
at retained posterior draws. Increase samples/build points if the validation
fails, but inspect noise, prior, model form, and identifiability first.

Each MCMC run saves `dakota.in`, `dakota_console.log` (verbose Dakota progress), full `config.json`, the actual `reference.csv`,
`calibration.dat`, `chain.dat`, four original `dakota_dream_chain*.txt` files,
`posterior_summary.json`, `validation.json`, plus `plots/corner.png`,
`plots/trace.png`, `plots/surrogate_validation.png`,
`plots/exact_posterior_curves.png` (with residuals), and
`params/validation_*/exact_validation_curves.csv`; the benchmark also saves
`plots/benchmark_marginals.png` and `plots/posterior_predictive.png`. The four chain files are used (rather than
mixing their flattened export) for ArviZ rank-normalized R̂ and bulk ESS, with
the first **half of each chain discarded**. Validate at least R̂ < 1.01 and bulk
ESS > 400; inspect trace/rank plots and increase run length as needed. A
`validation.json` with `inference_ready=false` explicitly marks exploratory
corner plots **not fit for interval claims**. The status only checks numerical
mixing and approximate-vs-exact log likelihood at a small sample; it does not
validate the physical model or independence of measurements. Synthetic tests:
the 24,000-draw benchmark achieved R̂ ≤ 1.004, ESS ≥ 1148 and 90th-percentile
exact-vs-GP log-likelihood difference < 0.001; the 4-parameter toy curve run
with 12,000 draws and 128 training points did **not** mix adequately (ESS < 24),
so do not interpret its parameter intervals.

The `sbc_benchmark.py` KS test uses **500 independently generated datasets**,
each with θ drawn from the bounded prior and Gaussian noise, and evaluates the
*exact truncated-normal posterior* PIT at the generating θ. Independent PITs
may validly be checked against Uniform(0,1): fixed-seed KS p-values were
0.39 (intercept) and 0.35 (slope). It checks the likelihood math, **not** Dakota
sampling; `test_mcmc.py` separately tests DREAM results against the analytic
posterior and uses a deliberately wrong PIT as a negative control. Do **not**
use an ordinary KS p-value on autocorrelated MCMC draws: `descriptive_marginal_ks_D`
reports only a distance.

Physics caution: for real hydro/EOS inference, specify physically admissible
priors (e.g. causality/stability/domain constraints), test identifiability and
simulator stochasticity, account for correlated observation errors and model
discrepancy, and consider validation on held-out observables. This example's
independent Gaussian residual likelihood, uniform bounds, equal σ and numerical
GP are *explicit modeling assumptions*, not universal EOS conventions. If
measurements are correlated, adapt Dakota's covariance-aware calibration data
format and analysis before interpreting posterior probabilities. See
[Dakota 6.23 release](https://github.com/snl-dakota/dakota/releases/tag/v6.23.0),
[Vehtari et al. (2021), rank-normalized R̂ and ESS](https://doi.org/10.1214/20-BA1221),
[Talts et al., simulation-based calibration](https://arxiv.org/abs/1804.06788),
and [Kennedy & O'Hagan (2001), computer-model calibration and discrepancy](https://doi.org/10.1111/1467-9868.00294).

## Installation and quick start

Use a Dakota 6.23 CLI binary from the [official releases](https://github.com/snl-dakota/dakota/releases/tag/v6.23.0), or your cluster's Dakota module. The **actual** binary on this development machine was installed under `$HOME/.local/opt/dakota-6.23.0-public-rhel8.Linux.x86_64-cli/bin/dakota`. This Ubuntu machine also required private HDF5 1.10, BLAS, LAPACK, and libnl runtime libraries; the RHEL8 binaries are not guaranteed portable. On a cluster use its own functional `dakota` command instead.

Python 3.10+ with NumPy and Matplotlib is needed for plots, and scikit-learn/SciPy for diagnostic GP/UQ. The objective driver uses only the standard library. The driver uses the **same interpreter** as `run.py`; make it accessible on compute nodes. This development machine's scikit-learn also needs the private `libgomp` directory in the `LD_LIBRARY_PATH` shown below; clusters should provide their own OpenMP runtime. For this machine, use:

    export LD_LIBRARY_PATH="$HOME/.local/opt/dakota-6.23-deps/usr/lib/x86_64-linux-gnu:$HOME/.local/opt/dakota-6.23-deps/usr/lib/x86_64-linux-gnu/blas:$HOME/.local/opt/dakota-6.23-deps/usr/lib/x86_64-linux-gnu/lapack${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DAKOTA="$HOME/.local/opt/dakota-6.23.0-public-rhel8.Linux.x86_64-cli/bin/dakota"
    PYTHON=/home/roypo/miniconda3/envs/scientific/bin/python
    "$PYTHON" run.py opt --dakota "$DAKOTA"
    "$PYTHON" run.py bo --dakota "$DAKOTA"

On another machine, from this directory: `python run.py opt` and `python run.py bo` (when Dakota is on PATH). Both commands now automatically write the plots after successful optimization; use `--no-plots` on a cluster without plotting dependencies, and generate them later with `python plot_results.py RUN_DIRECTORY`. `python run.py bo --prepare-only` generates a unique run and its **actual Dakota input** without executing it. Validate with `cd RUN_DIRECTORY && dakota -i dakota.in -check`. Example manual plotting command: `python plot_results.py runs/2026-10-04_121212_bo`. Optional exploratory noise-conditional candidate screening: `python uncertainty.py RUN_DIRECTORY --sigma 0.05` (only if 0.05 is an independently justified per-curve-point noise SD). Run local tests with `python -m unittest -v test_workflow` (the tests include 2D BO plots and numerical GP behavior).

## Files, configuration and output

`config.py` supplies defaults for direct calls without `--settings`; an editable JSON template overrides these defaults for template-driven runs. `run.py` maps each parameter to a Dakota `x_NAME` variable in [0,1], writes a unique `runs/YYYY-MM-DD_HHMMSS_KIND/` with `dakota.in` and `config.json` (a frozen run snapshot), then runs Dakota. In demo mode the target is synthetic; in measured mode the existing CSV supplied by `reference` or `--reference` is copied into the run. Do not mix runs with different references. The values in the demo simulator are not fit priors.

Each run contains `params/eval.N/` with Dakota parameters/results, physical parameter JSON and diagnostic JSON, and `results/eval.N.csv` with a **retained fitted curve**. `evaluations.csv` stores completion-order id, UTC timestamp, physical parameters, RMS, status, Slurm id, runtime, and curve path. Concurrent appends use advisory file locks on a shared POSIX filesystem. `best.json` records the best *successful* observed parameters and objective; `dakota.out`, `dakota.err`, `dakota_tabular.dat`, `dakota.rst`, per-job `slurm-*.out/err` and `plots/*.png` preserve convergence/debug history. The penalty for failure is `FAILURE_PENALTY` (default 1e6); full tracebacks are in `logs/eval.N.error.txt`. A failed evaluation is never considered a valid fit. The recorded parameters are **physical**, while Dakota's tabular parameters are normalized.

On a cluster choose `BACKEND = "slurm"` (one `sbatch`/evaluation) or `BACKEND = "allocation"` (`srun` steps in a preallocated job). Edit `CPUS_PER_EVALUATION`, `MEMORY`, `TIME_LIMIT`, `PARTITION`, `JOB_TIMEOUT_SECONDS` and `POLL_SECONDS` in `config.py`. `run_slurm.sh` has fallback `#SBATCH --cpus-per-task=1`, `--mem=2G`, `--time=00:05:00`; in `slurm` mode, `driver.py` overrides those with sbatch CLI flags from the frozen run configuration. The `allocation` mode instead takes memory/total time from `submit_allocation.sh`; directives in `run_slurm.sh` do not apply to `srun`. Make sure Slurm has `sbatch`, `squeue`, `sacct` (working accounting), and `scancel` for the first mode; the executable and reference paths must be visible on compute nodes. The driver verifies `sacct` COMPLETED and exit 0 and refuses to treat missing accounting as success. A queued job that exceeds the wall-clock **submission-to-finish** timeout is cancelled. If the simulation uses MPI, update its launch command in `run_slurm.sh` and its resource request; don't multiply Dakota concurrency by CPUs unintentionally.

Replace `simulate_curve()` in `simulate.py` or change the shell script to invoke your existing simulation. It must read `physical_params.json` and write `curve.csv` with columns `x,y` on the same strictly increasing x grid as `reference.csv`. `driver.py` explicitly checks missing/malformed/nonfinite data and x mismatch; if your grids differ, implement documented interpolation in `driver.py` before RMS. The local backend executes the same script with Bash for testing but does not submit Slurm.

## Methods and parallelism

**Deterministic minimization:** Dakota `asynch_pattern_search` (HOPSPACK/APPS), `synchronization nonblocking`, normalized bounded variables, no derivatives. Robust parallel derivative-free local search; not a global-optimality certificate. Other choices: Nelder–Mead/simplex is less naturally parallel; `mesh_adaptive_search` (NOMAD) is robust but evaluate its cluster concurrency on your build; global evolutionary methods may use many more runs. Restart with different starting points for multimodal surfaces.

**Native Bayesian-style optimization:** Dakota 6.23 `efficient_global`, `gaussian_process dakota`, `initial_samples`, `batch_size`, `max_iterations`, seed. This is native GP **efficient global optimization**, selecting points using expected improvement with batch selection; not full Bayesian inference over physical parameters. It uses GP predictive uncertainty for acquisition. The driver **does not** implement or override Dakota's acquisition. `BO_BATCHES * BO_BATCH_SIZE` is passed as Dakota's `max_iterations` because 6.23 counts acquisition point iterations, not whole batches (verified with a live 6.23 run). Budget is approximately initial samples + iterations, though convergence can stop early. Dakota's internal GP is not serialized here; `plot_results.py` independently refits a Matérn GP on **successful evaluated data** to visualize predictive mean and conditional standard deviation. It is **not Dakota's actual surrogate/acquisition function**; no acquisition trace is claimed. Penalties in EGO training can distort its GP: investigate failures before trusting EGO results. EGO can be sample-hungry or fragile for 20 dimensions; reduce dimensionality or do sensitivity screening first.

Both methods use `fork`, `asynchronous evaluation_concurrency = CONCURRENCY`, and tagged per-evaluation work directories. **Strategy A (recommended when queue latency is small):** set `BACKEND="slurm"`. Run Dakota on a login/interactive allocation *only if cluster policy permits*: 8/16/32 concurrent lightweight Python driver processes each submit exactly **one Slurm job** and wait. For 16 × 4 CPU, set `CONCURRENCY=16`, `CPUS_PER_EVALUATION=4` (up to 64 simulation CPUs). For 16 × 1 CPU, set 16 and 1. **Strategy B (recommended for significant per-job queue delay):** set `BACKEND="allocation"`, configure `submit_allocation.sh` with `--ntasks=16 --cpus-per-task=4 --mem=... --time=...` for 16 × 4 (or `--cpus-per-task=1` for 16 × 1), match those values in `config.py`, and submit `sbatch submit_allocation.sh bo` (or `opt`). Dakota and all 16 exclusive `srun` simulation steps run inside **one** allocation; only the initial allocation queues. Ensure total RAM and wall time cover the entire optimization, not one simulation. Each `srun` exit code is checked; `srun` timeout fails that evaluation. This is not a Slurm array: arrays suit a fixed set of independent points but are awkward for adaptive EGO. Do **not** request 16 CPUs *within* each of 16 evaluations unless each simulation actually needs 16. Dakota concurrency caps *in-flight evaluations*, Slurm determines resource placement, and CPUs per job control *within-evaluation* parallelism. EGO first design is parallel; acquisition batches parallelize up to `BO_BATCH_SIZE`; later batches depend on earlier results. Asynchronous pattern search can fill idle slots; Dakota may evaluate a few extra points while stopping. With 45 s evaluations and zero queue latency, 80 evaluations have lower-bound wall times ~3600/450/225/113 s at 1/8/16/32 concurrent slots respectively (plus optimization overhead and partially filled waves); 104 EGO evaluations ~4680/585/293/147 s, **plus GP refits and batch barriers**. Queue waits may dominate Strategy A.

## Plotting and uncertainty

Plots: `convergence.png` (raw + best-so-far; EGO initial labels and failed-evaluation markers), `parameters.png`, `landscape.png` (parameter/objective projections for all dimensions), `curve_fit.png` (curve + residual with RMS verified from saved CSVs), and for EGO `surrogate_diagnostics.png` (independently fitted GP 1D slices). A 2-parameter EGO run also has `landscape_2d_gp.png` (mean, std, evaluated points, best). Plots label projections as projections; they are **not** 20-dimensional marginal objective maps. The plotting GP uses a small numerical nugget and **assumes a deterministic objective**; if the real simulation is stochastic, replace this with an independently estimated observation-noise model before interpreting its bands. Neither the 1D slices nor GP std are Dakota's internal acquisition trace.

- Optimization-location uncertainty: multiple competitive minima and incomplete exploration; compare restarts, local refinement, and stability to additional evaluations. Dakota's single best point does not quantify it.
- GP predictive uncertainty: epistemic uncertainty conditional on surrogate assumptions and sampling; even a deterministic simulator has this uncertainty away from samples. Diagnostic GP bands are not frequentist parameter confidence intervals.
- Parameter uncertainty: requires a **likelihood and prior** for Bayesian credible regions, or a sampling model and repeated observations for confidence intervals. For independent Gaussian curve observations with known SD σ and forward model `m(x;θ)`, log likelihood is `−Σ_i (y_i−m_i(θ))²/(2σ²)`; with N equally weighted points, `Σ residual² = N·RMS²`. Posterior is proportional to that likelihood times a specified prior on θ. For correlated/heteroscedastic observations use their covariance; RMS alone is then insufficient. Dakota Bayesian calibration is appropriate once the **curve residual vector** (not just optimized RMS) and observation covariance/prior are provided; it is intentionally not represented here as a calibrated posterior.
- Objective noise: replicate the *same* parameter point to estimate its variance, store repetitions (Dakota cache may skip identical points, so plan replications in the evaluator or deactivate its evaluation cache), then use an explicit noise model. GP std from a noiseless surface is not simulation-noise SD.

`uncertainty.py --sigma ...` reports only observed parameter ranges among sampled points below an asymptotic χ² likelihood-ratio threshold, given independently justified Gaussian noise. These are **screened sampled candidates, not confidence or credible intervals**: adaptive sparse samples can omit large parts of the true region. Practical rigorous approaches: bootstrap *measured curves* and reoptimize each replicate, or construct curve-level likelihood/prior and perform Bayesian calibration (possibly on a validated local surrogate, checking with fresh exact simulations). Budget for UQ can exceed optimization significantly; e.g. 50 bootstrap replicates × 50 evaluations = 2500 extra expensive calls, and should be parallelized within resource limits.

Recommended sequence: (1) validate the wrapper and target on local/one Slurm job; (2) initial EGO design of roughly 5–10 points per dimension (24 here is a cheap demonstration, not sufficient for 20D); (3) native batched EGO for exploration, or start with APPS if the initial region is good; (4) derivative-free local refinement from promising points with ~80–300 calls; (5) replicate/noise or calibrated curve-level UQ as justified; (6) inspect final curve and residual plots. Real costs depend strongly on landscape complexity.

## What I need to change for my actual problem

1. Parameter names, physical bounds, starting values, concurrency and budgets in an editable template (or `config.py` for direct controller calls without templates).
2. Existing measured x,y reference CSV set in the template or passed as `--reference`; no change to the synthetic reference generation is needed.
3. `simulate.py` or `run_slurm.sh` with the real Slurm simulation command, actual output curve loader/CSV writer, and grid alignment in `driver.py` if required.
4. Cluster-specific Python/Dakota paths, Slurm partition, CPUs, memory, time limit and timeout; choose `BACKEND="slurm"` or `"allocation"`, and set the matching resources in `submit_allocation.sh` for allocation mode.
5. Independently measured observation-noise model/prior if actual parameter intervals are needed.