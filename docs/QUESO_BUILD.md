# Dakota 6.23 with QUESO (local, no Slurm)

The official RHEL8 Dakota 6.23 CLI release used on this development machine
reports `QUESO Bayesian calibration method unavailable` at `-check`; it then
exits with a segmentation fault. It must not be used for QUESO benchmarks.
The official **source** distribution includes QUESO. Obtain
`dakota-6.23.0-public-src-cli.tar.gz` from the v6.23.0 GitHub release and
verify SHA-256 `f8461a10d460333a539a3e6d319d45bd44a18d06c862ad8204294cfb418ac8dc`.
No vendor source, compiler, or binary is checked into this repository.
GSL is GPL-licensed; review the licenses of Dakota, QUESO, GSL and all linked
components before redistributing any binary built this way.

To reproduce this build, install CMake >=3.23, Ninja, C/C++/Fortran compilers,
GSL, Boost, and BLAS/LAPACK in an isolated environment (this machine used
conda-forge packages `cmake ninja gcc_linux-64 gxx_linux-64 gfortran_linux-64
gsl boost-cpp openblas pkg-config`). Configure a separate CMake build directory
against the unpacked Dakota source using [the build script](../scripts/build_queso.sh)
with `SOURCE`, `BUILD`, `PREFIX`, `TOOLCHAIN` and optionally `JOBS` set in your
environment. It installs outside the repository. Its options are:

- `HAVE_QUESO=ON`, `DAKOTA_HAVE_GSL=ON`, `HAVE_QUESO_GPMSA=OFF`,
  `DAKOTA_HAVE_MPI=OFF`, and a supported BLAS/LAPACK.
- `HAVE_DREAM=ON` (the default), `HAVE_SURFPACK=ON` for Dakota's GP emulator.
- Disable unrelated options `HAVE_DDACE`, `HAVE_ROL`, `HAVE_JEGA`,
  `HAVE_NOMAD`, `HAVE_PSUADE`, `HAVE_NLPQL`,
  `HAVE_NL2SOL` and `HAVE_NPSOL` for the local benchmark.
  Keep `HAVE_HOPSPACK=ON`: the local optimization workflow uses
  `asynch_pattern_search`, which is unavailable without this package.
  Keep `HAVE_NCSUOPT=ON` because Surfpack requires it, and
  `HAVE_OPTPP=ON` because Dakota's Bayesian calibration includes its headers.
- Disable Dakota's own test executables (`DAKOTA_ENABLE_TESTS=OFF`,
  `BUILD_TESTING=OFF`) when installing only the application; the repository's
  opt-in integration tests still check the installed executable.
- On this conda-forge cross-compiler toolchain, `CMAKE_EXE_LINKER_FLAGS`
  needs `-lm` and `-Wl,-rpath-link,<toolchain>/lib`, and `DL_LIBRARY=dl`
  is required. The latter linker option resolves transitive Fortran runtime
  libraries (`libgfortran`, `libquadmath`) at the final C++ executable link.
  `DAKOTA_NO_FIND_TRILINOS=ON` prevents a subsequent CMake configure from
  mistaking its own partially built bundled Trilinos for an external install.
  At runtime set `LD_LIBRARY_PATH=<toolchain>/lib` (or provide equivalent
  loader configuration). System toolchains may not need these. Build the
  default target, not only `dakota`: installation also requires `lhsdrv`,
  `dakota_order_input`, and other auxiliary executables.

The locally built binary and any runtime library paths go in a git-ignored
`clusters/local.json` profile or the process environment, never in committed
paths. Run `python tool.py doctor --cluster local` first; QUESO is present only
if the actual `bayes_calibration queso` `-check` reports `available`.

`python benchmark_samplers.py --dakota /path/to/built/dakota --workdir /path/to/project`
runs both samplers on the exact synthetic linear benchmark with identical
reference, priors, observation sigma, seed, GP training budget, and requested
samples. Use `--mode curve templates/file_benchmark.json` for the shipped
file model. The comparison JSON records medians, 95% intervals, wall times,
DREAM convergence diagnostics, and (in synthetic benchmark mode) the analytic
Gaussian reference. QUESO's single chain does **not** certify convergence;
neither run is automatically inference-ready without separate exact-vs-GP
validation. Set `DAKOTA_INTEGRATION=1`, `DAKOTA` and runtime library paths
before running the opt-in full-workflow tests; set `DAKOTA_EXPECT_QUESO=1`
when testing a QUESO-enabled build.

## Verified local results (2026-10-05)

The script installed Dakota 6.23 (revision `a3b2eb477`) in an isolated
temporary prefix outside the repository. `tool.py doctor --cluster local`
reported `QUESO: available` after actually instantiating the QUESO method.
Both opt-in tests passed with `DAKOTA_INTEGRATION=1` and
`DAKOTA_EXPECT_QUESO=1`: a real file-model optimization using
`asynch_pattern_search` and the live QUESO method check. The default Python
suite passed 55 tests (the two opt-in tests skipped without the environment
flag). Paths under `/tmp` and the ignored local profile are machine-specific;
the build must be repeated if these paths are cleared.

The locally generated comparison JSON (under `runs/`, which is not committed)
uses the final installed binary, seed 491, nine identical observations,
bounded uniform priors, observation SD 0.05 (Dakota receives variance 0.0025),
48 GP build samples and **4,000 total requested draws per sampler**. DREAM
splits them into four 1,000-draw chains; QUESO exported one 4,000-draw chain.
After discarding the first half, each has 2,000 draws. The analytic unbounded
Gaussian benchmark mean is $(a,b)=(1.150556, 0.396000)$; the prior bounds are
far enough away for this reference to apply.

| Sampler | Wall seconds | Median $a$, $b$ | 95% width $a$, $b$ | R-hat $a$, $b$ | Bulk ESS $a$, $b$ |
| --- | ---: | --- | --- | --- | --- |
| DREAM | 4.533 | 1.151033, 0.390928 | 0.063942, 0.090260 | 1.027, 1.013 | 240, 225 |
| QUESO (DRAM) | 3.472 | 1.151187, 0.397899 | 0.070404, 0.100545 | unavailable | unavailable |

These are single-run wall times on this machine, not repeated timing
measurements. QUESO completed faster in this run, and both posterior medians
are close to the analytic reference; the intervals differ and **neither run
establishes inference readiness**. DREAM $\hat R$ exceeds 1.01 and has low ESS
at this short budget; the single QUESO chain has no multichain convergence
diagnostics. Exact-versus-GP log-density validation was explicitly disabled
for the comparison, so surrogate accuracy is unverified. Longer chains,
independent repeats, GP checks, and domain review of the noise model and priors
are required before scientific conclusions.
