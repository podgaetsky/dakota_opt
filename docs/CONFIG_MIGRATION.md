# Template migration and model contract

New projects should use `python app/tool.py init /path/to/project`. The resulting
`template.json`, `model.py`, and `reference.csv` belong to the user, not this
repository. Replace the reference placeholder with a real CSV before validation.
Use `python app/tool.py validate-model /path/to/project/template.json --workdir /path/to/project`
to check a single midpoint simulation before a full run.

A measured template may specify:

```json
"simulation": {
  "command": "{python} model.py --params {params} --out {curve} --reference {reference}",
  "workdir": "."
}
```

`workdir` and the model script are relative to the template. The command is
split into arguments (not executed in a shell). The driver writes
`physical_params.json`, copies `reference.csv` and checks that `curve.csv` has
finite `x,y` values on the same strictly increasing grid. `{python}` uses the
interpreter recorded in the frozen run configuration. Legacy
`simulation_script` templates remain supported. Saved results remain readable,
but runs created before scripts moved into `app/` can have absolute driver
paths in their Dakota input and need regeneration before restart.

New templates use `execution` (`local`, `slurm`, `allocation`) instead of
`backend`, and `mcmc.sampler` (`dream`, `queso`) instead of `mcmc.backend`.
Old names issue deprecation warnings. The optional `resources` object accepts
`partition`, `cpus`, `mem`, `time`, `concurrency`, and `timeout`; do not mix
these with their corresponding legacy flat fields. The resource snapshot is
used to generate per-run Slurm scripts. Use `--cluster NAME` with a profile
under `clusters/` to select a machine's modules and executable paths.

The direct controller scripts and flat defaults live in `app/`. Run them as
`python app/run.py ...` or `python app/mcmc_run.py ...`, not from the old root
paths. Dakota `-read_restart` reuses cached evaluations but does not guarantee
continuation of an interrupted method's internal state.
