"""Validate user-editable JSON templates without dynamic code execution."""

import json
import math
import shlex
import string
import warnings
from pathlib import Path

import config

CONFIG_KEYS = {
    "parameters": "PARAMETERS", "backend": "BACKEND", "concurrency": "CONCURRENCY",
    "cpus_per_evaluation": "CPUS_PER_EVALUATION", "memory": "MEMORY",
    "time_limit": "TIME_LIMIT", "partition": "PARTITION",
    "job_timeout_seconds": "JOB_TIMEOUT_SECONDS", "poll_seconds": "POLL_SECONDS",
    "failure_penalty": "FAILURE_PENALTY", "opt_evaluations": "OPT_EVALUATIONS",
    "bo_initial_samples": "BO_INITIAL_SAMPLES", "bo_batches": "BO_BATCHES",
    "bo_batch_size": "BO_BATCH_SIZE", "seed": "SEED",
}
DEFAULTS = {attr: getattr(config, attr) for attr in CONFIG_KEYS.values()}


def load_settings(path, reference_override=None):
    """Apply settings for the *current controller process*; runs freeze a snapshot."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    allowed = set(CONFIG_KEYS) | {"mode", "reference", "mcmc", "simulation_script", "simulation", "resources", "execution"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown template keys: {sorted(unknown)}")
    for attr, value in DEFAULTS.items():
        setattr(config, attr, value)
    if data.get("mode") not in ("demo", "measured"):
        raise ValueError("Template must set mode to demo or measured")
    if not isinstance(data.get("parameters"), dict) or not data["parameters"]:
        raise ValueError("Template needs a nonempty parameters dictionary")
    for name, spec in data["parameters"].items():
        if not name.isidentifier() or name.startswith("x_"):
            raise ValueError(f"Invalid parameter name {name!r}")
        if set(spec) != {"initial", "lower", "upper"}:
            raise ValueError(f"Parameter {name} needs initial, lower, upper")
        low, x, high = (float(spec[k]) for k in ("lower", "initial", "upper"))
        if not low < high or not low <= x <= high:
            raise ValueError(f"Invalid physical bounds for {name}")
    for key, attr in CONFIG_KEYS.items():
        if key in data:
            setattr(config, attr, data[key])
    if "backend" in data:
        warnings.warn("backend is deprecated; use execution", DeprecationWarning, stacklevel=2)
    if "execution" in data:
        if "backend" in data:
            raise ValueError("Use execution instead of backend, not both")
        config.BACKEND = data["execution"]
    resources = data.get("resources", {})
    resource_keys = {"partition": "PARTITION", "cpus": "CPUS_PER_EVALUATION",
                     "mem": "MEMORY", "time": "TIME_LIMIT", "concurrency": "CONCURRENCY",
                     "timeout": "JOB_TIMEOUT_SECONDS"}
    if not isinstance(resources, dict) or set(resources) - set(resource_keys):
        raise ValueError("resources expects partition, cpus, mem, time, concurrency, timeout")
    for key, attr in resource_keys.items():
        if key in resources:
            if next(k for k, v in CONFIG_KEYS.items() if v == attr) in data:
                raise ValueError(f"resources.{key} conflicts with flat resource setting; use one location")
            setattr(config, attr, resources[key])
    if config.BACKEND not in {"local", "slurm", "allocation"}:
        raise ValueError("backend must be local, slurm or allocation")
    for key in ("concurrency", "cpus_per_evaluation", "opt_evaluations",
                "bo_initial_samples", "bo_batches", "bo_batch_size"):
        value = getattr(config, CONFIG_KEYS[key])
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.BO_BATCH_SIZE > config.CONCURRENCY:
        raise ValueError("bo_batch_size cannot exceed concurrency")
    for key in ("job_timeout_seconds", "poll_seconds", "failure_penalty"):
        value = getattr(config, CONFIG_KEYS[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number; set {key} to a value greater than zero")
    if type(config.SEED) is not int or config.SEED < 0:
        raise ValueError("seed must be a nonnegative integer; set seed to 0 or greater")
    for key in ("memory", "time_limit", "partition"):
        value = getattr(config, CONFIG_KEYS[key])
        if not isinstance(value, str) or (key != "partition" and not value.strip()) or "\n" in value:
            raise ValueError(f"{key} must be a single-line string; set a valid Slurm {key}")
    mcmc = data.get("mcmc", {})
    if not isinstance(mcmc, dict) or set(mcmc) - {"samples", "build_samples", "sigma", "max_ordinates", "validate_samples", "backend", "sampler"}:
        raise ValueError("Invalid mcmc settings; check template keys")
    if "backend" in mcmc:
        warnings.warn("mcmc.backend is deprecated; use mcmc.sampler", DeprecationWarning, stacklevel=2)
    if "sampler" in mcmc:
        if "backend" in mcmc:
            raise ValueError("Use mcmc.sampler instead of mcmc.backend, not both")
        mcmc = {**mcmc, "backend": mcmc["sampler"]}
    if mcmc.get("backend", "dream") not in {"dream", "queso"}:
        raise ValueError("mcmc.backend must be dream or queso")
    for key, minimum in (("samples", 100), ("build_samples", 8), ("validate_samples", 0),
                         ("max_ordinates", 2)):
        value = mcmc.get(key)
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"mcmc.{key} must be an integer >= {minimum}; change mcmc.{key}")
    sigma = mcmc.get("sigma")
    if sigma is not None and (isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or
                              not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("mcmc.sigma must be a positive finite observation SD; set mcmc.sigma > 0")
    script = data.get("simulation_script")
    simulation = data.get("simulation")
    if simulation is not None:
        if script is not None:
            raise ValueError("Use simulation.command instead of simulation_script, not both")
        if not isinstance(simulation, dict) or set(simulation) != {"command", "workdir"}:
            raise ValueError("simulation needs command (string) and workdir (directory path)")
        command, workdir = simulation["command"], simulation["workdir"]
        if not isinstance(command, str) or not command.strip():
            raise ValueError("simulation.command must be a nonempty command string; use {params} and {curve}")
        if not isinstance(workdir, str) or not workdir.strip():
            raise ValueError("simulation.workdir must be a directory path relative to the template")
        try:
            tokens = shlex.split(command)
        except ValueError as error:
            raise ValueError(f"simulation.command has invalid quoting: {error}") from error
        fields = {field for token in tokens for _, field, _, _ in string.Formatter().parse(token) if field}
        if not tokens or not {"params", "curve"}.issubset(fields):
            raise ValueError("simulation.command must include {params} and {curve} placeholders")
        if fields - {"params", "curve", "reference", "work", "python"}:
            raise ValueError(f"simulation.command unknown placeholders: {sorted(fields)}; use params, curve, reference, work or python")
        workdir = Path(workdir).expanduser()
        if not workdir.is_absolute():
            workdir = path.parent / workdir
        if not workdir.is_dir():
            raise FileNotFoundError(f"simulation.workdir not found: {workdir}; create the directory")
        simulation = {"command": command, "workdir": str(workdir.resolve())}
    if script is not None:
        if not isinstance(script, str) or not script.strip():
            raise ValueError("simulation_script must be a path to an existing Python file")
        script = Path(script).expanduser()
        if not script.is_absolute():
            script = path.parent / script
        script = script.resolve()
        if not script.is_file() or script.suffix != ".py":
            raise FileNotFoundError(f"Simulation Python script not found: {script}")
    reference = reference_override if reference_override is not None else data.get("reference")
    if data["mode"] == "measured":
        if not isinstance(reference, (str, Path)) or not str(reference).strip():
            raise ValueError("Measured template needs a reference path (x,y CSV)")
        reference = Path(reference)
        if not reference.is_absolute():
            reference = (Path.cwd() if reference_override is not None else path.parent) / reference
        if not reference.is_file():
            raise FileNotFoundError(f"Measured reference not found: {reference}. Replace the template placeholder.")
    else:
        if set(config.PARAMETERS) != {"p1", "p2", "p3", "p4"}:
            raise ValueError("Synthetic demo requires the four demo parameters; use measured mode for real data")
    return {"mode": data["mode"], "reference": str(reference) if reference else None,
            "simulation_script": str(script) if script else None, "simulation": simulation, "mcmc": mcmc,
            "resources": resources, "settings_path": str(path)}
