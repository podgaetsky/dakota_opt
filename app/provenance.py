"""Immutable-at-creation input provenance for Dakota run directories."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _command(*command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def write_provenance(run, settings, dakota=None, reference_source=None):
    run = Path(run).resolve()
    script = Path(settings.get("simulation_script", "simulate.py"))
    if not script.is_absolute():
        script = APP / script
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "scikit-learn", "arviz", "corner"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    status = _command("git", "status", "--porcelain")
    info = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _command("git", "rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None,
            "python_version": sys.version, "platform": platform.platform(), "packages": packages,
            "dakota_version": _command(dakota, "-v") if dakota else None,
            "command_line": sys.argv, "parameters_and_resources": settings,
            "reference_source": str(reference_source) if reference_source else None,
            "reference_sha256": digest(run / "reference.csv"), "simulator_path": str(script),
            "simulator_sha256": digest(script) if script.is_file() else None,
            "noise_specification": {key: settings.get(key) for key in ("sigma", "diagnostic_sigma") if key in settings},
            "slurm_job_ids": "See per-evaluation diagnostic.json or evaluations.csv (assigned after run)."}
    (run / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
