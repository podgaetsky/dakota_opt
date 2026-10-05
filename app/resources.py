"""Freeze template resources into per-run Slurm scripts and cluster settings."""

import json
import shlex
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent

EVALUATION = Template("""#!/bin/bash
#SBATCH --job-name=dakota-eval
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$memory
#SBATCH --time=$time_limit
$partition
set -euo pipefail
$modules
$library_path
exec "$$@"
""")

ALLOCATION = Template("""#!/bin/bash
#SBATCH --job-name=dakota-controller
#SBATCH --nodes=1
#SBATCH --ntasks=$concurrency
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$memory
#SBATCH --time=$time_limit
$partition
set -euo pipefail
$modules
$library_path
exec $python $tool $mode --workdir $workdir $template $dakota
""")


def cluster_settings(name):
    """Resolve a named, user-owned cluster profile; never guess machine paths."""
    path = ROOT / "clusters" / f"{name}.json"
    if not name.isidentifier() or not path.is_file():
        raise FileNotFoundError(f"Cluster profile not found: {path}; add clusters/{name}.json")
    profile = json.loads(path.read_text())
    allowed = {"partition", "modules", "python", "dakota", "ld_library_path"}
    if set(profile) - allowed or not isinstance(profile.get("modules", []), list):
        raise ValueError("Cluster profile keys: partition, modules, python, dakota, ld_library_path")
    return profile


def write_scripts(run, settings, cluster=None, mode=None, template=None, dakota=None, workdir=None,
                  cluster_name=None):
    """Generate scripts from one frozen resource snapshot, not checked-in SBATCH hints."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    profile = cluster or {}
    modules = "\n".join(f"module load {shlex.quote(module)}" for module in profile.get("modules", []))
    libraries = profile.get("ld_library_path", [])
    if not isinstance(libraries, list):
        raise ValueError("ld_library_path must be a list of directories")
    library_path = ("export LD_LIBRARY_PATH=" + shlex.quote(":".join(libraries)) + ":${LD_LIBRARY_PATH:-}"
                    if libraries else "")
    values = {"cpus": settings["cpus_per_evaluation"], "memory": settings["memory"],
              "time_limit": settings["time_limit"], "concurrency": settings["concurrency"],
              "partition": f"#SBATCH --partition={settings['partition']}" if settings.get("partition") else "",
              "modules": modules, "library_path": library_path}
    evaluation = run / "run_slurm.sh"
    evaluation.write_text(EVALUATION.substitute(values))
    evaluation.chmod(0o755)
    if mode is not None:
        values.update(python=shlex.quote(settings["python_executable"]),
                      tool=shlex.quote(str(ROOT / "app" / "tool.py")), mode=shlex.quote(mode),
                      workdir=shlex.quote(str(workdir)), template=shlex.quote(str(template)),
                     dakota=(f"--dakota {shlex.quote(str(dakota))}" if dakota else "") +
                         (f" --cluster {shlex.quote(cluster_name)}" if cluster_name else ""))
        allocation = run / "submit_allocation.sh"
        allocation.write_text(ALLOCATION.substitute(values))
        allocation.chmod(0o755)
    return evaluation
