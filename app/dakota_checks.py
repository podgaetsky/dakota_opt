"""Check Dakota runtime and optional QUESO method without starting a chain."""

import subprocess
import tempfile
from pathlib import Path

from mcmc_run import BENCHMARK, calibration_record, input_text


def check_dakota(dakota, env, python_executable):
    """Return evidence from the executable and a real QUESO method-instantiation check."""
    version = subprocess.run([dakota, "-v"], env=env, capture_output=True, text=True,
                             timeout=30, check=False)
    version_text = (version.stdout + version.stderr).strip()
    result = {"version": version_text, "version_exit_code": version.returncode,
              "queso": "not_checked", "queso_detail": ""}
    if version.returncode:
        result["queso_detail"] = "Dakota could not start; check the executable and shared libraries"
        return result
    with tempfile.TemporaryDirectory(prefix="dakota-queso-check-") as temp:
        run = Path(temp)
        (run / "calibration.dat").write_text(calibration_record([{"y": "0"}] * 9, .05))
        (run / "dakota.in").write_text(input_text(run, BENCHMARK, 9, 100, 8, 491,
                                                   "queso", python_executable))
        try:
            checked = subprocess.run([dakota, "-i", "dakota.in", "-check"], cwd=run, env=env,
                                     capture_output=True, text=True, timeout=60, check=False)
        except subprocess.TimeoutExpired:
            result.update(queso="unknown", queso_detail="QUESO method check timed out")
            return result
        output = checked.stdout + checked.stderr
        result["queso_exit_code"] = checked.returncode
        if "QUESO Bayesian calibration method unavailable" in output:
            result.update(queso="unavailable", queso_detail="Dakota build lacks QUESO (HAVE_QUESO)")
        elif checked.returncode == 0:
            result.update(queso="available", queso_detail="Dakota -check instantiated bayes_calibration queso")
        else:
            result.update(queso="unknown", queso_detail=output.strip()[-2000:])
    return result
