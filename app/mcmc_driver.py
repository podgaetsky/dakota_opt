"""Calibration interface: one physical parameter set -> full curve-vector response.

Unlike the optimizer this must fail loudly on simulation failure: replacing a
residual vector with a scalar RMS penalty would change the posterior likelihood.
"""

import json
import math
import shutil
import sys
import traceback
from pathlib import Path

from driver import curve, parse_parameters, run_job


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: mcmc_driver.py params.in results.out")
    work = Path.cwd()
    run = work.parent.parent
    settings = json.loads((run / "config.json").read_text())
    try:
        params = parse_parameters(Path(sys.argv[1]), settings["parameters"])
        (work / "physical_params.json").write_text(json.dumps(params))
        shutil.copyfile(run / "reference.csv", work / "reference.csv")
        job = run_job(work, settings, settings["simulation_script"])
        target, prediction = curve(run / "reference.csv"), curve(work / "curve.csv")
        if len(target) != len(prediction) or any(not math.isclose(t[0], p[0], abs_tol=1e-10)
                                                  for t, p in zip(target, prediction)):
            raise ValueError("Calibration curve must match the observation x grid")
        Path(sys.argv[2]).write_text("".join(f"{y:.17g}\n" for _, y in prediction))
        (work / "diagnostic.json").write_text(json.dumps({"parameters": params, "slurm_job_id": job}))
    except Exception:
        (run / "logs" / f"{work.name}.error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()