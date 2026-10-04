"""Offline, portable HTML + JSON run audit; no external dashboard server needed."""

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path

PLOT_NAMES = {
    "convergence.png": "Objective evaluations and best-so-far",
    "curve_fit.png": "Best curve and residual",
    "parameters.png": "Parameter evolution",
    "landscape.png": "Evaluated projections (not global surfaces)",
    "surrogate_diagnostics.png": "Separately refitted GP diagnostics (not Dakota acquisition)",
    "chi_squared.png": "File-based χ² comparison and standardized residuals (assumed independent noise)",
    "landscape_2d_gp.png": "Two-dimensional independently refitted GP (not Dakota acquisition)",
    "corner.png": "Posterior corner: conditional on diagnostics and likelihood",
    "trace.png": "Per-chain traces and discarded warm-up",
    "rank.png": "Rank distributions by chain",
    "autocorrelation.png": "Autocorrelation by chain",
    "surrogate_validation.png": "Dakota GP versus exact simulator log density",
    "exact_posterior_curves.png": "Exact simulator predictions and residuals",
    "posterior_predictive.png": "Benchmark latent-curve interval",
    "benchmark_marginals.png": "Benchmark: posterior versus analytic reference",
}


def audit(run):
    run = Path(run).resolve()
    settings = json.loads((run / "config.json").read_text())
    names = list(settings["parameters"])
    checks = []
    metrics = {}
    recommendations = []
    if "mcmc_backend" in settings:
        kind = "mcmc"
        summary = json.loads((run / "posterior_summary.json").read_text())
        validation_path = run / "validation.json"
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
        for name in names:
            rhat = summary.get("rhat", {}).get(name)
            bulk = summary.get("ess_bulk", {}).get(name)
            tail = summary.get("ess_tail", {}).get(name)
            mcse = summary.get("mcse_mean_fraction_of_posterior_sd", {}).get(name)
            state = "pass" if (all(isinstance(v, (float, int)) and math.isfinite(v)
                                   for v in (rhat, bulk, tail, mcse)) and
                               rhat < 1.01 and bulk >= 400 and tail >= 400) else "warn"
            details = (f"R̂={rhat:.4f}, ESS bulk={bulk:.0f}, tail={tail:.0f}, "
                       f"MCSE(mean)/SD={mcse:.3f}" if all(isinstance(v, (float, int)) for v in (rhat, bulk, tail, mcse))
                       else "Incomplete chain diagnostics: rerun mcmc_analyze.py")
            checks.append({"label": f"Chain mixing · {name}", "state": state, "detail": details})
        if any(item["state"] != "pass" for item in checks):
            recommendations.append("Increase DREAM generations, inspect rank/trace plots for multimodality and consider physically justified reparameterization; do not report intervals yet.")
        if validation:
            error = validation.get("log_density_error_p90", math.inf)
            samples = validation.get("validated_samples", 0)
            gp_pass = (isinstance(error, (float, int)) and math.isfinite(error) and error < 1
                       and isinstance(samples, int) and samples >= 16)
            checks.append({"label": "GP emulator vs exact simulations",
                           "state": "pass" if gp_pass else "warn",
                           "detail": f"90th percentile |Δ log density|={error if not isinstance(error, (float, int)) else format(error, '.3g')} (heuristic <1); "
                                     f"{samples} exact posterior draws (require ≥16); not a global surrogate guarantee."})
            metrics.update({"exact_checks": samples, "gp_logp_error_p90": error})
            if not gp_pass:
                recommendations.append("Run at least 16 exact posterior checks; if GP error is high, add training points near posterior mass and rerun MCMC.")
        else:
            checks.append({"label": "GP emulator vs exact simulations", "state": "missing",
                           "detail": "No validation.json: execute mcmc_validate.py RUN before inference."})
        finite = lambda values: [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
        rhats = finite(summary.get("rhat", {}).values())
        bulk_ess = finite(summary.get("ess_bulk", {}).values())
        tail_ess = finite(summary.get("ess_tail", {}).values())
        metrics.update({"retained_draws": summary.get("retained_draws"),
                "max_rhat": max(rhats) if rhats else None,
                "min_bulk_ess": min(bulk_ess) if bulk_ess else None,
                "min_tail_ess": min(tail_ess) if tail_ess else None,
                "chain_move_fraction": summary.get("chain_move_fraction", {})})
        checks.append({"label": "Scientific likelihood / model discrepancy", "state": "unknown",
                       "detail": "Noise covariance, priors, simulator bias and held-out observables require domain validation; numerically good chains cannot check these."})
        if settings.get("mode") == "benchmark":
            checks.append({"label": "Analytic Gaussian benchmark", "state": "pass" if
                           all(v < .08 for v in summary.get("descriptive_marginal_ks_D", {}).values())
                           and bool(summary.get("descriptive_marginal_ks_D")) else "warn",
                           "detail": "Marginal KS D is descriptive only; autocorrelated MCMC draws have no ordinary KS p-value."})
        else:
            recommendations.append("Verify the assumed independent Gaussian per-ordinate noise SD using measurements or replicate simulations; inspect residual correlation and held-out observables.")
        numerical = all(c["state"] == "pass" for c in checks
                if c["label"].startswith(("Chain mixing", "GP emulator", "Analytic Gaussian benchmark")))
        headline = "Numerical checks passed · scientific validity unverified" if numerical else "Not inference-ready · review chain/GP checks"
        metrics["numerical_checks_passed"] = numerical
        metrics["posterior_intervals"] = summary.get("posterior", {})
    else:
        kind = "optimization"
        with (run / "evaluations.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        valid = [r for r in rows if r["status"] == "ok" and math.isfinite(float(r["objective"]))]
        if not valid:
            raise ValueError("No valid evaluations; inspect logs before reporting")
        objectives = [float(r["objective"]) for r in valid]
        best = min(valid, key=lambda r: float(r["objective"]))
        failures = len(rows) - len(valid)
        metrics = {"evaluations": len(rows), "successful": len(valid), "failed": failures,
                   "first_rms": objectives[0], "best_rms": float(best["objective"]),
                   "best_parameters": {name: float(best[name]) for name in names},
                   "method": settings["kind"]}
        checks.append({"label": "Simulation failures", "state": "pass" if failures == 0 else "warn",
                       "detail": f"{failures} / {len(rows)} evaluations failed or were nonfinite. Inspect per-evaluation error logs."})
        n = max(1, len(valid) // 5)
        prior_best = min(objectives[:-n]) if len(valid) > n else objectives[0]
        recent_gain = max(0., (prior_best - min(objectives[-n:])) / max(abs(prior_best), 1e-12))
        checks.append({"label": "Late objective improvement", "state": "warn" if recent_gain > .01 else "unknown",
                       "detail": f"Best changed by {recent_gain:.2%} over last 20% of completed successful evaluations. "
                                 "Small gain does not prove convergence; large gain suggests extending the budget."})
        metrics["late_relative_improvement"] = recent_gain
        chi2_file = run / "plots" / "chi_squared.json"
        if chi2_file.is_file():
            diagnostic = json.loads(chi2_file.read_text())
            metrics.update({"chi_squared": diagnostic["chi_squared"],
                            "reduced_chi_squared": diagnostic["reduced_chi_squared"],
                            "assumed_sigma": diagnostic["sigma"]})
            checks.append({"label": "Gaussian χ² model check", "state": "unknown",
                           "detail": "Computed from saved best curve; known independent Gaussian noise, effective degrees of freedom and model adequacy must be checked. A small χ² does not certify the optimizer."})
        at_edge = [name for name in names if min(
            (float(best[name]) - settings["parameters"][name]["lower"]) /
            (settings["parameters"][name]["upper"] - settings["parameters"][name]["lower"]),
            (settings["parameters"][name]["upper"] - float(best[name])) /
            (settings["parameters"][name]["upper"] - settings["parameters"][name]["lower"])) < .02]
        checks.append({"label": "Best point near parameter bounds", "state": "warn" if at_edge else "pass",
                       "detail": "Near (<2% of range): " + (", ".join(at_edge) if at_edge else "none")})
        checks.append({"label": "Optimality / parameter uncertainty", "state": "unknown",
                       "detail": "A small RMS, GP predictive SD, or late plateau alone cannot establish a global optimum or parameter confidence region."})
        recommendations.append("Use independent starts, a larger budget and held-out curve predictions for optimization robustness; compare to observation noise only if justified.")
        headline = "Optimization completed · optimum is not certified"
    return {"run": run.name, "kind": kind, "headline": headline,
            "metrics": metrics, "checks": checks, "recommendations": recommendations,
            "generated_utc": datetime.now().astimezone().isoformat()}


def render(run, result):
    run = Path(run).resolve()
    esc = lambda value: html.escape(str(value), quote=True)
    cards = "".join(f'<div class="card"><small>{esc(key.replace("_", " "))}</small><strong>{esc(json.dumps(value) if isinstance(value, (dict, list)) else value)}</strong></div>'
                    for key, value in result["metrics"].items() if not isinstance(value, (dict, list)))
    checks = "".join(f'<tr><td><span class="badge {esc(item["state"])}">{esc(item["state"])}</span></td>'
                     f'<td>{esc(item["label"])}</td><td>{esc(item["detail"])}</td></tr>' for item in result["checks"])
    figures = "".join(f'<figure><img loading="lazy" alt="{esc(title)}" src="plots/{esc(name)}">'
                      f'<figcaption>{esc(title)}</figcaption></figure>'
                      for name, title in PLOT_NAMES.items() if (run / "plots" / name).exists())
    recommendations = "".join(f"<li>{esc(value)}</li>" for value in result["recommendations"])
    parameters = (f'<h2>Conditional parameter intervals</h2><p>Not valid until chain + surrogate checks pass AND physics assumptions are independently justified.</p>'
                  f'<pre>{esc(json.dumps(result["metrics"].get("posterior_intervals"), indent=2))}</pre>'
                  f'<h2>Per-chain move fractions</h2><p>Fraction of retained consecutive draws that differ; inspect traces and ranks for sticking. Nonzero movement alone does not prove mixing.</p>'
                  f'<pre>{esc(json.dumps(result["metrics"].get("chain_move_fraction", {}), indent=2))}</pre>'
                  if result["kind"] == "mcmc" else
                  f'<h2>Best observed parameters</h2><pre>{esc(json.dumps(result["metrics"]["best_parameters"], indent=2))}</pre>')
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(result["run"])} | Dakota audit</title><style>
:root{{font:16px/1.55 system-ui,sans-serif;color:#1e293b;background:#f4f7fa}}*{{box-sizing:border-box}}
body{{max-width:1200px;margin:auto;padding:2rem}}h1,h2{{letter-spacing:-.035em;color:#17253a}}h1{{font-size:2.2rem}}
header,.panel{{background:white;border-radius:14px;padding:1.5rem 2rem;box-shadow:0 2px 14px #16213514;margin-bottom:1.5rem}}
.sub{{color:#58677e}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:.9rem;margin:1.5rem 0}}
.card{{background:#ecf3fa;padding:1rem;border-radius:10px;overflow-wrap:anywhere}}small,strong{{display:block}}small{{color:#5f7188}}strong{{font-size:1.2rem}}
table{{width:100%;border-collapse:collapse}}td{{padding:.8rem;border-bottom:1px solid #e3e9ee;vertical-align:top}}
.badge{{display:inline-block;border-radius:20px;padding:.15rem .65rem;text-transform:uppercase;font-size:.7rem;font-weight:bold}}
.pass{{background:#d9f5e4;color:#166534}}.warn{{background:#fff0ce;color:#92400e}}.unknown,.missing{{background:#e9edf5;color:#455775}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:1rem}}
figure{{margin:0;background:white;border-radius:12px;padding:1rem;box-shadow:0 2px 14px #16213514}}img{{max-width:100%;height:auto;display:block}}
figcaption{{color:#536279;font-size:.9rem;margin:.5rem}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f9;padding:1rem;border-radius:8px}}
@media print{{body{{background:white}}header,.panel,figure{{box-shadow:none;break-inside:avoid}}}}
</style></head><body><header><p class="sub">Dakota run diagnostic • {esc(result["kind"])}</p>
<h1>{esc(result["headline"])}</h1><p>{esc(result["run"])}</p><p class="sub">Generated {esc(result["generated_utc"])}</p></header>
<section class="panel"><h2>At a glance</h2><div class="metrics">{cards}</div></section>
<section class="panel"><h2>Checks and interpretation</h2><table><thead><tr><th>State</th><th>Check</th><th>Meaning</th></tr></thead><tbody>{checks}</tbody></table>
<h2>What to do next</h2><ul>{recommendations}</ul>{parameters}</section><h2>Figures</h2><div class="grid">{figures}</div>
<footer><p>Exploratory numerics are not a physical-model validation. KS p-values require independent datasets; MCMC draws are correlated.</p></footer></body></html>'''
    (run / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "report.html").write_text(document)
    return run / "report.html"


def main():
    parser = argparse.ArgumentParser(description="Build a self-contained run audit from saved outputs")
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(render(args.run, audit(args.run)))


if __name__ == "__main__":
    main()
