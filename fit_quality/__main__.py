"""Generic file-based scientific model audit; no Dakota executable needed."""

import argparse
import hashlib
import html
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .core import analyze_curve, read_curve, read_noise
from .figures import plot_identifiability, plot_predictive, plot_residuals
from .inference import identifiability, posterior_predictive
from .residuals import diagnose


def _dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _optional_json(path):
    return json.loads(path.read_text()) if path else None


def _parameters(path, definitions, count):
    values = _optional_json(path)
    if values is not None and "parameters" in values:
        values = values["parameters"]
    if values is not None and not isinstance(values, dict):
        raise ValueError("Parameter values must be a JSON object or contain 'parameters' mapping")
    specs = _optional_json(definitions)
    if specs is not None and "parameters" in specs:
        specs = specs["parameters"]
    if specs is not None and not isinstance(specs, dict):
        raise ValueError("Parameter definitions must be a JSON mapping")
    if values is not None and specs is not None and not set(values).issubset(specs):
        raise ValueError("Missing parameter definitions for supplied values")
    # A parameter file does not establish how many were fitted on these points.
    return values, specs, count


def _validation(args, training):
    entries = []
    if args.validation_reference or args.validation_prediction:
        if not args.validation_reference or not args.validation_prediction:
            raise ValueError("Hold-out evaluation requires both validation curves, predicted without fitting to held-out observations")
        entries.append({"reference": str(args.validation_reference), "prediction": str(args.validation_prediction),
                        "name": "held-out curve"})
    if args.validation_manifest:
        manifest = json.loads(args.validation_manifest.read_text())
        if not isinstance(manifest, list):
            raise ValueError("Validation manifest must be a JSON list of independent/blocked curve pairs")
        for entry in manifest:
            item = dict(entry)
            for key in ("reference", "prediction"):
                if key not in item:
                    raise ValueError("Every validation entry needs reference and prediction")
                path = Path(item[key])
                item[key] = str(path if path.is_absolute() else args.validation_manifest.parent / path)
            entries.append(item)
    if not entries:
        return {"status": "UNKNOWN", "note": "No independent predictive validation was available."}
    curves = []
    for entry in entries:
        held, _ = analyze_curve(read_curve(entry["reference"]), read_curve(entry["prediction"]))
        curves.append({"name": entry.get("name", "curve"), "kind": entry.get("kind", "independent curve"),
                       "n": held["n"], "validation_rms": held["rms"], "validation_mae": held["mae"],
                       "generalization_gap_rms": held["rms"] - training["rms"]})
    total = sum(item["n"] for item in curves)
    return {"status": "CONDITIONAL", "training_rms": training["rms"], "training_mae": training["mae"],
            "validation_rms": float(np.sqrt(sum(item["n"] * item["validation_rms"] ** 2 for item in curves) / total)),
            "validation_mae": float(sum(item["n"] * item["validation_mae"] for item in curves) / total),
            "curves": curves, "note": "User must confirm entire held-out curves or contiguous x blocks were excluded from fitting, model selection and preprocessing; no random point-wise CV."}


def render_html(report, output):
    esc = lambda value: html.escape(str(value), quote=True)
    table = lambda rows: "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
    statuses = report["scientific_status"]
    summary = table(statuses.items())
    assumptions = table(report["assumption_audit"].items())
    sections = ["Data and assumptions", "Optimization", "Goodness of fit", "Residual diagnostics", "Independence/correlation",
                "Parameter identifiability", "Bayesian inference", "Posterior predictive checks", "Surrogate validation",
                "Independent validation", "Model comparison", "Reproducibility/provenance"]
    content = [report["data_and_assumptions"], report["optimization"], report["fit_metrics"], report["residual_diagnostics"],
               {"independence": report["residual_diagnostics"]["independence_note"]}, report["identifiability"],
               report["bayesian_inference"], report["posterior_predictive"], report["surrogate_validation"],
               report["validation"], report["model_comparison"], report["provenance"]]
    body = "".join(f'<section><h2>{i}. {esc(name)}</h2><pre>{esc(json.dumps(info, indent=2))}</pre></section>'
                   for i, (name, info) in enumerate(zip(sections, content), 1))
    figures = "".join(f'<figure><img alt="{esc(name)}" src="{esc(name)}"><figcaption>{esc(name)}</figcaption></figure>'
                      for name in ("residual_diagnostics.png", "identifiability.png", "posterior_predictive_diagnostics.png")
                      if (output / name).is_file())
    (output / "report.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scientific curve audit</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1050px;margin:auto;padding:2rem;background:#f2f6fb;color:#17243a}}
section,header,figure{{background:white;padding:1.3rem 1.8rem;border-radius:12px;margin:1rem 0;box-shadow:0 2px 12px #14203612}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace}}
th,td{{text-align:left;vertical-align:top;padding:.45rem;border-bottom:1px solid #dde4ec}}th{{min-width:220px}}
img{{max-width:100%}}.note{{color:#9a3412}}h1,h2{{line-height:1.2}}@media print{{body{{background:white}}section,header,figure{{box-shadow:none}}}}
</style></head><body><header><h1>Scientific curve-fit audit</h1><p class="note">No single fit score. Numerical checks are conditional on declared assumptions; UNKNOWN is not PASS.</p><table>{summary}</table><h2>Scientific assumption audit</h2><table>{assumptions}</table></header>{body}{figures}</body></html>''')


def execute(args):
    ref, pred = read_curve(args.reference), read_curve(args.prediction)
    noise = read_noise(args.noise_model) if args.noise_model else None
    values, definitions, k = _parameters(args.parameters, args.parameter_definitions, args.parameter_count)
    weights = np.loadtxt(args.weights, delimiter=",", ndmin=1) if args.weights else None
    metrics, data = analyze_curve(ref, pred, noise=noise, parameter_count=k,
                                  weights=weights, chi2_test_justified=args.chi2_test_justified)
    residual = diagnose(data)
    validation = _validation(args, metrics)
    identifiable = {"status": "UNKNOWN", "note": "Provide a normalized-parameter Jacobian; data alone cannot determine parameter identifiability."}
    if args.jacobian:
        matrix = np.loadtxt(args.jacobian, delimiter=",", ndmin=2)
        identifiable = identifiability(matrix, noise=noise)
    predictive = {"status": "UNKNOWN", "note": "Provide posterior model curves (draws by ordinate); parameter samples alone cannot produce predictions without a simulator."}
    curves = None
    if args.posterior_curves:
        curves = np.loadtxt(args.posterior_curves, delimiter=",", ndmin=2)
        predictive, _, _ = posterior_predictive(ref, curves, noise=noise)
        predictive["status"] = "CONDITIONAL"
    bayes = {"status": "UNKNOWN", "note": "Posterior samples, sampler diagnostics and prior sensitivity need separate review."}
    if args.posterior_samples:
        samples = np.loadtxt(args.posterior_samples, delimiter=",", ndmin=2)
        if not np.isfinite(samples).all():
            raise ValueError("Nonfinite posterior samples")
        correlation = (np.corrcoef(samples, rowvar=False).tolist()
                       if samples.shape[1] > 1 and len(samples) >= 3 and np.all(np.std(samples, axis=0) > 0) else None)
        bayes = {"status": "CONDITIONAL", "count": len(samples), "parameter_correlation": correlation,
                 "note": "Sample correlations do not verify MCMC mixing or prior robustness; inspect chain-wise diagnostics separately."}
    status = {"OPTIMIZATION": "UNKNOWN", "GOODNESS OF FIT": "CONDITIONAL" if noise else "DESCRIPTIVE",
              "RESIDUAL INDEPENDENCE": residual["independence_status"], "IDENTIFIABILITY": identifiable["status"],
              "MCMC NUMERICS": "UNKNOWN", "SURROGATE VALIDATION": "UNKNOWN", "PREDICTIVE VALIDATION": validation["status"]}
    boundary = "UNKNOWN"
    if values and definitions and all(isinstance(definitions.get(name), dict) and {"lower", "upper"}.issubset(definitions[name]) for name in values):
        edge = [name for name, value in values.items() if min(
            (value - definitions[name]["lower"]) / (definitions[name]["upper"] - definitions[name]["lower"]),
            (definitions[name]["upper"] - value) / (definitions[name]["upper"] - definitions[name]["lower"])) < .02]
        boundary = "YES: " + ", ".join(edge) if edge else "NO (within 2% of bounds)"
    assumptions = {"Gaussian noise": "UNKNOWN" if noise else "NOT APPLICABLE",
                   "Independent residuals": residual["independence_status"], "Constant variance": "UNKNOWN",
                   "Covariance supplied": "YES" if noise and "covariance" in noise else "NO", "Model discrepancy assessed": "NO",
                   "Independent validation data": "CONDITIONAL" if validation["status"] == "CONDITIONAL" else "NO",
                   "Parameter identifiability": identifiable["status"], "MCMC convergence": "UNKNOWN",
                   "Surrogate validation": "UNKNOWN", "Multiple optimizer starts": "UNKNOWN", "Boundary optimum": boundary}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"scientific_status": status, "assumption_audit": assumptions,
              "data_and_assumptions": {"reference": str(args.reference.resolve()), "prediction": str(args.prediction.resolve()),
                "noise": noise, "parameters": values, "parameter_definitions": definitions,
                "chi2_p_value_eligible_user_attested": args.chi2_test_justified,
                "note": "p-value requires established Gaussian model, known covariance, appropriate fitted dof and selection protocol; checkbox is user attestation, not software validation."},
              "optimization": {"status": "UNKNOWN", "note": "No independent optimizer starts or convergence histories supplied; low RMS does not prove optimality."},
              "fit_metrics": metrics, "residual_diagnostics": residual, "identifiability": identifiable,
              "bayesian_inference": bayes, "posterior_predictive": predictive,
              "surrogate_validation": {"status": "UNKNOWN", "note": "No exact surrogate hold-out predictions supplied."},
              "validation": validation, "model_comparison": {"status": "NOT APPLICABLE", "note": "One candidate only; compare compatible data and likelihoods across models."},
              "provenance": {"mode": "file-only, Dakota absent", "command_line": sys.argv, "python": sys.version,
                              "platform": platform.platform(),
                              "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                               (args.reference, args.prediction, args.parameters, args.parameter_definitions,
                                                args.noise_model, args.jacobian, args.posterior_curves,
                                                args.posterior_samples, args.validation_reference, args.validation_prediction,
                                                args.validation_manifest,
                                                args.weights) if path is not None}}}
    _dump(output / "report.json", report)
    plot_residuals(data, output)
    if args.jacobian:
        plot_identifiability(identifiable, output)
    if curves is not None:
        plot_predictive(ref, curves, predictive, output)
    render_html(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description="Optimizer-independent scientific curve audit; requires no Dakota installation")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, help="JSON object of fitted parameter values (NOT fitted count)")
    parser.add_argument("--parameter-definitions", type=Path, help="JSON name -> {lower,upper,...}")
    parser.add_argument("--parameter-count", type=int, help="effective number fitted on these data; required for dof and AIC/BIC")
    parser.add_argument("--noise-model", type=Path, help="JSON with exactly one of sigma, sigma_i, covariance")
    parser.add_argument("--jacobian", type=Path, help="N by K CSV, derivatives with respect to normalized parameters")
    parser.add_argument("--posterior-samples", type=Path, help="draw by parameter CSV without header")
    parser.add_argument("--posterior-curves", type=Path, help="draw by ordinate CSV without header, exact curves")
    parser.add_argument("--validation-reference", type=Path)
    parser.add_argument("--validation-prediction", type=Path)
    parser.add_argument("--validation-manifest", type=Path, help="JSON list of independently predicted whole curves or blocked x regions")
    parser.add_argument("--weights", type=Path, help="N positive/nonnegative curve integration weights CSV")
    parser.add_argument("--chi2-test-justified", action="store_true", help="attest formal chi² assumptions; otherwise no p-value")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args)
    print(f"Fit quality: {args.output.resolve() / 'report.html'}; RMSE={result['fit_metrics']['rmse']:.5g}; independent validation={result['validation']['status']}")


if __name__ == "__main__":
    main()
