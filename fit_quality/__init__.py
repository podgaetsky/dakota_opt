"""Optimizer-independent, assumption-explicit scientific curve diagnostics.

No Dakota imports or run-directory conventions. Statistical checks are conditional
on a supplied noise model; absence of evidence is represented as UNKNOWN.
"""

from .core import analyze_curve, read_curve, read_noise

__all__ = ["analyze_curve", "read_curve", "read_noise"]
