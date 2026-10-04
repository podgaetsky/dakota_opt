"""Compatibility entry point for saved runs; edit linear_benchmark/model.py instead."""

import sys
from linear_benchmark.model import simulate_from_files


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: linear_file_model.py WORK_DIR")
    print(simulate_from_files(sys.argv[1]))
