#!/bin/bash
#SBATCH --job-name=dakota-controller
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Example from repository root: sbatch scripts/submit_allocation.sh bo
# Match --ntasks to CONCURRENCY and --cpus-per-task to CPUS_PER_EVALUATION.
# Request enough memory for all simultaneously active simulations.
set -euo pipefail
cd "$(dirname "$0")/.."
python app/run.py "${1:-bo}"   # activate your compute-node Python environment first