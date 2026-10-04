#!/bin/bash
#SBATCH --job-name=dakota-eval
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
# The Python driver overrides CPU/memory/time via sbatch flags from config.py.
set -euo pipefail
"$1" "$2" "$3"