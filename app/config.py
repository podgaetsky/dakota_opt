"""Edit this file for your problem; run.py regenerates both Dakota input files."""

PARAMETERS = {
    "p1": {"initial": 1.0, "lower": 0.1, "upper": 3.0},
    "p2": {"initial": 0.5, "lower": 0.01, "upper": 2.0},
    "p3": {"initial": 10.0, "lower": 1.0, "upper": 20.0},
    "p4": {"initial": 2.5, "lower": 0.1, "upper": 10.0},
}

# "slurm": one queued sbatch per evaluation; "allocation": srun steps in
# one already allocated Slurm job; "local": workstation demonstration.
BACKEND = "local"
CONCURRENCY = 8
CPUS_PER_EVALUATION = 1
MEMORY = "2G"
TIME_LIMIT = "00:05:00"
PARTITION = ""  # optional
JOB_TIMEOUT_SECONDS = 600
POLL_SECONDS = 2
FAILURE_PENALTY = 1e6

OPT_EVALUATIONS = 80
BO_INITIAL_SAMPLES = 24
BO_BATCHES = 10
BO_BATCH_SIZE = 8  # at most CONCURRENCY; total approx initial + batches * batch size
SEED = 1729