import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--stage", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
record = {
    "stage": args.stage,
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    "cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
    "node_list": os.getenv("SLURM_JOB_NODELIST"),
}
path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.{os.getpid()}")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
tmp.replace(path)
