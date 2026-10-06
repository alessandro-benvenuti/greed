import json
import os
import platform
import socket
import subprocess
from pathlib import Path


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


gurobi_home = Path(os.environ["GUROBI_HOME"])
record = {
    "hostname": socket.gethostname(),
    "gcc": command("gcc", "--version").splitlines()[0],
    "cmake": command("cmake", "--version").splitlines()[0],
    "gurobi": command("gurobi_cl", "--version").splitlines()[0],
    "gurobi_home": str(gurobi_home),
    "gurobi_libraries": sorted(str(path) for path in (gurobi_home / "lib").glob("libgurobi*")),
    "python": platform.python_version(),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "loaded_modules": os.getenv("LOADEDMODULES", "").split(":"),
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
}
print(json.dumps(record, indent=2, sort_keys=True))
