import json
import os
import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


gurobi_home = Path(os.environ["GUROBI_HOME"])
record = {
    "hostname": socket.gethostname(),
    "gcc": command("gcc", "--version").splitlines()[0],
    "cmake": command("cmake", "--version").splitlines()[0],
    "gurobi": command("gurobi_cl", "--version").splitlines()[0],
    "gurobi_home": str(gurobi_home),
    "gurobi_libraries": sorted(str(path) for path in (gurobi_home / "lib").glob("libgurobi*")),
    "python": platform.python_version(),
    "python_executable": command("python", "-c", "import sys; print(sys.executable)"),
    "python_packages": {
        name: package_version(name)
        for name in ("numpy", "pybind11", "pytest", "PyYAML", "scipy", "torch", "torch-geometric")
    },
    "git_commit": command("git", "rev-parse", "HEAD"),
    "loaded_modules": os.getenv("LOADEDMODULES", "").split(":"),
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
}
print(json.dumps(record, indent=2, sort_keys=True))
