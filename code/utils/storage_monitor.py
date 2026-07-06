"""Storage monitor — checks disk usage and enforces safety thresholds.

Call check_storage() at the start of every SLURM job. If free space
drops below the critical threshold, the job will exit with an error.
"""

import os
import sys
import json
import shutil
from datetime import datetime


# Default paths
PROJECT_ROOT = "/proj/nobackup/hpc2n2025-278/academic-works/papers/plant-science-metascience"
STORAGE_MOUNT = "/proj/nobackup/hpc2n2025-278"

WARNING_GB = 100   # Log warning when free space < this
CRITICAL_GB = 50   # Hard stop when free space < this


def get_free_space_gb(path: str = STORAGE_MOUNT) -> float:
    """Return free space in GB for the filesystem containing path."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def get_project_size_gb(path: str = PROJECT_ROOT) -> float:
    """Return total size of project directory in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 ** 3)


def check_storage(
    warning_gb: float = WARNING_GB,
    critical_gb: float = CRITICAL_GB,
    log: bool = True,
) -> dict:
    """
    Check storage and enforce thresholds.

    Returns dict with free_gb, project_gb, status ('ok', 'warning', 'critical').
    Exits with code 1 if critical threshold breached.
    """
    free_gb = get_free_space_gb()
    project_gb = get_project_size_gb()

    if free_gb < critical_gb:
        status = "critical"
    elif free_gb < warning_gb:
        status = "warning"
    else:
        status = "ok"

    result = {
        "timestamp": datetime.now().isoformat(),
        "free_gb": round(free_gb, 1),
        "project_gb": round(project_gb, 1),
        "status": status,
        "hostname": os.uname().nodename,
    }

    if log:
        log_path = os.path.join(PROJECT_ROOT, "data", "metadata", "storage_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Print status
    icon = {"ok": "OK", "warning": "WARNING", "critical": "CRITICAL"}[status]
    print(f"  [{icon}] Storage: {free_gb:.1f} GB free, project uses {project_gb:.1f} GB")

    if status == "critical":
        print(f"  CRITICAL: Free space ({free_gb:.1f} GB) below threshold ({critical_gb} GB)!")
        print("  Stopping job to prevent storage overflow.")
        sys.exit(1)
    elif status == "warning":
        print(f"  WARNING: Free space ({free_gb:.1f} GB) approaching threshold ({critical_gb} GB).")

    return result


if __name__ == "__main__":
    result = check_storage()
    print(json.dumps(result, indent=2))
