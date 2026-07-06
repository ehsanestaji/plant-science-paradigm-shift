"""Checkpoint utilities for resumable data collection and processing.

Every long-running pipeline should call checkpoint_save() periodically
and checkpoint_load() at startup to resume from where it left off.
"""

import os
import json
from datetime import datetime


CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "metadata"
)


def checkpoint_save(name: str, state: dict) -> str:
    """Save a checkpoint for a named pipeline stage."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{name}.json")
    state["_checkpoint_time"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return path


def checkpoint_load(name: str) -> dict | None:
    """Load a checkpoint. Returns None if no checkpoint exists."""
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def checkpoint_clear(name: str) -> bool:
    """Remove a checkpoint after successful completion."""
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{name}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
