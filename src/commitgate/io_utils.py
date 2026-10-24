"""Run directory creation, JSONL event logging, summary writing."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from commitgate.config import Config, dump_config


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def make_run_dir(root: str | Path, exp_name: str, method: str, stream: str, seed: int, cfg: Config) -> Path:
    """results/<exp_name>/<method>/<stream>/seed<k>/ with the resolved config written in."""
    run_dir = Path(root) / exp_name / method / stream / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    return run_dir


class RunLogger:
    """Appends one JSON line per event to log.jsonl."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "log.jsonl"

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        summary = {**summary, "git_commit": git_commit_hash()}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
