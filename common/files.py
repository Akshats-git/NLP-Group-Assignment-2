"""Saving results in formats the report and the other teammates can read back."""

import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .paths import REPO_ROOT


def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path, torch.device, torch.dtype)):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(obj, path: str | Path, indent: int = 2) -> Path:
    """json.dump that also takes numpy / torch scalars and arrays, Paths and dataclasses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_to_jsonable)  # keep Hindi readable
        f.write("\n")
    return path


def load_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows, path: str | Path) -> Path:
    """One JSON object per line. Good for per-example predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_to_jsonable) + "\n")
    return path


def load_jsonl(path: str | Path) -> list:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def markdown_table(rows, headers=None, floatfmt: str = ".2f") -> str:
    """Render rows as a markdown table, ready to paste into the report.

    rows is a list of dicts (headers default to the first row's keys) or a list of lists.
    """
    rows = list(rows)
    if not rows:
        return ""
    if isinstance(rows[0], dict):
        headers = headers or list(rows[0].keys())
        rows = [[row.get(h, "") for h in headers] for row in rows]
    elif headers is None:
        raise ValueError("headers are required when rows are lists")

    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return format(value, floatfmt)
        return str(value)

    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict:
    """Python, library and hardware versions. The README and report need these for reproducibility."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for name in ("transformers", "datasets", "tokenizers", "sentencepiece", "sacrebleu", "rouge_score"):
        module = sys.modules.get(name)
        if module is not None:
            info[name] = getattr(module, "__version__", "unknown")
    return info


def save_run_info(out_dir: str | Path, config, seed: int | None = None, **extra) -> Path:
    """Write run_info.json next to a run's results: the config, the seed and the environment.

    config can be a dict or a dataclass. Call it once at the start of every training run.
    """
    payload = {"config": config, "seed": seed, "environment": environment_info(), **extra}
    return save_json(payload, Path(out_dir) / "run_info.json")
