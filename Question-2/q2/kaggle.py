"""Carrying finished runs from one Kaggle notebook version to the next.

Every Kaggle version starts from a fresh clone, so earlier results are gone. Attach the
earlier version's output as an input (Add Input -> Your Work -> Notebooks) and this copies
its finished runs back into Question-2/outputs/, where the training loops will skip them
and the evaluation can load their checkpoints.
"""

import shutil
from pathlib import Path

from common.paths import output_dir

KAGGLE_INPUT = Path("/kaggle/input")


def import_finished_runs(pattern: str = "*", inputs: Path = KAGGLE_INPUT) -> list[str]:
    """Copy results and checkpoints of runs with a summary.json found under `inputs`. Returns the run names copied."""
    copied = []
    if not inputs.exists():
        return copied
    outputs = output_dir(2)
    for summary in sorted(inputs.glob(f"**/Question-2/outputs/results/{pattern}/summary.json")):
        run, source = summary.parent.name, summary.parents[2]
        if (outputs / "results" / run / "summary.json").exists():
            continue
        for kind in ("results", "checkpoints"):
            if (source / kind / run).exists():
                shutil.copytree(source / kind / run, outputs / kind / run, dirs_exist_ok=True)
        copied.append(run)
    return copied
