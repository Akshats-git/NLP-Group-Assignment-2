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
    """Copy results and checkpoints of runs with a summary.json found under `inputs`. Returns the run names copied.

    Results and checkpoints are checked separately: a fresh clone already has the committed
    results of finished runs but never their checkpoints, which the evaluation needs.
    """
    copied = []
    if not inputs.exists():
        return copied
    outputs = output_dir(2)
    for summary in sorted(inputs.glob(f"**/Question-2/outputs/results/{pattern}/summary.json")):
        run, source = summary.parent.name, summary.parents[2]
        missing = [kind for kind, marker in (("results", "summary.json"), ("checkpoints", "best.pt"))
                   if (source / kind / run).exists() and not (outputs / kind / run / marker).exists()]
        for kind in missing:
            shutil.copytree(source / kind / run, outputs / kind / run, dirs_exist_ok=True)
        if missing:
            copied.append(run)
    return copied
