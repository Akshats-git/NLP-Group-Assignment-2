"""One place that decides where each question keeps its data, results, figures and checkpoints.

Layout inside every question folder:

    Question-N/
        data/          downloaded or cached data       (gitignored, never in the ZIP)
        outputs/
            results/   metrics, predictions, run info  (committed)
            figures/   plots used in the report        (committed)
            checkpoints/ model weights                 (gitignored, except Question-4)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def question_dir(question: int) -> Path:
    if question not in (1, 2, 3, 4):
        raise ValueError(f"question must be 1-4, got {question!r}")
    return REPO_ROOT / f"Question-{question}"


def _ensure(path: Path, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(question: int, create: bool = True) -> Path:
    return _ensure(question_dir(question) / "data", create)


def output_dir(question: int, *parts: str, create: bool = True) -> Path:
    return _ensure(question_dir(question).joinpath("outputs", *parts), create)


def results_dir(question: int, create: bool = True) -> Path:
    return output_dir(question, "results", create=create)


def figures_dir(question: int, create: bool = True) -> Path:
    return output_dir(question, "figures", create=create)


def checkpoints_dir(question: int, create: bool = True) -> Path:
    return output_dir(question, "checkpoints", create=create)
