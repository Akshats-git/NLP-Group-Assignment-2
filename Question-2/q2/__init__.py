"""Question 2: 5-class Amazon review rating with a BiGRU baseline and fine-tuned RoBERTa.

    data          loads the reviews and puts each title back in front of its body
    classic       BiGRU pipeline: clean, lowercase, spaCy tokens, Vocab, GloVe matrix
    roberta_data  RoBERTa pipeline: byte-level BPE, truncation, dynamic padding
    batching      length-grouped batches for faster training
    models        RobertaClassifier (mean or first-token pooling)
    metrics       accuracy, macro / per-class F1, MAE, error distance
    train_roberta fine-tuning loop, also runnable as `python -m q2.train_roberta`
"""

import sys
from pathlib import Path

# make `common` importable for `python -m q2...` runs from Question-2/ without installing the repo
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
