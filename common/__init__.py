"""Shared helpers for all four questions. See common/README.md.

The everyday pieces are importable from the package directly:

    from common import set_seed, get_device, Vocab, pad_collate, History, EarlyStopping

Plotting lives in common.plotting and is not imported here, so importing common
does not pull in matplotlib.
"""

from .decoding import Decoding, generate, next_token
from .device import amp_dtype, autocast, get_device, grad_scaler, to_device
from .files import load_json, load_jsonl, markdown_table, save_json, save_jsonl, save_run_info
from .metrics import bucket_indices, bucket_labels, format_mean_std, masked_accuracy, mean_std, perplexity
from .paths import REPO_ROOT, checkpoints_dir, data_dir, figures_dir, results_dir
from .seed import DEFAULT_SEEDS, dataloader_kwargs, make_generator, set_seed
from .sequences import causal_mask, lengths_to_mask, pad_collate, pad_sequences, padding_mask, teacher_forcing_pair
from .text import normalize, strip_html_and_urls
from .training import (
    AverageMeter,
    EarlyStopping,
    History,
    Timer,
    count_parameters,
    format_count,
    load_checkpoint,
    save_checkpoint,
)
from .vocab import EOS, PAD, SOS, UNK, Vocab

__all__ = [
    "AverageMeter", "DEFAULT_SEEDS", "Decoding", "EOS", "EarlyStopping", "History", "PAD", "REPO_ROOT",
    "SOS", "Timer", "UNK", "Vocab", "amp_dtype", "autocast", "bucket_indices", "bucket_labels",
    "causal_mask", "checkpoints_dir", "count_parameters", "data_dir", "dataloader_kwargs", "figures_dir",
    "format_count", "format_mean_std", "generate", "get_device", "grad_scaler", "lengths_to_mask",
    "load_checkpoint", "load_json", "load_jsonl", "make_generator", "markdown_table", "masked_accuracy", "mean_std",
    "next_token", "normalize", "pad_collate", "pad_sequences", "padding_mask", "perplexity", "results_dir",
    "save_checkpoint", "save_json", "save_jsonl", "save_run_info", "set_seed", "strip_html_and_urls",
    "teacher_forcing_pair", "to_device",
]
