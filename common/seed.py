"""Seeding, so a run gives the same numbers on every teammate's machine."""

import os
import random

import numpy as np
import torch

# Q2 needs mean and std over at least two seeds. Use these unless there is a reason not to.
DEFAULT_SEEDS = (42, 43)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU and every GPU).

    deterministic=True also forces deterministic cuDNN kernels. That makes GPU runs
    bit-for-bit repeatable but slower, so leave it off for long training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # only reaches subprocesses, e.g. DataLoader workers
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """worker_init_fn for DataLoader, so shuffling and augmentation in workers is seeded too."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def dataloader_kwargs(seed: int) -> dict:
    """Extra DataLoader arguments for a seeded shuffle: DataLoader(ds, shuffle=True, **dataloader_kwargs(seed))."""
    return {"worker_init_fn": seed_worker, "generator": make_generator(seed)}
