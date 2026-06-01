"""
Multi-task DataLoader for joint training (understanding + generation + editing).

Wraps multiple independent DataLoaders (each with its own dataset, sampler, collate_fn)
and yields batches interleaved by task weights. Compatible with StepScheduler.
"""

import itertools
import logging
from typing import Dict, Iterator, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MultiTaskDataLoader:
    """Interleaved multi-task DataLoader that yields batches from task-specific DataLoaders
    according to configured task weights.

    Each task has its own DataLoader (dataset + sampler + collate_fn). This wrapper
    presents a unified iteration interface so StepScheduler and the training loop
    work without modification.

    Args:
        dataloaders: Mapping from task name to its DataLoader.
        task_weights: Mapping from task name to sampling weight (will be normalized).
        seed: Random seed for task selection reproducibility.
    """

    def __init__(
        self,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        task_weights: Dict[str, float],
        seed: int = 42,
    ):
        self.dataloaders = dataloaders
        self.task_names = list(dataloaders.keys())

        # Normalize weights to probabilities (only for tasks that have dataloaders)
        total = sum(task_weights.get(name, 0.0) for name in self.task_names)
        if total <= 0:
            raise ValueError(f"task_weights sum must be > 0, got weights for {self.task_names}")
        self.task_probs = np.array(
            [task_weights.get(name, 0.0) / total for name in self.task_names],
            dtype=np.float64,
        )

        self.seed = seed
        self.epoch = 0

        logger.info(
            f"MultiTaskDataLoader: tasks={self.task_names}, "
            f"probs={[f'{p:.3f}' for p in self.task_probs]}, "
            f"lengths={[len(dl) for dl in self.dataloaders.values()]}"
        )

    def __len__(self) -> int:
        return sum(len(dl) for dl in self.dataloaders.values())

    def __iter__(self) -> Iterator:
        rng = np.random.default_rng(self.seed + self.epoch)

        # Create infinite iterators for each task dataloader
        iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        exhausted = set()

        # Pre-compute total batches to yield (sum of all dataloader lengths)
        total_batches = sum(len(dl) for dl in self.dataloaders.values())
        yielded = 0

        while yielded < total_batches and len(exhausted) < len(self.task_names):
            # Mask out exhausted tasks and re-normalize
            active_mask = np.array([name not in exhausted for name in self.task_names])
            if not active_mask.any():
                break
            probs = self.task_probs * active_mask
            prob_sum = probs.sum()
            if prob_sum <= 0:
                break
            probs = probs / prob_sum

            task_idx = rng.choice(len(self.task_names), p=probs)
            task_name = self.task_names[task_idx]

            try:
                batch = next(iters[task_name])
            except StopIteration:
                exhausted.add(task_name)
                continue

            yield batch
            yielded += 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        for name, dl in self.dataloaders.items():
            sampler = getattr(dl, "sampler", None) or getattr(dl, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    @property
    def dataset(self):
        """Return the first dataset for compatibility with code that accesses dataloader.dataset."""
        first_name = self.task_names[0]
        return self.dataloaders[first_name].dataset

    @property
    def sampler(self):
        """Return the first sampler for compatibility."""
        first_name = self.task_names[0]
        dl = self.dataloaders[first_name]
        return getattr(dl, "sampler", None) or getattr(dl, "batch_sampler", None)
