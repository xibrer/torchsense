"""Base data module for TorchSense with configurable split strategies.

Follows anomalib's pattern: subclasses implement :meth:`_setup` to create a
full dataset, and the base class handles train/val/test splitting according
to :class:`ValSplitMode` and :class:`TestSplitMode`.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from torchsense.data.datasets.base import TorchSenseDataset


class ValSplitMode(str, Enum):
    """How the validation set is created."""

    NONE = "none"
    """No validation set."""

    FROM_TRAIN = "from_train"
    """Randomly sample a fraction of the training set."""

    FROM_TEST = "from_test"
    """Randomly sample a fraction of the test set."""

    SAME_AS_TEST = "same_as_test"
    """Reuse the test set as validation set."""

    BY_SUBJECT = "by_subject"
    """Hold out specific subjects for validation."""


class TestSplitMode(str, Enum):
    """How the test set is created."""

    NONE = "none"
    """No test set (all data goes to train)."""

    BY_SUBJECT = "by_subject"
    """Hold out specific subjects for testing."""

    RANDOM = "random"
    """Randomly sample a fraction of the full dataset."""


class TorchSenseDataModule(pl.LightningDataModule):
    """Base class for all TorchSense data modules.

    Subclasses implement :meth:`_setup` which must populate
    ``self._full_dataset`` (a :class:`TorchSenseDataset`).  The base
    class then handles splitting into train / val / test according to
    the configured :class:`TestSplitMode` and :class:`ValSplitMode`.

    Args:
        data_root: Root directory containing the data files.
        batch_size: Batch size for all dataloaders.
        num_workers: DataLoader worker count.
        test_split_mode: How to create the test set.
        test_split_ratio: Fraction of data for test (``random`` mode).
        test_subjects: Subject IDs to hold out (``by_subject`` mode).
        val_split_mode: How to create the validation set.
        val_split_ratio: Fraction of data for validation.
        val_subjects: Subject IDs for validation (``by_subject`` mode).
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        test_split_mode: str | TestSplitMode = TestSplitMode.NONE,
        test_split_ratio: float = 0.15,
        test_subjects: list[int] | None = None,
        val_split_mode: str | ValSplitMode = ValSplitMode.FROM_TRAIN,
        val_split_ratio: float = 0.15,
        val_subjects: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_split_mode = TestSplitMode(test_split_mode)
        self.test_split_ratio = test_split_ratio
        self.test_subjects = test_subjects or []
        self.val_split_mode = ValSplitMode(val_split_mode)
        self.val_split_ratio = val_split_ratio
        self.val_subjects = val_subjects or []
        self.seed = seed

        self.train_dataset: TorchSenseDataset | None = None
        self.val_dataset: TorchSenseDataset | None = None
        self.test_dataset: TorchSenseDataset | None = None

        self._full_dataset: TorchSenseDataset | None = None

    @abstractmethod
    def _setup(self) -> None:
        """Load and preprocess data, populating ``self._full_dataset``.

        After this method returns, ``self._full_dataset`` must be a valid
        :class:`TorchSenseDataset` instance containing **all** samples.
        The base class will then split it into train/val/test.
        """

    # ------------------------------------------------------------------
    # Lightning setup
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        """Load data and create train/val/test datasets."""
        if self._full_dataset is None:
            self._setup()

        assert self._full_dataset is not None, "_setup() must set self._full_dataset"

        train_indices, val_indices, test_indices = self._compute_splits()

        if stage in ("fit", None):
            self.train_dataset = self._full_dataset.subset(train_indices)
            if val_indices is not None:
                self.val_dataset = self._full_dataset.subset(val_indices)

        if stage in ("test", None) and test_indices is not None:
            self.test_dataset = self._full_dataset.subset(test_indices)

        if stage in ("validate", None) and self.val_dataset is None and val_indices is not None:
            self.val_dataset = self._full_dataset.subset(val_indices)

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------
    def _compute_splits(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Compute train/val/test index arrays based on configured modes.

        Returns:
            ``(train_indices, val_indices, test_indices)`` where val and test
            may be ``None`` if their mode is ``NONE``.
        """
        n = len(self._full_dataset)
        rng = np.random.default_rng(self.seed)

        # --- Test split ---
        test_indices: np.ndarray | None = None
        train_val_indices: np.ndarray

        if self.test_split_mode == TestSplitMode.BY_SUBJECT:
            subject_ids = self._full_dataset.subject_ids
            if subject_ids is None:
                msg = "test_split_mode='by_subject' requires dataset to expose subject_ids"
                raise ValueError(msg)
            test_mask = np.isin(subject_ids, self.test_subjects)
            test_indices = np.where(test_mask)[0]
            train_val_indices = np.where(~test_mask)[0]

        elif self.test_split_mode == TestSplitMode.RANDOM:
            all_idx = rng.permutation(n)
            n_test = int(n * self.test_split_ratio)
            test_indices = all_idx[:n_test]
            train_val_indices = all_idx[n_test:]

        else:  # NONE
            train_val_indices = np.arange(n)

        # --- Val split ---
        val_indices: np.ndarray | None = None

        if self.val_split_mode == ValSplitMode.FROM_TRAIN:
            shuffled = rng.permutation(len(train_val_indices))
            n_val = int(len(train_val_indices) * self.val_split_ratio)
            val_indices = train_val_indices[shuffled[:n_val]]
            train_indices = train_val_indices[shuffled[n_val:]]

        elif self.val_split_mode == ValSplitMode.FROM_TEST:
            if test_indices is None or len(test_indices) == 0:
                msg = "val_split_mode='from_test' requires a non-empty test split"
                raise ValueError(msg)
            shuffled = rng.permutation(len(test_indices))
            n_val = int(len(test_indices) * self.val_split_ratio)
            val_indices = test_indices[shuffled[:n_val]]
            test_indices = test_indices[shuffled[n_val:]]
            train_indices = train_val_indices

        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            val_indices = test_indices
            train_indices = train_val_indices

        elif self.val_split_mode == ValSplitMode.BY_SUBJECT:
            subject_ids = self._full_dataset.subject_ids
            if subject_ids is None:
                msg = "val_split_mode='by_subject' requires dataset to expose subject_ids"
                raise ValueError(msg)
            val_mask = np.isin(subject_ids[train_val_indices], self.val_subjects)
            val_indices = train_val_indices[val_mask]
            train_indices = train_val_indices[~val_mask]

        else:  # NONE
            train_indices = train_val_indices

        return train_indices, val_indices, test_indices

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') first"
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') or setup('validate') first"
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Call setup('test') first"
        return self._make_dataloader(self.test_dataset, shuffle=False)

    def _make_dataloader(self, dataset: TorchSenseDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            **self._extra_dataloader_kwargs(),
        )

    def _extra_dataloader_kwargs(self) -> dict[str, Any]:
        """Override to add custom DataLoader kwargs (e.g. collate_fn)."""
        return {}
