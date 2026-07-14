"""Small offline end-to-end check for the EEG-only training and report path."""

import os
import tempfile

import numpy as np

from src.config import TrainConfig
from src.data import MultimodalData
from src.experiment import run_experiments
from src.features import N_FAMILIES, feature_names
from src.reporting import build_summary


def main():
    rng = np.random.default_rng(7)
    n_sentences, n_subjects, n_channels = 60, 3, 5
    labels = np.tile(np.arange(3), n_sentences // 3)
    eeg = rng.normal(
        size=(n_sentences, n_subjects, n_channels * N_FAMILIES)
    ).astype(np.float32)
    eeg[:, :, 0] += labels[:, None] * 0.25
    data = MultimodalData(
        sentence_ids=np.arange(n_sentences),
        sentences=[f"synthetic sentence {index}" for index in range(n_sentences)],
        labels=labels,
        eeg=eeg,
        subject_mask=np.ones((n_sentences, n_subjects), dtype=bool),
        subjects=[f"S{index}" for index in range(n_subjects)],
        feature_names=feature_names(n_channels),
        n_channels=n_channels,
    )
    config = TrainConfig(
        n_folds=2,
        val_size=0.2,
        epochs=1,
        patience=1,
        batch_size=16,
        channel_dim=8,
        eeg_dim=12,
        num_workers=0,
    )
    with tempfile.TemporaryDirectory() as run_dir:
        run_experiments(data, ["eeg_only"], [42], config, run_dir)
        summary = build_summary(run_dir, bootstrap_samples=50)
        assert len(summary) == 1
        assert os.path.exists(os.path.join(run_dir, "tables", "summary.csv"))
        assert os.path.exists(os.path.join(run_dir, "plots", "scores.png"))
    print("offline pipeline smoke test passed")


if __name__ == "__main__":
    main()
