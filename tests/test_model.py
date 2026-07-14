import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from src.features import N_FAMILIES
from src.model import EEGSetEncoder, MultimodalClassifier


class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(20, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ModelTests(unittest.TestCase):
    def test_eeg_encoder_handles_missing_subjects(self):
        batch, subjects, channels = 3, 4, 5
        encoder = EEGSetEncoder(channels, channel_dim=8, eeg_dim=12, dropout=0.0)
        eeg = torch.randn(batch, subjects, channels * N_FAMILIES)
        mask = torch.tensor(
            [
                [True, True, True, True],
                [True, False, True, False],
                [False, True, False, False],
            ]
        )
        embedding, weights = encoder(eeg, mask)
        self.assertEqual(tuple(embedding.shape), (batch, 12))
        self.assertEqual(tuple(weights.shape), (batch, subjects, channels))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(batch, subjects))

    @patch("src.model.AutoModel.from_pretrained", return_value=DummyTextEncoder())
    def test_gated_fusion_forward(self, _):
        model = MultimodalClassifier(
            model_name="dummy",
            fusion="gated",
            text_mode="finetune",
            n_channels=5,
            text_dim=10,
            channel_dim=8,
            eeg_dim=6,
            dropout=0.0,
        )
        logits, weights = model(
            input_ids=torch.randint(0, 20, (4, 7)),
            attention_mask=torch.ones(4, 7, dtype=torch.long),
            eeg=torch.randn(4, 3, 5 * N_FAMILIES),
            subject_mask=torch.ones(4, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(weights.shape), (4, 3, 5))
        self.assertLess(model.gate_mean(), 0.2)


if __name__ == "__main__":
    unittest.main()
