"""Focused compatibility tests for the CLIP encoder boundary."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from visual_memory_lab.encoder import ClipEncoder


def test_feature_tensor_accepts_current_transformers_model_output() -> None:
    expected = torch.tensor([[1.0, 2.0]])
    output = SimpleNamespace(pooler_output=expected)

    assert ClipEncoder._feature_tensor(output) is expected


def test_feature_tensor_accepts_older_transformers_tensor_output() -> None:
    expected = torch.tensor([[1.0, 2.0]])

    assert ClipEncoder._feature_tensor(expected) is expected
