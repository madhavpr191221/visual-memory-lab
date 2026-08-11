"""Replaceable storage boundary for visual-memory applications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from visual_memory_lab.memory import MemoryIndex


@dataclass(frozen=True)
class StoredMemory:
    rank: int
    score: float
    record: dict[str, object]


class MemoryStore(Protocol):
    """Minimum retrieval contract required by the office-memory UI."""

    @property
    def count(self) -> int: ...

    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def search(self, vector: np.ndarray, *, top_k: int) -> list[StoredMemory]: ...

    def get(self, observation_id: str) -> dict[str, object]: ...

    def embedding(self, observation_id: str) -> np.ndarray: ...

    def image_path(self, observation_id: str) -> Path: ...

    def records(self) -> list[dict[str, object]]: ...


class NumpyMemoryStore:
    """Exact in-memory NumPy implementation backed by a Phase 2/3 index."""

    def __init__(self, index: MemoryIndex) -> None:
        self.index = index
        self._record_by_id = {
            str(record["observation_id"]): record for record in index.records
        }

    @classmethod
    def load(cls, root: Path, *, verify_source: bool = False) -> NumpyMemoryStore:
        return cls(MemoryIndex.load(root, verify_source=verify_source))

    @property
    def count(self) -> int:
        return len(self.index.records)

    @property
    def model_id(self) -> str:
        return self.index.model_id

    @property
    def model_revision(self) -> str:
        return self.index.model_revision

    def search(self, vector: np.ndarray, *, top_k: int) -> list[StoredMemory]:
        return [
            StoredMemory(
                rank=result.rank,
                score=result.score,
                record=result.observation,
            )
            for result in self.index.search(vector, top_k=top_k)
        ]

    def get(self, observation_id: str) -> dict[str, object]:
        try:
            return self._record_by_id[observation_id]
        except KeyError as error:
            raise KeyError(f"unknown observation ID: {observation_id}") from error

    def embedding(self, observation_id: str) -> np.ndarray:
        return self.index.observation_embedding(observation_id)

    def image_path(self, observation_id: str) -> Path:
        record = self.get(observation_id)
        path = (self.index.image_root / str(record["image_path"])).resolve()
        try:
            path.relative_to(self.index.image_root.resolve())
        except ValueError as error:
            raise ValueError("observation image resolves outside its image root") from error
        if not path.is_file():
            raise FileNotFoundError(f"observation image is missing: {observation_id}")
        return path

    def records(self) -> list[dict[str, object]]:
        return self.index.records
