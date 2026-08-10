"""Persistent exact-search visual memory built over trajectory observations."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

INDEX_SCHEMA_VERSION = "1.0"


class Encoder(Protocol):
    model_id: str
    model_revision: str
    embedding_dim: int

    @property
    def processor_config(self) -> dict[str, object]: ...

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class IndexSummary:
    output: Path
    observation_count: int
    embedding_dim: int


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    observation: dict[str, object]
    image_path: Path
    nearby_actions: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "image_path": str(self.image_path),
            "observation": self.observation,
            "nearby_actions": list(self.nearby_actions),
        }


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_records(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid JSONL from {path}: {error}") from error
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected at least one observation object in {path}")
    return records


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_records(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _source_files(
    source: Path,
    records: Sequence[dict[str, object]],
) -> list[Path]:
    files = [source / "run.json", source / "observations.jsonl"]
    for record in records:
        image_path = record.get("image_path")
        if not isinstance(image_path, str):
            raise ValueError("every observation must contain a string image_path")
        files.append(source / image_path)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"source artifact is missing {missing[0]}")
    return files


def _corpus_fingerprint(source: Path, files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _validate_source(source: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    manifest = _read_json(source / "run.json")
    records = _read_records(source / "observations.jsonl")
    expected = manifest.get("observation_count")
    if expected != len(records):
        raise ValueError(
            f"run.json declares {expected} observations but JSONL contains {len(records)}"
        )
    ids = [record.get("observation_id") for record in records]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("observation IDs must be unique strings")
    files = _source_files(source, records)
    agent_view = manifest.get("agent_view")
    if not isinstance(agent_view, dict):
        raise ValueError("run.json is missing the agent_view contract")
    expected_size = (
        int(agent_view.get("image_width", -1)),
        int(agent_view.get("image_height", -1)),
    )
    for frame_path in files[2:]:
        try:
            with Image.open(frame_path) as image:
                image.load()
                if image.mode != "RGB" or image.size != expected_size:
                    raise ValueError(
                        f"expected an RGB frame of size {expected_size}, got "
                        f"{image.mode} {image.size} at {frame_path}"
                    )
        except OSError as error:
            raise ValueError(
                f"could not read trajectory frame {frame_path}: {error}"
            ) from error
    return manifest, records, _corpus_fingerprint(source, files)


def _validate_embeddings(
    embeddings: np.ndarray,
    *,
    observation_count: int,
    embedding_dim: int,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    expected_shape = (observation_count, embedding_dim)
    if values.shape != expected_shape:
        raise ValueError(
            f"expected embeddings with shape {expected_shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("embeddings must be L2-normalized")
    if observation_count > 1 and np.allclose(values, values[0], atol=1e-7):
        raise ValueError("all image embeddings are identical")
    return values


def build_index(
    *,
    source: Path,
    output: Path,
    encoder: Encoder,
    batch_size: int = 64,
) -> IndexSummary:
    """Encode a trajectory artifact and persist an atomic exact-search index."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    source = source.resolve()
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output path is not empty: {output}")

    _, records, fingerprint = _validate_source(source)
    image_paths = [source / str(record["image_path"]) for record in records]
    batches = [
        encoder.encode_images(image_paths[start : start + batch_size])
        for start in range(0, len(image_paths), batch_size)
    ]
    embeddings = _validate_embeddings(
        np.concatenate(batches, axis=0),
        observation_count=len(records),
        embedding_dim=encoder.embedding_dim,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        np.save(temporary / "embeddings.npy", embeddings, allow_pickle=False)
        _write_records(temporary / "records.jsonl", records)
        _write_json(
            temporary / "index.json",
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "model": {
                    "id": encoder.model_id,
                    "revision": encoder.model_revision,
                    "processor": encoder.processor_config,
                },
                "source": {
                    "path": str(source),
                    "fingerprint_sha256": fingerprint,
                },
                "embeddings": {
                    "path": "embeddings.npy",
                    "observation_count": len(records),
                    "dimension": encoder.embedding_dim,
                    "dtype": "float32",
                    "normalized": True,
                },
                "search": {
                    "kind": "flat",
                    "similarity": "cosine",
                },
            },
        )
        if output.exists():
            output.rmdir()
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return IndexSummary(output, len(records), encoder.embedding_dim)


class MemoryIndex:
    """A validated visual-memory artifact loaded into RAM for exact search."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: dict[str, object],
        records: list[dict[str, object]],
        embeddings: np.ndarray,
        source: Path,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.records = records
        self.embeddings = embeddings
        self.source = source
        self._record_by_id = {
            str(record["observation_id"]): index
            for index, record in enumerate(records)
        }
        self._record_by_step = {
            (str(record["episode_id"]), int(record["step"])): record
            for record in records
        }

    @classmethod
    def load(cls, root: Path, *, verify_source: bool = True) -> MemoryIndex:
        root = root.resolve()
        manifest = _read_json(root / "index.json")
        if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported visual-memory index schema")
        records = _read_records(root / "records.jsonl")

        embedding_info = manifest.get("embeddings")
        source_info = manifest.get("source")
        model_info = manifest.get("model")
        if (
            not isinstance(embedding_info, dict)
            or not isinstance(source_info, dict)
            or not isinstance(model_info, dict)
        ):
            raise ValueError("index manifest is missing model, embeddings, or source metadata")
        if not isinstance(model_info.get("id"), str) or not isinstance(
            model_info.get("revision"), str
        ):
            raise ValueError("index manifest has invalid model metadata")
        source_value = source_info.get("path")
        if not isinstance(source_value, str):
            raise ValueError("index manifest has an invalid source path")
        source = Path(source_value)
        embedding_path = embedding_info.get("path")
        if not isinstance(embedding_path, str):
            raise ValueError("index manifest has an invalid embedding path")
        try:
            embeddings = np.load(root / embedding_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"could not load index embeddings: {error}") from error
        embeddings = _validate_embeddings(
            embeddings,
            observation_count=len(records),
            embedding_dim=int(embedding_info.get("dimension", -1)),
        )
        if embedding_info.get("observation_count") != len(records):
            raise ValueError("index record count does not match its manifest")

        if verify_source:
            _, source_records, fingerprint = _validate_source(source)
            if source_records != records:
                raise ValueError("indexed observation metadata no longer matches the source")
            if fingerprint != source_info.get("fingerprint_sha256"):
                raise ValueError("source trajectory has changed since the index was built")
        return cls(
            root=root,
            manifest=manifest,
            records=records,
            embeddings=embeddings,
            source=source,
        )

    @property
    def model_id(self) -> str:
        model = self.manifest["model"]
        assert isinstance(model, dict)
        return str(model["id"])

    @property
    def model_revision(self) -> str:
        model = self.manifest["model"]
        assert isinstance(model, dict)
        return str(model["revision"])

    def observation_embedding(self, observation_id: str) -> np.ndarray:
        try:
            return self.embeddings[self._record_by_id[observation_id]]
        except KeyError as error:
            raise ValueError(f"unknown observation ID: {observation_id}") from error

    def _nearby_actions(self, record: dict[str, object]) -> tuple[dict[str, object], ...]:
        episode_id = str(record["episode_id"])
        step = int(record["step"])
        nearby: list[dict[str, object]] = []
        for candidate_step in range(step - 1, step + 2):
            candidate = self._record_by_step.get((episode_id, candidate_step))
            if candidate is not None:
                nearby.append(
                    {
                        "step": candidate_step,
                        "action": candidate.get("action"),
                    }
                )
        return tuple(nearby)

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
        episode_id: str | None = None,
        exclude_observation_id: str | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self.embeddings.shape[1],) or not np.isfinite(vector).all():
            raise ValueError(
                f"query must be a finite vector with shape ({self.embeddings.shape[1]},)"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("query vector must have a non-zero norm")
        vector = vector / norm

        candidates = [
            index
            for index, record in enumerate(self.records)
            if (episode_id is None or record.get("episode_id") == episode_id)
            and record.get("observation_id") != exclude_observation_id
        ]
        if not candidates:
            if episode_id is not None and not any(
                record.get("episode_id") == episode_id for record in self.records
            ):
                raise ValueError(f"unknown episode ID: {episode_id}")
            raise ValueError("no observations remain after applying query filters")

        scores = self.embeddings[candidates] @ vector
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        return [
            SearchResult(
                rank=rank,
                score=float(np.clip(score, -1.0, 1.0)),
                observation=self.records[index],
                image_path=self.source / str(self.records[index]["image_path"]),
                nearby_actions=self._nearby_actions(self.records[index]),
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


def ensure_matching_encoder(index: MemoryIndex, encoder: Encoder) -> None:
    if (
        encoder.model_id != index.model_id
        or encoder.model_revision != index.model_revision
    ):
        raise ValueError("query encoder does not match the index model and revision")
