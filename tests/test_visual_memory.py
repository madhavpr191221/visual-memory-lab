"""Tests for the persistent CLIP memory and exact retrieval behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_memory_lab.cli import main
from visual_memory_lab.memory import MemoryIndex, build_index


class FakeEncoder:
    model_id = "test/fake-clip"
    model_revision = "test-revision"
    embedding_dim = 4

    def __init__(self) -> None:
        self.next_image = 0

    @property
    def processor_config(self) -> dict[str, object]:
        return {"test": True}

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        rows = []
        for _ in image_paths:
            index = self.next_image
            self.next_image += 1
            row = np.array(
                [1.0, float(index % 5), float((index // 5) % 5), index / 100.0],
                dtype=np.float32,
            )
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows = np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )
        return rows


def _build_test_index(tmp_path: Path, *, episodes: int = 1) -> tuple[Path, Path]:
    source = tmp_path / "run"
    output = tmp_path / "index"
    _write_source(source, episodes=episodes)
    build_index(
        source=source,
        output=output,
        encoder=FakeEncoder(),
        batch_size=7,
    )
    return source, output


def _write_source(source: Path, *, episodes: int) -> None:
    image_root = source / "images"
    image_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for episode in range(episodes):
        episode_id = f"episode-{episode:03d}"
        for step in range(38):
            image_name = f"{episode_id}-{step:04d}.png"
            Image.new("RGB", (56, 56), (episode * 20, step % 255, 100)).save(
                image_root / image_name
            )
            records.append(
                {
                    "observation_id": f"{episode_id}:{step:04d}",
                    "episode_id": episode_id,
                    "step": step,
                    "action": None if step == 0 else "forward",
                    "image_path": image_name,
                    "nearby_actions": [
                        {"step": nearby, "action": "forward"}
                        for nearby in range(max(0, step - 1), min(38, step + 2))
                    ],
                }
            )
    (source / "run.json").write_text(
        json.dumps(
            {
                "observation_count": len(records),
                "image_root": "images",
                "image": {"width": 56, "height": 56, "mode": "RGB"},
            }
        ),
        encoding="utf-8",
    )
    (source / "observations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_index_artifact_preserves_embedding_record_alignment(tmp_path: Path) -> None:
    source, output = _build_test_index(tmp_path)

    manifest = json.loads((output / "index.json").read_text(encoding="utf-8"))
    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["source"]["path"] == str(source.resolve())
    assert manifest["model"] == {
        "id": "test/fake-clip",
        "processor": {"test": True},
        "revision": "test-revision",
    }
    assert embeddings.shape == (38, 4)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert [record["observation_id"] for record in records] == [
        f"episode-000:{step:04d}" for step in range(38)
    ]


def test_index_load_rejects_changed_source_frame(tmp_path: Path) -> None:
    source, output = _build_test_index(tmp_path)
    frame = source / "images" / "episode-000-0000.png"
    frame.write_bytes(frame.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="changed since the index was built"):
        MemoryIndex.load(output)


def test_index_build_rejects_invalid_frame(tmp_path: Path) -> None:
    source = tmp_path / "run"
    _write_source(source, episodes=1)
    frame = source / "images" / "episode-000-0000.png"
    frame.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="could not read trajectory frame"):
        build_index(
            source=source,
            output=tmp_path / "index",
            encoder=FakeEncoder(),
        )


def test_index_build_refuses_occupied_output(tmp_path: Path) -> None:
    source = tmp_path / "run"
    output = tmp_path / "index"
    _write_source(source, episodes=1)
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        build_index(source=source, output=output, encoder=FakeEncoder())
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_exact_search_filters_episode_and_clamps_top_k(tmp_path: Path) -> None:
    _, output = _build_test_index(tmp_path, episodes=2)
    index = MemoryIndex.load(output)
    query = index.observation_embedding("episode-001:0000")

    results = index.search(query, top_k=100, episode_id="episode-001")

    assert len(results) == 38
    assert results[0].observation["observation_id"] == "episode-001:0000"
    assert all(result.observation["episode_id"] == "episode-001" for result in results)
    assert results == sorted(results, key=lambda result: (-result.score, result.rank))


def test_exact_search_breaks_ties_in_source_order(tmp_path: Path) -> None:
    _, output = _build_test_index(tmp_path)
    embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
    embeddings[1] = embeddings[0]
    np.save(output / "embeddings.npy", embeddings, allow_pickle=False)
    index = MemoryIndex.load(output)

    results = index.search(embeddings[0], top_k=2)

    assert [result.observation["observation_id"] for result in results] == [
        "episode-000:0000",
        "episode-000:0001",
    ]
    assert [result.score for result in results] == [1.0, 1.0]


def test_search_excludes_observation_and_returns_action_context(tmp_path: Path) -> None:
    _, output = _build_test_index(tmp_path)
    index = MemoryIndex.load(output)
    query_id = "episode-000:0010"

    results = index.search(
        index.observation_embedding(query_id),
        exclude_observation_id=query_id,
        top_k=37,
    )

    assert all(result.observation["observation_id"] != query_id for result in results)
    step_one = next(result for result in results if result.observation["step"] == 1)
    assert [item["step"] for item in step_one.nearby_actions] == [0, 1, 2]
    step_zero = next(result for result in results if result.observation["step"] == 0)
    assert [item["step"] for item in step_zero.nearby_actions] == [0, 1]
    assert step_zero.nearby_actions[0]["action"] is None


def test_search_validates_filters_and_query_vectors(tmp_path: Path) -> None:
    _, output = _build_test_index(tmp_path)
    index = MemoryIndex.load(output)

    with pytest.raises(ValueError, match="unknown episode ID"):
        index.search(np.ones(4), episode_id="episode-999")
    with pytest.raises(ValueError, match="shape"):
        index.search(np.ones(3))
    with pytest.raises(ValueError, match="non-zero"):
        index.search(np.zeros(4))
    with pytest.raises(ValueError, match="unknown observation ID"):
        index.observation_embedding("missing")


def test_cli_observation_query_supports_json_and_human_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, output = _build_test_index(tmp_path)

    exit_code = main(
        [
            "query",
            "--index",
            str(output),
            "--observation-id",
            "episode-000:0000",
            "--top-k",
            "2",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["query"] == {
        "kind": "observation",
        "observation_id": "episode-000:0000",
    }
    assert len(payload["results"]) == 2
    assert all(
        result["observation"]["observation_id"] != "episode-000:0000"
        for result in payload["results"]
    )

    main(
        [
            "query",
            "--index",
            str(output),
            "--observation-id",
            "episode-000:0000",
            "--include-self",
            "--top-k",
            "1",
        ]
    )
    output_text = capsys.readouterr().out
    assert "episode-000:0000" in output_text
    assert "nearby actions" in output_text
