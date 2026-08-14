"""Small, reproducible Charades temporal-memory artifacts.

The first video slice deliberately keeps the data model simple: official
Charades annotations become searchable temporal windows.  CLIP/temporal-model
training can consume the same windows later without changing the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_ACTION_RE = re.compile(r"(?P<id>c\d+)\s+(?P<start>\d+(?:\.\d+)?)\s+(?P<end>\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class CharadesAction:
    action_id: str
    name: str
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_classes(path: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    for line in _read_lines(path):
        action_id, name = line.split(maxsplit=1)
        classes[action_id] = name
    return classes


def parse_actions(value: str, classes: dict[str, str]) -> list[CharadesAction]:
    actions: list[CharadesAction] = []
    for match in _ACTION_RE.finditer(value or ""):
        action_id = match.group("id")
        actions.append(
            CharadesAction(
                action_id=action_id,
                name=classes.get(action_id, action_id),
                start_s=float(match.group("start")),
                end_s=float(match.group("end")),
            )
        )
    return actions


def load_annotation_rows(root: Path) -> list[dict[str, object]]:
    annotations = root / "annotations_and_evaluations"
    classes = load_classes(annotations / "Charades_v1_classes.txt")
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        with (annotations / f"Charades_v1_{split}.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                video_id = str(row["id"])
                video_path = root / "videos" / f"{video_id}.mp4"
                if not video_path.is_file():
                    continue
                actions = parse_actions(str(row.get("actions", "")), classes)
                rows.append(
                    {
                        "video_id": video_id,
                        "split": split,
                        "video_path": str(video_path.resolve()),
                        "subject": row.get("subject", ""),
                        "scene": row.get("scene", ""),
                        "description": row.get("descriptions", ""),
                        "script": row.get("script", ""),
                        "objects": [item for item in str(row.get("objects", "")).split(";") if item],
                        "length_s": float(row.get("length", 0.0) or 0.0),
                        "actions": [action.to_dict() for action in actions],
                    }
                )
    return rows


def _stable_key(row: dict[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row['video_id']}".encode()).hexdigest()


def _select(rows: Iterable[dict[str, object]], limit: int, seed: int) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: _stable_key(row, seed))
    return ordered[:limit]


def prepare_charades_dataset(
    dataset_root: Path,
    output: Path,
    *,
    train_limit: int = 300,
    test_limit: int = 100,
    seed: int = 42,
) -> dict[str, object]:
    """Write a deterministic, small Charades manifest without copying videos."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output path is not empty: {output.resolve()}")
    rows = load_annotation_rows(dataset_root)
    train = _select((row for row in rows if row["split"] == "train"), train_limit, seed)
    test = _select((row for row in rows if row["split"] == "test"), test_limit, seed + 1)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in [*train, *test]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "dataset": "charades",
        "seed": seed,
        "train_count": len(train),
        "test_count": len(test),
        "video_count": len(train) + len(test),
        "manifest": str(manifest.resolve()),
        "annotation_source": str((dataset_root / "annotations_and_evaluations").resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_temporal_windows(
    manifest: Path,
    output: Path,
    *,
    window_s: float = 4.0,
    stride_s: float = 2.0,
) -> dict[str, object]:
    if window_s <= 0 or stride_s <= 0:
        raise ValueError("window_s and stride_s must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output path is not empty: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    windows: list[dict[str, object]] = []
    for row in load_manifest(manifest):
        duration = float(row.get("length_s", 0.0))
        if duration <= 0:
            continue
        actions = row.get("actions", [])
        start = 0.0
        while start < duration:
            end = min(start + window_s, duration)
            overlapping = [
                action for action in actions
                if isinstance(action, dict)
                and float(action.get("end_s", 0.0)) > start
                and float(action.get("start_s", 0.0)) < end
            ]
            windows.append(
                {
                    "window_id": f"{row['video_id']}:{start:.2f}-{end:.2f}",
                    "video_id": row["video_id"],
                    "split": row["split"],
                    "video_path": row["video_path"],
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "actions": overlapping,
                    "objects": row.get("objects", []),
                    "description": row.get("description", ""),
                }
            )
            if end >= duration:
                break
            start += stride_s
    with (output / "windows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for window in windows:
            handle.write(json.dumps(window, sort_keys=True) + "\n")
    summary = {
        "window_count": len(windows),
        "window_s": window_s,
        "stride_s": stride_s,
        "windows": str((output / "windows.jsonl").resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def search_windows(windows: list[dict[str, object]], query: str, *, top_k: int = 8) -> list[dict[str, object]]:
    """Transparent lexical baseline used before learned temporal retrieval."""
    stopwords = {
        "when", "did", "the", "person", "what", "was", "were", "where",
        "show", "me", "please", "this", "that", "with", "from", "into",
        "does", "happen", "happened", "someone", "something",
    }
    terms = {
        term.lower()
        for term in re.findall(r"[a-z0-9]+", query)
        if len(term) > 2 and term.lower() not in stopwords
    }
    action_hints = {
        "open", "opening", "close", "closing", "sit", "sitting", "stand",
        "standing", "take", "taking", "pick", "picking", "put", "putting",
        "hold", "holding", "carry", "carrying", "walk", "walking", "eat",
        "eating", "drink", "drinking", "look", "looking", "fix", "fixing",
    }
    asks_for_action = bool(terms & action_hints)

    def variants(tokens: set[str]) -> set[str]:
        expanded = set(tokens)
        for token in tokens:
            if token.endswith("ing") and len(token) > 5:
                stem = token[:-3]
                expanded.add(stem)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    expanded.add(stem[:-1])
            if token.endswith("ed") and len(token) > 4:
                expanded.add(token[:-2])
        return expanded

    scored: list[tuple[float, dict[str, object]]] = []
    for window in windows:
        actions = window.get("actions", [])
        action_text = " ".join(str(action.get("name", "")) for action in actions if isinstance(action, dict))
        action_terms = variants(set(re.findall(r"[a-z0-9]+", action_text.lower())))
        description_terms = variants(set(re.findall(r"[a-z0-9]+", str(window.get("description", "")).lower())))
        object_terms = variants(set(re.findall(r"[a-z0-9]+", " ".join(map(str, window.get("objects", []))).lower())))
        action_hits = len(terms & action_terms)
        if asks_for_action and action_hits == 0:
            continue
        hits = (2.0 * action_hits) + len(terms & description_terms) + (0.5 * len(terms & object_terms))
        score = hits / max(1.0, 2.0 * len(terms))
        if score > 0:
            scored.append((score, window))
    scored.sort(key=lambda item: (-item[0], str(item[1]["window_id"])))
    selected: list[tuple[float, dict[str, object]]] = []
    per_video: dict[str, int] = {}
    for score, window in scored:
        video_id = str(window.get("video_id", ""))
        if per_video.get(video_id, 0) >= 2:
            continue
        selected.append((score, window))
        per_video[video_id] = per_video.get(video_id, 0) + 1
        if len(selected) >= top_k:
            break
    return [{**window, "score": round(score, 4), "retrieval_mode": "annotation_lexical_baseline"} for score, window in selected]
