"""Pose-grounded and semantic-zone evaluation for real-image memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from visual_memory_lab.memory import MemoryIndex, ensure_matching_encoder


class TextEncoder(Protocol):
    model_id: str
    model_revision: str

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class PoseThreshold:
    distance_m: float
    angle_deg: float


STRICT_THRESHOLD = PoseThreshold(0.25, 30.0)
RELAXED_THRESHOLD = PoseThreshold(0.50, 30.0)


def pose_matrix(record: dict[str, object]) -> np.ndarray:
    pose = record.get("camera_pose")
    if not isinstance(pose, dict) or pose.get("convention") != "camera_to_world":
        raise ValueError("every evaluation record requires a camera_to_world pose")
    matrix = np.asarray(pose.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("every evaluation record requires a finite 4x4 pose matrix")
    return matrix


def rotation_errors_deg(query_rotation: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Return SO(3) geodesic angles from one rotation to many rotations."""

    relative = np.einsum("ij,njk->nik", query_rotation.T, rotations)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _rate(hits: list[bool], eligible: list[bool]) -> dict[str, float | int]:
    eligible_count = sum(eligible)
    covered_hits = sum(hit for hit, is_eligible in zip(hits, eligible, strict=True) if is_eligible)
    return {
        "eligible_queries": eligible_count,
        "covered_rate": float(covered_hits / eligible_count) if eligible_count else 0.0,
        "all_query_rate": float(sum(hits) / len(hits)) if hits else 0.0,
    }


def evaluate_pose_retrieval(
    memory: MemoryIndex,
    queries: MemoryIndex,
    *,
    seed: int = 42,
    query_indices: list[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate held-out image embeddings against pose-grounded relevance."""

    if memory.model_id != queries.model_id or memory.model_revision != queries.model_revision:
        raise ValueError("memory and query indexes must use the same encoder revision")
    selected = query_indices if query_indices is not None else list(range(len(queries.records)))
    if not selected:
        raise ValueError("pose evaluation requires at least one query")

    memory_poses = np.stack([pose_matrix(record) for record in memory.records])
    memory_translations = memory_poses[:, :3, 3]
    memory_rotations = memory_poses[:, :3, :3]
    rng = np.random.default_rng(seed)
    per_query: list[dict[str, object]] = []

    for query_index in selected:
        record = queries.records[query_index]
        query_pose = pose_matrix(record)
        translation_errors = np.linalg.norm(
            memory_translations - query_pose[:3, 3], axis=1
        )
        angle_errors = rotation_errors_deg(query_pose[:3, :3], memory_rotations)
        strict_relevant = (translation_errors <= STRICT_THRESHOLD.distance_m) & (
            angle_errors <= STRICT_THRESHOLD.angle_deg
        )
        relaxed_relevant = (translation_errors <= RELAXED_THRESHOLD.distance_m) & (
            angle_errors <= RELAXED_THRESHOLD.angle_deg
        )

        scores = memory.embeddings @ queries.embeddings[query_index]
        ranked = np.argsort(-scores, kind="stable")[:10]
        random_ranked = rng.choice(len(memory.records), size=min(10, len(memory.records)), replace=False)
        top1 = int(ranked[0])
        row: dict[str, object] = {
            "query_id": record["observation_id"],
            "sequence_id": record.get("sequence_id", record.get("episode_id")),
            "step": record.get("step"),
            "strict_eligible": bool(strict_relevant.any()),
            "relaxed_eligible": bool(relaxed_relevant.any()),
            "strict_relevant_count": int(strict_relevant.sum()),
            "relaxed_relevant_count": int(relaxed_relevant.sum()),
            "top1_translation_error_m": float(translation_errors[top1]),
            "top1_rotation_error_deg": float(angle_errors[top1]),
            "nearest_pose_translation_m": float(translation_errors.min()),
            "retrievals": [
                {
                    "rank": rank,
                    "observation_id": memory.records[int(index)]["observation_id"],
                    "score": float(scores[index]),
                    "translation_error_m": float(translation_errors[index]),
                    "rotation_error_deg": float(angle_errors[index]),
                }
                for rank, index in enumerate(ranked, start=1)
            ],
        }
        for name, relevant in (("strict", strict_relevant), ("relaxed", relaxed_relevant)):
            for k in (1, 5, 10):
                row[f"{name}_hit_at_{k}"] = bool(relevant[ranked[:k]].any())
                row[f"random_{name}_hit_at_{k}"] = bool(relevant[random_ranked[:k]].any())
        per_query.append(row)

    metrics: dict[str, object] = {
        "query_count": len(per_query),
        "thresholds": {
            "strict": {"distance_m": 0.25, "angle_deg": 30.0},
            "relaxed": {"distance_m": 0.50, "angle_deg": 30.0},
        },
        "seed": seed,
    }
    for name in ("strict", "relaxed"):
        eligible = [bool(row[f"{name}_eligible"]) for row in per_query]
        group: dict[str, object] = {
            "coverage": float(sum(eligible) / len(eligible)),
            "oracle_eligible_queries": sum(eligible),
        }
        for k in (1, 5, 10):
            group[f"hit_at_{k}"] = _rate(
                [bool(row[f"{name}_hit_at_{k}"]) for row in per_query], eligible
            )
            group[f"random_hit_at_{k}"] = _rate(
                [bool(row[f"random_{name}_hit_at_{k}"]) for row in per_query], eligible
            )
        metrics[name] = group
    metrics["top1_translation_error_m"] = _summary(
        [float(row["top1_translation_error_m"]) for row in per_query]
    )
    metrics["top1_rotation_error_deg"] = _summary(
        [float(row["top1_rotation_error_deg"]) for row in per_query]
    )

    per_sequence: dict[str, dict[str, object]] = {}
    for sequence in sorted({str(row["sequence_id"]) for row in per_query}):
        rows = [row for row in per_query if str(row["sequence_id"]) == sequence]
        per_sequence[sequence] = {
            "query_count": len(rows),
            "strict_coverage": float(sum(bool(row["strict_eligible"]) for row in rows) / len(rows)),
            "strict_hit_at_1_all_queries": float(
                sum(bool(row["strict_hit_at_1"]) for row in rows) / len(rows)
            ),
            "strict_hit_at_5_all_queries": float(
                sum(bool(row["strict_hit_at_5"]) for row in rows) / len(rows)
            ),
        }
    metrics["per_sequence"] = per_sequence
    return metrics, per_query


def evaluate_text_zones(
    memory: MemoryIndex,
    zones_path: Path,
    encoder: TextEncoder,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate frozen zone prompts against VLM-assisted silver labels."""

    ensure_matching_encoder(memory, encoder)
    try:
        artifact = json.loads(zones_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read zone artifact {zones_path}: {error}") from error
    assignments = artifact.get("assignments")
    zones = artifact.get("zones")
    if not isinstance(assignments, dict) or not isinstance(zones, list) or not zones:
        raise ValueError("zone artifact must contain assignments and at least one zone")

    record_ids = [str(record["observation_id"]) for record in memory.records]
    prompts: list[tuple[str, str, str]] = []
    for zone in zones:
        if not isinstance(zone, dict) or not isinstance(zone.get("slug"), str):
            raise ValueError("zone definitions must contain string slugs")
        zone_prompts = zone.get("prompts")
        if not isinstance(zone_prompts, dict):
            raise ValueError("each zone must contain prompt variants")
        for style in ("name", "landmarks", "technician_question"):
            prompt = zone_prompts.get(style)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"zone {zone['slug']} is missing prompt style {style}")
            prompts.append((str(zone["slug"]), style, prompt))

    embeddings = encoder.encode_texts([prompt for _, _, prompt in prompts])
    rows: list[dict[str, object]] = []
    for prompt_index, (slug, style, prompt) in enumerate(prompts):
        relevant = np.asarray([assignments.get(record_id) == slug for record_id in record_ids])
        if not relevant.any():
            raise ValueError(f"zone {slug} has no assigned memory frames")
        scores = memory.embeddings @ embeddings[prompt_index]
        ranked = np.argsort(-scores, kind="stable")[:10]
        row: dict[str, object] = {
            "zone": slug,
            "style": style,
            "prompt": prompt,
            "relevant_frame_count": int(relevant.sum()),
            "retrievals": [
                {
                    "rank": rank,
                    "observation_id": record_ids[int(index)],
                    "score": float(scores[index]),
                    "relevant": bool(relevant[index]),
                }
                for rank, index in enumerate(ranked, start=1)
            ],
        }
        for k in (1, 5, 10):
            row[f"hit_at_{k}"] = bool(relevant[ranked[:k]].any())
            row[f"precision_at_{k}"] = float(relevant[ranked[:k]].mean())
        rows.append(row)

    assigned_count = sum(record_id in assignments and assignments[record_id] != "unassigned" for record_id in record_ids)
    metrics: dict[str, object] = {
        "zone_count": len(zones),
        "prompt_count": len(rows),
        "assigned_frame_count": assigned_count,
        "unassigned_frame_count": len(record_ids) - assigned_count,
        "assignment_coverage": float(assigned_count / len(record_ids)),
    }
    for k in (1, 5, 10):
        metrics[f"macro_hit_at_{k}"] = float(np.mean([bool(row[f"hit_at_{k}"]) for row in rows]))
        metrics[f"macro_precision_at_{k}"] = float(
            np.mean([float(row[f"precision_at_{k}"]) for row in rows])
        )
    metrics["per_zone"] = {
        slug: {
            f"macro_hit_at_{k}": float(
                np.mean([bool(row[f"hit_at_{k}"]) for row in rows if row["zone"] == slug])
            )
            for k in (1, 5, 10)
        }
        for slug in sorted({str(row["zone"]) for row in rows})
    }
    return metrics, rows


def write_evaluation(
    *,
    memory: MemoryIndex,
    queries: MemoryIndex,
    zones_path: Path,
    encoder: TextEncoder,
    output: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Run and persist full, stride-10, and text-query evaluations."""

    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output path is not empty: {output}")
    full_metrics, per_query = evaluate_pose_retrieval(memory, queries, seed=seed)
    stride_indices = [
        index
        for index, record in enumerate(queries.records)
        if int(record.get("step", index)) % 10 == 0
    ]
    stride_metrics, _ = evaluate_pose_retrieval(
        memory, queries, seed=seed, query_indices=stride_indices
    )
    text_metrics, text_rows = evaluate_text_zones(memory, zones_path, encoder)
    metrics = {
        "schema_version": "1.0",
        "memory_index": str(memory.root),
        "query_index": str(queries.root),
        "zones": str(zones_path.resolve()),
        "pose": full_metrics,
        "pose_stride_10": stride_metrics,
        "text_zones": text_metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "per_query.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in per_query),
        encoding="utf-8",
    )
    (output / "text_queries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in text_rows),
        encoding="utf-8",
    )
    return metrics
