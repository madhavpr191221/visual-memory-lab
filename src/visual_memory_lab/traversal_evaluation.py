"""Cross-traversal, pose-grounded evaluation for real-image memory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visual_memory_lab.evaluation import (
    RELAXED_THRESHOLD,
    STRICT_THRESHOLD,
    pose_matrix,
    rotation_errors_deg,
)
from visual_memory_lab.memory import MemoryIndex

TRAVERSAL_EVALUATION_SCHEMA_VERSION = "1.0"
TOP_K_VALUES = (1, 5, 10)


def _sequence_id(record: dict[str, object]) -> str:
    value = record.get("sequence_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every traversal-evaluation record requires a sequence_id")
    return value


def _rate(hits: list[bool], eligible: list[bool]) -> dict[str, float | int]:
    eligible_count = sum(eligible)
    covered_hits = sum(
        hit for hit, is_eligible in zip(hits, eligible, strict=True) if is_eligible
    )
    return {
        "eligible_query_targets": eligible_count,
        "covered_rate": float(covered_hits / eligible_count) if eligible_count else 0.0,
        "all_query_target_rate": float(sum(hits) / len(hits)) if hits else 0.0,
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _metric_group(rows: list[dict[str, object]], threshold: str) -> dict[str, object]:
    eligible = [bool(row[f"{threshold}_eligible"]) for row in rows]
    group: dict[str, object] = {
        "coverage": float(sum(eligible) / len(eligible)) if eligible else 0.0,
        "oracle_eligible_query_targets": sum(eligible),
    }
    for k in TOP_K_VALUES:
        group[f"hit_at_{k}"] = _rate(
            [bool(row[f"{threshold}_hit_at_{k}"]) for row in rows], eligible
        )
        group[f"random_hit_at_{k}"] = _rate(
            [bool(row[f"random_{threshold}_hit_at_{k}"]) for row in rows], eligible
        )
    return group


def _validate_indexes(memory: MemoryIndex, queries: MemoryIndex) -> None:
    if memory.model_id != queries.model_id or memory.model_revision != queries.model_revision:
        raise ValueError("memory and query indexes must use the same encoder revision")

    memory_ids = {str(record.get("observation_id")) for record in memory.records}
    query_ids = {str(record.get("observation_id")) for record in queries.records}
    overlap = memory_ids & query_ids
    if overlap:
        raise ValueError("memory and query indexes must have disjoint observation identities")

    for record in [*memory.records, *queries.records]:
        _sequence_id(record)
        pose_matrix(record)


def evaluate_cross_traversal_retrieval(
    memory: MemoryIndex,
    queries: MemoryIndex,
    *,
    seed: int = 42,
    query_indices: list[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate every selected query against each designated memory traversal."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    _validate_indexes(memory, queries)
    selected = query_indices if query_indices is not None else list(range(len(queries.records)))
    if not selected:
        raise ValueError("traversal evaluation requires at least one query")
    if any(index < 0 or index >= len(queries.records) for index in selected):
        raise ValueError("query index is out of range")

    target_indices = {
        sequence: np.asarray(
            [
                index
                for index, record in enumerate(memory.records)
                if _sequence_id(record) == sequence
            ],
            dtype=np.int64,
        )
        for sequence in sorted({_sequence_id(record) for record in memory.records})
    }
    target_poses = {
        sequence: np.stack([pose_matrix(memory.records[int(index)]) for index in indices])
        for sequence, indices in target_indices.items()
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for query_index in selected:
        query_record = queries.records[query_index]
        query_pose = pose_matrix(query_record)
        source_sequence = _sequence_id(query_record)
        for target_sequence, candidates in target_indices.items():
            poses = target_poses[target_sequence]
            translation_errors = np.linalg.norm(
                poses[:, :3, 3] - query_pose[:3, 3], axis=1
            )
            angle_errors = rotation_errors_deg(query_pose[:3, :3], poses[:, :3, :3])
            strict_relevant = (translation_errors <= STRICT_THRESHOLD.distance_m) & (
                angle_errors <= STRICT_THRESHOLD.angle_deg
            )
            relaxed_relevant = (translation_errors <= RELAXED_THRESHOLD.distance_m) & (
                angle_errors <= RELAXED_THRESHOLD.angle_deg
            )

            scores = memory.embeddings[candidates] @ queries.embeddings[query_index]
            ranked_local = np.argsort(-scores, kind="stable")[: min(10, len(candidates))]
            random_local = rng.choice(
                len(candidates), size=min(10, len(candidates)), replace=False
            )
            top1_local = int(ranked_local[0])
            row: dict[str, object] = {
                "query_id": query_record["observation_id"],
                "source_sequence_id": source_sequence,
                "target_sequence_id": target_sequence,
                "step": query_record.get("step"),
                "strict_eligible": bool(strict_relevant.any()),
                "relaxed_eligible": bool(relaxed_relevant.any()),
                "strict_relevant_count": int(strict_relevant.sum()),
                "relaxed_relevant_count": int(relaxed_relevant.sum()),
                "top1_translation_error_m": float(translation_errors[top1_local]),
                "top1_rotation_error_deg": float(angle_errors[top1_local]),
                "nearest_pose_translation_m": float(translation_errors.min()),
                "retrievals": [
                    {
                        "rank": rank,
                        "observation_id": memory.records[int(candidates[local_index])][
                            "observation_id"
                        ],
                        "score": float(scores[local_index]),
                        "translation_error_m": float(translation_errors[local_index]),
                        "rotation_error_deg": float(angle_errors[local_index]),
                    }
                    for rank, local_index in enumerate(ranked_local, start=1)
                ],
            }
            for threshold, relevant in (
                ("strict", strict_relevant),
                ("relaxed", relaxed_relevant),
            ):
                for k in TOP_K_VALUES:
                    row[f"{threshold}_hit_at_{k}"] = bool(relevant[ranked_local[:k]].any())
                    row[f"random_{threshold}_hit_at_{k}"] = bool(
                        relevant[random_local[:k]].any()
                    )
            rows.append(row)

    source_sequences = sorted({_sequence_id(queries.records[index]) for index in selected})
    target_sequences = sorted(target_indices)
    metrics: dict[str, object] = {
        "schema_version": TRAVERSAL_EVALUATION_SCHEMA_VERSION,
        "protocol": "designated-reference-traversal",
        "chronology": "not_available",
        "query_count": len(selected),
        "query_target_count": len(rows),
        "pair_count": len(source_sequences) * len(target_sequences),
        "source_sequences": source_sequences,
        "target_sequences": target_sequences,
        "thresholds": {
            "strict": {
                "distance_m": STRICT_THRESHOLD.distance_m,
                "angle_deg": STRICT_THRESHOLD.angle_deg,
            },
            "relaxed": {
                "distance_m": RELAXED_THRESHOLD.distance_m,
                "angle_deg": RELAXED_THRESHOLD.angle_deg,
            },
        },
        "seed": seed,
        "strict": _metric_group(rows, "strict"),
        "relaxed": _metric_group(rows, "relaxed"),
        "top1_translation_error_m": _summary(
            [float(row["top1_translation_error_m"]) for row in rows]
        ),
        "top1_rotation_error_deg": _summary(
            [float(row["top1_rotation_error_deg"]) for row in rows]
        ),
    }

    per_pair: dict[str, dict[str, object]] = {}
    for source in source_sequences:
        for target in target_sequences:
            pair_rows = [
                row
                for row in rows
                if row["source_sequence_id"] == source
                and row["target_sequence_id"] == target
            ]
            per_pair[f"{source}->{target}"] = {
                "query_count": len(pair_rows),
                "strict": _metric_group(pair_rows, "strict"),
                "relaxed": _metric_group(pair_rows, "relaxed"),
                "top1_translation_error_m": _summary(
                    [float(row["top1_translation_error_m"]) for row in pair_rows]
                ),
                "top1_rotation_error_deg": _summary(
                    [float(row["top1_rotation_error_deg"]) for row in pair_rows]
                ),
            }
    metrics["per_pair"] = per_pair

    macro_pair: dict[str, object] = {}
    for threshold in ("strict", "relaxed"):
        pair_groups = [pair[threshold] for pair in per_pair.values()]
        assert all(isinstance(group, dict) for group in pair_groups)
        threshold_macro: dict[str, object] = {
            "coverage": float(np.mean([float(group["coverage"]) for group in pair_groups]))
        }
        for k in TOP_K_VALUES:
            eligible_groups = [
                group
                for group in pair_groups
                if int(group["oracle_eligible_query_targets"]) > 0
            ]
            threshold_macro[f"hit_at_{k}_covered_rate"] = float(
                np.mean(
                    [float(group[f"hit_at_{k}"]["covered_rate"]) for group in eligible_groups]
                )
            ) if eligible_groups else 0.0
            threshold_macro[f"hit_at_{k}_all_query_target_rate"] = float(
                np.mean(
                    [
                        float(group[f"hit_at_{k}"]["all_query_target_rate"])
                        for group in pair_groups
                    ]
                )
            )
            threshold_macro[f"random_hit_at_{k}_covered_rate"] = float(
                np.mean(
                    [
                        float(group[f"random_hit_at_{k}"]["covered_rate"])
                        for group in eligible_groups
                    ]
                )
            ) if eligible_groups else 0.0
        macro_pair[threshold] = threshold_macro
    metrics["macro_pair"] = macro_pair
    return metrics, rows


def write_traversal_evaluation(
    *,
    memory: MemoryIndex,
    queries: MemoryIndex,
    output: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Run and persist the complete cross-traversal evaluation."""

    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output path is not empty: {output}")
    metrics, rows = evaluate_cross_traversal_retrieval(memory, queries, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "per_query_target.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return metrics
