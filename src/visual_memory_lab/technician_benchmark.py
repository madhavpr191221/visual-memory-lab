"""Self-guided technician-style benchmark for the office evidence system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ANSWERABILITY = {
    "supported",
    "supported_with_limits",
    "requires_manual_review",
    "unsupported",
}


@dataclass(frozen=True)
class TechnicianQuestion:
    question_id: str
    question: str
    category: str
    dataset: str
    answerability: str
    expected_evidence_ids: tuple[str, ...]
    expected_zone: str | None
    expected_visit: str | None
    source_observation_id: str
    expected_object_class: str | None
    expected_artifact: str | None
    rationale: str


def load_questions(path: Path) -> list[TechnicianQuestion]:
    """Load and validate the versioned benchmark manifest."""
    questions: list[TechnicianQuestion] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            required = {
                "question_id", "question", "category", "dataset", "answerability",
                "source_observation_id",
            }
            missing = required - payload.keys()
            if missing:
                raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
            question_id = str(payload["question_id"])
            if question_id in seen:
                raise ValueError(f"duplicate question_id: {question_id}")
            answerability = str(payload["answerability"])
            if answerability not in ANSWERABILITY:
                raise ValueError(f"invalid answerability: {answerability}")
            evidence_ids = tuple(str(item) for item in payload.get("expected_evidence_ids", []))
            if not evidence_ids and not payload.get("expected_zone") and not payload.get("expected_visit") and not payload.get("expected_object_class") and not payload.get("expected_artifact"):
                raise ValueError(f"line {line_number} needs an expected evidence rule")
            questions.append(
                TechnicianQuestion(
                    question_id=question_id,
                    question=str(payload["question"]),
                    category=str(payload["category"]),
                    dataset=str(payload["dataset"]),
                    answerability=answerability,
                    expected_evidence_ids=evidence_ids,
                    expected_zone=(str(payload["expected_zone"]) if payload.get("expected_zone") else None),
                    expected_visit=(str(payload["expected_visit"]) if payload.get("expected_visit") else None),
                    source_observation_id=str(payload["source_observation_id"]),
                    expected_object_class=(str(payload["expected_object_class"]) if payload.get("expected_object_class") else None),
                    expected_artifact=(str(payload["expected_artifact"]) if payload.get("expected_artifact") else None),
                    rationale=str(payload.get("rationale", "")),
                )
            )
            seen.add(question_id)
    if not questions:
        raise ValueError(f"question manifest is empty: {path}")
    return questions


def reciprocal_rank(results: Iterable[dict[str, object]], expected_ids: set[str]) -> float:
    for result in results:
        if str(result.get("observation_id", "")) in expected_ids:
            rank = int(result.get("rank", 0))
            return 1.0 / rank if rank > 0 else 0.0
    return 0.0


def evaluate_questions(
    questions: Iterable[TechnicianQuestion],
    *,
    search: Callable[[str], list[dict[str, object]]] | None = None,
    available_artifacts: set[str] | None = None,
    artifact_records: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    """Evaluate retrieval evidence and safe boundary handling.

    The evaluator intentionally does not generate natural-language answers. It
    scores evidence recovery and records whether a question should be answered
    or handed to a human for review.
    """
    rows: list[dict[str, object]] = []
    available_artifacts = available_artifacts or set()
    artifact_records = artifact_records or {}
    for item in questions:
        results = search(item.source_observation_id) if search and item.dataset == "7-scenes-office" else []
        if item.dataset == "eth-office":
            results = [
                {
                    "artifact_id": record.get("artifact_id"),
                    "observation_id": record.get("frame_id", record.get("observation_id")),
                    "object_class": record.get("object_class", record.get("canonical_class")),
                    "artifact": artifact,
                    "rank": 1,
                }
                for artifact, records in artifact_records.items()
                for record in records
                if str(record.get("frame_id", record.get("observation_id"))) == item.source_observation_id
            ]
        result_ids = {str(result.get("observation_id", "")) for result in results}
        expected = set(item.expected_evidence_ids)
        zone_hit = any(
            item.expected_zone
            and str(result.get("zone_slug", "")) == item.expected_zone
            for result in results
        ) if item.expected_zone else False
        visit_hit = any(str(result.get("visit_id", "")) == item.expected_visit for result in results) if item.expected_visit else False
        object_hit = any(str(result.get("object_class", "")) == item.expected_object_class for result in results) if item.expected_object_class else False
        artifact_hit = any(
            item.expected_artifact == result.get("artifact")
            and (not item.expected_object_class or item.expected_object_class == result.get("object_class"))
            for result in results
        ) if item.expected_artifact else False
        evidence_hit = (bool(expected & result_ids) or zone_hit or visit_hit or object_hit or artifact_hit) if (expected or item.expected_zone or item.expected_visit or item.expected_object_class or item.expected_artifact) else None
        artifact_available = item.dataset == "7-scenes-office" or artifact_hit
        safe_boundary = item.answerability in {"requires_manual_review", "unsupported"}
        rows.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "category": item.category,
                "dataset": item.dataset,
                "expected_answerability": item.answerability,
                "expected_evidence_ids": list(item.expected_evidence_ids),
                "source_observation_id": item.source_observation_id,
                "expected_zone": item.expected_zone,
                "expected_visit": item.expected_visit,
                "expected_object_class": item.expected_object_class,
                "expected_artifact": item.expected_artifact,
                "returned_evidence_ids": [str(result.get("observation_id", "")) for result in results],
                "evidence_hit": evidence_hit,
                "zone_hit": zone_hit,
                "visit_hit": visit_hit,
                "object_hit": object_hit,
                "artifact_hit": artifact_hit,
                "reciprocal_rank": reciprocal_rank(results, expected) if expected else None,
                "artifact_available": artifact_available,
                "safe_boundary_case": safe_boundary,
                "rationale": item.rationale,
            }
        )
    scored = [row for row in rows if row["evidence_hit"] is not None and not row["safe_boundary_case"]]
    hits = sum(bool(row["evidence_hit"]) for row in scored)
    boundary_cases = [row for row in rows if row["safe_boundary_case"]]
    return {
        "question_count": len(rows),
        "scored_question_count": len(scored),
        "evidence_recall": hits / len(scored) if scored else None,
        "mean_reciprocal_rank": (
            sum(float(row["reciprocal_rank"] or 0.0) for row in scored) / len(scored)
            if scored else None
        ),
        "boundary_question_count": len(boundary_cases),
        "category_metrics": {
            category: {
                "question_count": len(group),
                "scored_question_count": sum(row["evidence_hit"] is not None for row in group),
                "evidence_recall": (
                    sum(bool(row["evidence_hit"]) for row in group if row["evidence_hit"] is not None)
                    / sum(row["evidence_hit"] is not None for row in group)
                    if any(row["evidence_hit"] is not None for row in group) else None
                ),
            }
            for category in sorted({str(row["category"]) for row in rows})
            for group in [[row for row in rows if row["category"] == category]]
        },
        "artifact_available_count": sum(bool(row["artifact_available"]) for row in rows),
        "questions": rows,
    }


def write_benchmark(
    *,
    questions_path: Path,
    output: Path,
    search: Callable[[str], list[dict[str, object]]] | None = None,
    available_artifacts: set[str] | None = None,
    artifact_records: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output path is not empty: {output.resolve()}")
    output.mkdir(parents=True, exist_ok=True)
    payload = evaluate_questions(
        load_questions(questions_path),
        search=search,
        available_artifacts=available_artifacts,
        artifact_records=artifact_records,
    )
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["questions"]:
            handle.write(json.dumps(row) + "\n")
    category_lines = [
        f"- {category}: {metrics['scored_question_count']} scored, "
        f"evidence recall {metrics['evidence_recall']}"
        for category, metrics in payload["category_metrics"].items()
    ]
    (output / "report.md").write_text(
        "# Technician benchmark report\n\n"
        f"Questions: {payload['question_count']}\n\n"
        "## Category metrics\n\n"
        + "\n".join(category_lines)
        + "\n\nThis report separates retrieval evidence from boundary questions; it is not a single accuracy score.\n",
        encoding="utf-8",
    )
    return payload
