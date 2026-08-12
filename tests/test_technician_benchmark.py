import json
from pathlib import Path

import pytest

from visual_memory_lab.technician_benchmark import evaluate_questions, load_questions


def test_load_questions_validates_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    row = {"question_id": "q1", "question": "Where?", "category": "place", "dataset": "7-scenes-office", "source_observation_id": "x", "answerability": "supported", "expected_zone": "zone"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_evaluate_questions_scores_evidence_and_zone() -> None:
    path = Path("data/phase7/technician_questions.jsonl")
    questions = load_questions(path)[:1]
    payload = evaluate_questions(
        questions,
        search=lambda _: [{"rank": 1, "observation_id": "frame-1", "zone_slug": questions[0].expected_zone}],
    )
    assert payload["question_count"] == 1
    assert payload["evidence_recall"] == 1.0
    assert payload["questions"][0]["zone_hit"] is True


def test_boundary_questions_are_not_evidence_scored() -> None:
    questions = [load_questions(Path("data/phase7/technician_questions.jsonl"))[-2]]
    payload = evaluate_questions(questions)
    assert payload["questions"][0]["safe_boundary_case"] is True
    assert payload["scored_question_count"] == 0
