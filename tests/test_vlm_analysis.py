"""Tests for explicit, cached, citation-checked evidence analysis."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from visual_memory_lab.vlm_analysis import EvidenceAnalyzer, GroundedJudgment


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **_: object) -> object:
        self.calls += 1
        parsed = GroundedJudgment(
            question_type="location",
            supported=True,
            answer="The evidence shows a desk beside the window.",
            evidence_citations=[
                {"observation_id": "memory:0", "claim": "A window is visible beside the desk."}
            ],
            evidence_strength="high",
            limitations=["The public dataset has no calendar time."],
        )
        return SimpleNamespace(output_parsed=parsed, model="test-model-2026")


def test_text_analysis_is_cached_but_image_analysis_is_not(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (12, 8), "navy").save(image_path)
    responses = FakeResponses()
    analyzer = EvidenceAnalyzer(
        model="test-model",
        cache_dir=tmp_path / "cache",
        client=SimpleNamespace(responses=responses),
    )

    first = analyzer.analyze(question="Where is the desk?", evidence=[("memory:0", image_path)])
    second = analyzer.analyze(question="Where is the desk?", evidence=[("memory:0", image_path)])
    with Image.open(image_path) as query:
        third = analyzer.analyze(
            question="Where was this view taken?",
            evidence=[("memory:0", image_path)],
            query_image=query,
        )

    assert first["cached"] is False
    assert second["cached"] is True
    assert third["cached"] is False
    assert responses.calls == 2
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
