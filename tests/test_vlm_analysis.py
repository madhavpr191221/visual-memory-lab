"""Tests for explicit, cached, citation-checked evidence analysis."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from visual_memory_lab.vlm_analysis import EvidenceAnalyzer, GroundedJudgment, VideoGroundedAnswer


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


class FakeVideoResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs: object) -> object:
        self.calls += 1
        schema = kwargs["text_format"]
        if schema is VideoGroundedAnswer:
            parsed = VideoGroundedAnswer(
                answer="A person walks through the doorway carrying a bag.",
                event_label="Walking through a doorway",
                supported=True,
                confidence="medium",
                evidence_citations=[{"observation_id": "frame-00", "claim": "A person is visible in the doorway."}],
                limitations=["The recording does not establish persistent identity."],
            )
        else:
            parsed = GroundedJudgment(
                question_type="location", supported=True, answer="A desk is visible.",
                evidence_citations=[{"observation_id": "memory:0", "claim": "A desk is visible."}],
                evidence_strength="high", limitations=[],
            )
        return SimpleNamespace(output_parsed=parsed, model="test-video-model")


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


def test_video_synthesis_is_structured_and_cached(tmp_path: Path) -> None:
    responses = FakeVideoResponses()
    analyzer = EvidenceAnalyzer(model="test-model", cache_dir=tmp_path / "cache", client=SimpleNamespace(responses=responses))
    frame = Image.new("RGB", (12, 8), "navy")
    first = analyzer.synthesize_video(
        question="When did the person walk through the doorway?",
        video_id="ABC12", event_label="Walking through a doorway", start_s=2.0, end_s=4.0,
        frames=[("frame-00", 2.5, frame)], actions=["Walking through a doorway"], objects=["doorway"],
    )
    second = analyzer.synthesize_video(
        question="When did the person walk through the doorway?",
        video_id="ABC12", event_label="Walking through a doorway", start_s=2.0, end_s=4.0,
        frames=[("frame-00", 2.5, frame)], actions=["Walking through a doorway"], objects=["doorway"],
    )
    assert first["source"] == "vlm_video_synthesis"
    assert first["cached"] is False
    assert second["cached"] is True
    assert responses.calls == 1
