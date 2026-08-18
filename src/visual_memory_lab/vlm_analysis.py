"""Explicit, evidence-grounded VLM analysis for selected public Office frames."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

PROMPT_VERSION = "phase4-evidence-analysis-v1"
VIDEO_PROMPT_VERSION = "video-event-synthesis-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceCitation(StrictModel):
    observation_id: str
    claim: str


class GroundedJudgment(StrictModel):
    question_type: Literal[
        "location", "context", "revisit", "visible-state", "maintenance",
        "safety-evidence", "comparison", "object-recall", "unsupported",
    ]
    supported: bool
    answer: str
    evidence_citations: list[EvidenceCitation]
    evidence_strength: Literal["low", "medium", "high"]
    limitations: list[str]


class VisualSummary(StrictModel):
    summary: str
    visible_objects: list[str]
    visible_conditions: list[str]
    limitations: list[str]


class InspectionReport(StrictModel):
    status: Literal["observed", "possible_difference", "insufficient_evidence", "manual_review_required"]
    summary: str
    visible_objects: list[str]
    visible_conditions: list[str]
    comparison_observations: list[str]
    supporting_evidence: list[EvidenceCitation]
    limitations: list[str]
    recommended_manual_check: str


class VideoGroundedAnswer(StrictModel):
    answer: str
    event_label: str
    supported: bool
    confidence: Literal["low", "medium", "high"]
    evidence_citations: list[EvidenceCitation]
    limitations: list[str]
    visible_evidence: str = ""
    visual_evidence_supported: bool | None = None
    visual_support_status: Literal["supported", "partially_visible", "unclear", "not_visibly_confirmed"] = "unclear"


def _image_part(image: Image.Image) -> tuple[dict[str, str], str]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    data = buffer.getvalue()
    return (
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}",
            "detail": "high",
        },
        hashlib.sha256(data).hexdigest(),
    )


class EvidenceAnalyzer:
    """Call the VLM only after an explicit analysis request."""

    def __init__(self, *, model: str, cache_dir: Path, client: object | None = None) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self._client = client

    def _client_value(self) -> object:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def analyze(
        self,
        *,
        question: str,
        evidence: list[tuple[str, Path]],
        query_image: Image.Image | None = None,
    ) -> dict[str, object]:
        if not 1 <= len(evidence) <= 5:
            raise ValueError("select between one and five evidence frames")
        evidence_ids = [item[0] for item in evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence IDs must be unique")

        prompt = (
            "You are reviewing retrieved frames from the public 7-Scenes Office dataset. "
            "Answer only from the supplied images. Separate what is visibly supported from "
            "inference. Do not claim calendar time, a person's identity, who moved an object, "
            "or any event outside the visible frames. If the evidence cannot answer the "
            "question, set supported=false and explain the limitation. Cite only the exact "
            "observation IDs supplied below.\n\n"
            f"Question: {question.strip()}\n"
            f"Evidence image order: {', '.join(evidence_ids)}"
        )
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        image_hashes: list[str] = []
        if query_image is not None:
            query_part, query_hash = _image_part(query_image)
            content.append({"type": "input_text", "text": "Reference query image:"})
            content.append(query_part)
            image_hashes.append(query_hash)
        for observation_id, path in evidence:
            with Image.open(path) as image:
                part, digest = _image_part(image)
            content.append({"type": "input_text", "text": f"Evidence {observation_id}:"})
            content.append(part)
            image_hashes.append(digest)

        cache_path = self._cache_path(prompt, image_hashes) if query_image is None else None
        if cache_path is not None and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = GroundedJudgment.model_validate(cached["parsed"])
            self._validate_citations(parsed, evidence_ids)
            return {**parsed.model_dump(mode="json"), "model": cached["model"], "cached": True}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                responses = getattr(self._client_value(), "responses")
                response = responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                    text_format=GroundedJudgment,
                    store=False,
                )
                parsed = response.output_parsed
                if not isinstance(parsed, GroundedJudgment):
                    raise ValueError("analysis response did not match the required schema")
                self._validate_citations(parsed, evidence_ids)
                response_model = str(getattr(response, "model", self.model))
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(
                            {"model": response_model, "parsed": parsed.model_dump(mode="json")},
                            indent=2,
                            sort_keys=True,
                        ) + "\n",
                        encoding="utf-8",
                    )
                return {**parsed.model_dump(mode="json"), "model": response_model, "cached": False}
            except Exception as error:  # SDK exception types depend on transport/status.
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise RuntimeError(f"evidence analysis failed after three attempts: {last_error}") from last_error

    def summarize_image(self, image: Image.Image) -> dict[str, object]:
        part, _ = _image_part(image)
        content = [
            {"type": "input_text", "text": (
                "Describe this current office inspection photo in plain language. "
                "List only visible objects and visible maintenance-relevant conditions. "
                "Do not infer identity, history, ownership, or events. Mention blur, occlusion, "
                "and missing coverage as limitations."
            )},
            part,
        ]
        parsed = self._parse(content, VisualSummary)
        return {**parsed.model_dump(mode="json"), "model": self.model, "cached": False}

    def synthesize_video(
        self,
        *,
        question: str,
        video_id: str,
        event_label: str,
        start_s: float,
        end_s: float,
        frames: list[tuple[str, float, Image.Image]],
        actions: list[str],
        objects: list[str],
        mode: Literal["preview", "detailed"] = "preview",
    ) -> dict[str, object]:
        """Write a grounded answer from timestamped RGB evidence frames."""
        if not frames:
            raise ValueError("at least one video evidence frame is required")
        evidence_ids = [item[0] for item in frames]
        prompt = (
            "You are reviewing sampled RGB frames from one public Charades recording. "
            "Answer the user's question only from the supplied frames and metadata. "
            "Use the event interval as an annotation/model reference, not as proof of an "
            "unseen action. Do not invent identity, intent, or events outside the interval. "
            "If the frames do not support the answer, set supported=false. Cite only the "
            "supplied evidence IDs. Return `visible_evidence` as one plain-language sentence "
            "describing only what can be seen. Set `visual_support_status` to supported, "
            "partially_visible, unclear, or not_visibly_confirmed. Use partially_visible when "
            "the action/object appears in only some sampled frames. Keep the legacy boolean "
            "`visual_evidence_supported` consistent with supported or partially_visible. "
            f"Mode: {mode}. Video: {video_id}. Question: {question.strip()}\n"
            f"Candidate event: {event_label} ({start_s:.2f}-{end_s:.2f} seconds).\n"
            f"Annotated actions: {', '.join(actions) or 'none supplied'}.\n"
            f"Associated objects: {', '.join(objects) or 'none supplied'}.\n"
            f"Evidence IDs: {', '.join(evidence_ids)}"
        )
        content: list[dict[str, object]] = [{"type": "input_text", "text": prompt}]
        image_hashes: list[str] = []
        for evidence_id, timestamp, image in frames:
            part, digest = _image_part(image)
            content.extend([
                {"type": "input_text", "text": f"Evidence {evidence_id} at {timestamp:.2f} seconds:"},
                part,
            ])
            image_hashes.append(digest)
        cache_path = self.cache_dir / "video" / f"{hashlib.sha256(json.dumps({'model': self.model, 'prompt_version': VIDEO_PROMPT_VERSION, 'prompt': prompt, 'images': image_hashes}, sort_keys=True).encode()).hexdigest()}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = VideoGroundedAnswer.model_validate(cached["parsed"])
            self._validate_video_citations(parsed, evidence_ids)
            return {**parsed.model_dump(mode="json"), "model": cached["model"], "cached": True, "source": "vlm_video_synthesis"}
        parsed = self._parse(content, VideoGroundedAnswer)
        if not isinstance(parsed, VideoGroundedAnswer):
            raise ValueError("video synthesis response did not match the required schema")
        self._validate_video_citations(parsed, evidence_ids)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"model": self.model, "parsed": parsed.model_dump(mode="json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**parsed.model_dump(mode="json"), "model": self.model, "cached": False, "source": "vlm_video_synthesis"}

    def report(
        self,
        *,
        question: str,
        evidence: list[tuple[str, Path]],
        query_image: Image.Image | None = None,
    ) -> dict[str, object]:
        if not 1 <= len(evidence) <= 5:
            raise ValueError("select between one and five evidence frames")
        evidence_ids = [item[0] for item in evidence]
        prompt = (
            "You are preparing a cautious office maintenance inspection report. "
            "Use only the supplied current photo and earlier evidence. Separate visible facts "
            "from inference. Do not claim calendar time, persistent object identity, a person's "
            "identity, or confirmed movement. Cite only the supplied evidence IDs. If evidence "
            "is incomplete, use insufficient_evidence or manual_review_required.\n\n"
            f"Maintenance question: {question.strip()}\n"
            f"Evidence IDs: {', '.join(evidence_ids)}"
        )
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        allowed_ids = list(evidence_ids)
        if query_image is not None:
            query_part, _ = _image_part(query_image)
            content.extend([{"type": "input_text", "text": "Current uploaded photo (cite as current-upload if needed):"}, query_part])
            allowed_ids.insert(0, "current-upload")
        for observation_id, path in evidence:
            with Image.open(path) as image:
                part, _ = _image_part(image)
            content.extend([{"type": "input_text", "text": f"Earlier evidence {observation_id}:"}, part])
        parsed = self._parse(content, InspectionReport)
        unknown = sorted({item.observation_id for item in parsed.supporting_evidence} - set(allowed_ids))
        if unknown:
            raise ValueError(f"report cited unknown evidence: {', '.join(unknown)}")
        if parsed.status == "observed" and not parsed.supporting_evidence:
            raise ValueError("an observed report must cite evidence")
        return {**parsed.model_dump(mode="json"), "model": self.model, "cached": False}

    def _parse(self, content: list[dict[str, str]], schema: type[StrictModel]) -> StrictModel:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                responses = getattr(self._client_value(), "responses")
                response = responses.parse(model=self.model, input=[{"role": "user", "content": content}], text_format=schema, store=False)
                parsed = response.output_parsed
                if not isinstance(parsed, schema):
                    raise ValueError("structured response did not match the required schema")
                return parsed
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise RuntimeError(f"structured analysis failed after three attempts: {last_error}") from last_error

    def _cache_path(self, prompt: str, image_hashes: list[str]) -> Path:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "schema": GroundedJudgment.model_json_schema(),
                "prompt": prompt,
                "image_hashes": image_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"

    @staticmethod
    def _validate_citations(parsed: GroundedJudgment, evidence_ids: list[str]) -> None:
        cited = [citation.observation_id for citation in parsed.evidence_citations]
        unknown = sorted(set(cited) - set(evidence_ids))
        if unknown:
            raise ValueError(f"analysis cited unknown evidence: {', '.join(unknown)}")
        if parsed.supported and not cited:
            raise ValueError("a supported answer must cite at least one supplied observation")

    @staticmethod
    def _validate_video_citations(parsed: VideoGroundedAnswer, evidence_ids: list[str]) -> None:
        cited = [citation.observation_id for citation in parsed.evidence_citations]
        unknown = sorted(set(cited) - set(evidence_ids))
        if unknown:
            raise ValueError(f"video synthesis cited unknown evidence: {', '.join(unknown)}")
        if parsed.supported and not cited:
            raise ValueError("a supported video answer must cite supplied evidence")
