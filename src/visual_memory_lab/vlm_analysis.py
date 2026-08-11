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
