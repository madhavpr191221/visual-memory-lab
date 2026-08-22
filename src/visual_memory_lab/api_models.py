"""Stable JSON contracts shared by the office-memory API and React client."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextSearchRequest(ApiModel):
    question: str = Field(min_length=1, max_length=500)
    display_k: Literal[3, 5, 10] = 5


class AnalysisRequest(ApiModel):
    question: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)


class SearchQuery(ApiModel):
    kind: Literal["text", "image"]
    question: str | None = None


class TemporalCapability(ApiModel):
    captured_at: None = None
    message: str


class ZoneSummary(ApiModel):
    slug: str
    name: str


class LikelyArea(ApiModel):
    slug: str
    name: str
    support_count: int
    evidence_count: int
    strength: Literal["strong", "moderate", "mixed"]


class EvidenceItem(ApiModel):
    rank: int
    score: float
    observation_id: str
    collection: Literal["memory", "query"]
    sequence_id: str | None
    frame: int | None
    captured_at: None = None
    zone: ZoneSummary | None
    image_url: str
    result_kind: str = ""


class SearchResponse(ApiModel):
    query: SearchQuery
    display_k: Literal[3, 5, 10]
    temporal: TemporalCapability
    likely_area: LikelyArea | None
    evidence: list[EvidenceItem]
    retrieval_mode: str = ""
    candidate_count: int = 0
    diversity_note: str = ""


class InspectionCreateRequest(ApiModel):
    title: str = Field(default="Office inspection", min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)


class InspectionCompareRequest(ApiModel):
    earlier_observation_id: str = Field(min_length=1)


class InspectionReportRequest(ApiModel):
    question: str = Field(min_length=1, max_length=500)
    earlier_observation_id: str = Field(min_length=1)


class InspectionSummaryRequest(ApiModel):
    summary: str = Field(min_length=1, max_length=2000)
    visible_objects: list[str] = Field(default_factory=list, max_length=30)
    visible_conditions: list[str] = Field(default_factory=list, max_length=30)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    model: str = ""
    cached: bool = False


class VideoSummaryRequest(ApiModel):
    video_id: str = Field(min_length=1, max_length=32)


class VideoUploadStatus(ApiModel):
    upload_id: str
    video_id: str
    status: str
    progress: float = 1.0
    duration_s: float | None = None
    error: str | None = None


class VideoFollowUpRequest(ApiModel):
    video_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=500)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)


class VideoSynthesisRequest(ApiModel):
    video_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=500)
    event_label: str = Field(min_length=1, max_length=240)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    evidence_window_ids: list[str] = Field(default_factory=list, max_length=32)
    mode: Literal["preview", "detailed"] = "preview"


class VideoObjectEvidenceRequest(ApiModel):
    video_id: str = Field(min_length=1, max_length=32)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    query: str = Field(min_length=1, max_length=500)
    event_label: str = Field(default="", max_length=240)
    object_prompts: list[str] = Field(default_factory=list, max_length=20)
    frame_timestamps_s: list[float] = Field(default_factory=list, max_length=32)


class VideoFindingCreateRequest(ApiModel):
    video_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=500)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    answer: str = Field(min_length=1, max_length=4000)
    evidence_window_ids: list[str] = Field(default_factory=list, max_length=32)
    status: Literal["confirmed", "unclear", "needs_manual_review", "rejected"] = "unclear"
    note: str = Field(default="", max_length=1000)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    source: str = Field(default="official_charades_annotations", max_length=120)


class EvidenceCitation(ApiModel):
    observation_id: str
    claim: str


class AnalysisResponse(ApiModel):
    question_type: Literal[
        "location",
        "context",
        "revisit",
        "visible-state",
        "maintenance",
        "safety-evidence",
        "comparison",
        "object-recall",
        "unsupported",
    ]
    supported: bool
    answer: str
    evidence_citations: list[EvidenceCitation]
    evidence_strength: Literal["low", "medium", "high"]
    limitations: list[str]
    model: str
    cached: bool
