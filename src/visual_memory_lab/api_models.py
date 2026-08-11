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


class SearchResponse(ApiModel):
    query: SearchQuery
    display_k: Literal[3, 5, 10]
    temporal: TemporalCapability
    likely_area: LikelyArea | None
    evidence: list[EvidenceItem]


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
