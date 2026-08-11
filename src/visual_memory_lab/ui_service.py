"""Application service for office-memory search and evidence inspection."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from visual_memory_lab.memory_store import MemoryStore, StoredMemory

SEARCH_K = 10
DISPLAY_K_VALUES = (3, 5, 10)


class QueryEncoder(Protocol):
    model_id: str
    model_revision: str

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...

    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray: ...


@dataclass(frozen=True)
class ZoneCatalog:
    zones: dict[str, dict[str, object]]
    assignments: dict[str, str]
    counts: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> ZoneCatalog:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read zone artifact {path}: {error}") from error
        zone_values = artifact.get("zones")
        assignments = artifact.get("assignments")
        counts = artifact.get("assignment_counts")
        if (
            not isinstance(zone_values, list)
            or not isinstance(assignments, dict)
            or not isinstance(counts, dict)
        ):
            raise ValueError("zone artifact is missing zones, assignments, or counts")
        zones = {
            str(zone["slug"]): zone
            for zone in zone_values
            if isinstance(zone, dict) and isinstance(zone.get("slug"), str)
        }
        if not zones:
            raise ValueError("zone artifact contains no usable zones")
        return cls(zones=zones, assignments=assignments, counts=counts)


@dataclass(frozen=True)
class EvaluationCatalog:
    metrics: dict[str, object]
    queries: list[dict[str, object]]
    query_by_id: dict[str, dict[str, object]]

    @classmethod
    def load(cls, root: Path) -> EvaluationCatalog:
        try:
            metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
            queries = [
                json.loads(line)
                for line in (root / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read evaluation artifact {root}: {error}") from error
        query_by_id = {str(row["query_id"]): row for row in queries}
        if not metrics or not queries or len(query_by_id) != len(queries):
            raise ValueError("evaluation artifact is empty or has duplicate query IDs")
        return cls(metrics=metrics, queries=queries, query_by_id=query_by_id)


def failure_tags(row: dict[str, object]) -> list[str]:
    tags: list[str] = []
    if not bool(row.get("strict_eligible")):
        tags.append("uncovered")
    if bool(row.get("strict_hit_at_1")):
        tags.append("strict-top1")
    elif bool(row.get("strict_hit_at_5")):
        tags.append("strict-rescued-at5")
    elif bool(row.get("strict_hit_at_10")):
        tags.append("strict-rescued-at10")
    elif bool(row.get("strict_eligible")):
        tags.append("strict-miss-at10")
    if bool(row.get("relaxed_hit_at_1")) and not bool(row.get("strict_hit_at_1")):
        tags.append("relaxed-only-top1")
    if float(row.get("top1_translation_error_m", 0.0)) > 0.5:
        tags.append("large-translation")
    if float(row.get("top1_rotation_error_deg", 0.0)) > 30.0:
        tags.append("large-rotation")
    return tags


class OfficeMemoryService:
    def __init__(
        self,
        *,
        memory: MemoryStore,
        queries: MemoryStore,
        encoder: QueryEncoder,
        zones: ZoneCatalog,
        evaluation: EvaluationCatalog,
    ) -> None:
        if (
            memory.model_id != queries.model_id
            or memory.model_revision != queries.model_revision
            or encoder.model_id != memory.model_id
            or encoder.model_revision != memory.model_revision
        ):
            raise ValueError("memory, query, and encoder model revisions must match")
        self.memory = memory
        self.queries = queries
        self.encoder = encoder
        self.zones = zones
        self.evaluation = evaluation

    def capabilities(self, *, analysis_available: bool) -> dict[str, object]:
        return {
            "dataset": "7-scenes-office",
            "memory_count": self.memory.count,
            "query_count": self.queries.count,
            "captured_at_available": False,
            "analysis_available": analysis_available,
            "analysis_requires_confirmation": True,
            "search_modes": ["text", "image"],
            "supported_question_families": [
                "location",
                "context",
                "revisit",
                "visible-state",
                "maintenance",
                "safety-evidence",
                "comparison",
                "object-recall",
            ],
            "unsupported_claims": [
                "calendar-time",
                "person-identification",
                "who-moved-an-object",
                "unseen-events",
            ],
        }

    def search_text(self, question: str, *, display_k: int) -> dict[str, object]:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        vector = self.encoder.encode_texts([cleaned])[0]
        return self._search_payload(
            vector,
            query={"kind": "text", "question": cleaned},
            display_k=display_k,
        )

    def search_image(self, image: Image.Image, *, display_k: int) -> dict[str, object]:
        vector = self.encoder.encode_pil_images([image])[0]
        return self._search_payload(
            vector,
            query={"kind": "image"},
            display_k=display_k,
        )

    def _search_payload(
        self,
        vector: np.ndarray,
        *,
        query: dict[str, object],
        display_k: int,
    ) -> dict[str, object]:
        if display_k not in DISPLAY_K_VALUES:
            raise ValueError("display_k must be 3, 5, or 10")
        results = self.memory.search(vector, top_k=SEARCH_K)
        evidence = [self._stored_memory_payload(result, "memory") for result in results]
        zone_slugs = [
            str(item["zone"]["slug"])
            for item in evidence
            if isinstance(item.get("zone"), dict)
        ]
        agreement = self._zone_agreement(zone_slugs, evidence)
        return {
            "query": query,
            "display_k": display_k,
            "temporal": {
                "captured_at": None,
                "message": "Calendar time is unavailable for this public dataset.",
            },
            "likely_area": agreement,
            "evidence": evidence,
        }

    def _zone_agreement(
        self,
        zone_slugs: list[str],
        evidence: list[dict[str, object]],
    ) -> dict[str, object] | None:
        if not zone_slugs:
            return None
        counts = Counter(zone_slugs)
        maximum = max(counts.values())
        winners = {slug for slug, count in counts.items() if count == maximum}
        winner = next(
            str(item["zone"]["slug"])
            for item in evidence
            if isinstance(item.get("zone"), dict)
            and str(item["zone"]["slug"]) in winners
        )
        unique_winner = len(winners) == 1
        if maximum >= 7:
            strength = "strong"
        elif maximum >= 4 and unique_winner:
            strength = "moderate"
        else:
            strength = "mixed"
        zone = self.zones.zones[winner]
        return {
            "slug": winner,
            "name": zone["name"],
            "support_count": maximum,
            "evidence_count": SEARCH_K,
            "strength": strength,
        }

    def _stored_memory_payload(
        self, result: StoredMemory, collection: str
    ) -> dict[str, object]:
        record = result.record
        observation_id = str(record["observation_id"])
        slug = self.zones.assignments.get(observation_id)
        zone = self.zones.zones.get(slug) if slug else None
        return {
            "rank": result.rank,
            "score": result.score,
            "observation_id": observation_id,
            "collection": collection,
            "sequence_id": record.get("sequence_id", record.get("episode_id")),
            "frame": record.get("step"),
            "captured_at": None,
            "zone": (
                {"slug": slug, "name": zone["name"]}
                if slug and zone
                else None
            ),
            "image_url": f"/api/images/{collection}/{observation_id}",
        }

    def zone_list(self) -> list[dict[str, object]]:
        return [
            {
                **zone,
                "assigned_frame_count": int(self.zones.counts.get(slug, 0)),
            }
            for slug, zone in self.zones.zones.items()
        ]

    def zone_detail(self, slug: str) -> dict[str, object]:
        try:
            zone = self.zones.zones[slug]
        except KeyError as error:
            raise KeyError(f"unknown zone: {slug}") from error
        representative_ids = [
            observation_id
            for observation_id, assigned_slug in self.zones.assignments.items()
            if assigned_slug == slug
        ][:12]
        return {
            **zone,
            "assigned_frame_count": int(self.zones.counts.get(slug, 0)),
            "memories": [
                {
                    "observation_id": observation_id,
                    "image_url": f"/api/images/memory/{observation_id}",
                }
                for observation_id in representative_ids
            ],
        }

    def query_page(
        self,
        *,
        offset: int,
        limit: int,
        sequence: str | None,
        tag: str | None,
    ) -> dict[str, object]:
        rows = self.evaluation.queries
        if sequence:
            rows = [row for row in rows if row.get("sequence_id") == sequence]
        if tag:
            rows = [row for row in rows if tag in failure_tags(row)]
        page = rows[offset : offset + limit]
        return {
            "offset": offset,
            "limit": limit,
            "total": len(rows),
            "items": [
                {
                    "query_id": row["query_id"],
                    "sequence_id": row["sequence_id"],
                    "frame": row["step"],
                    "image_url": f"/api/images/query/{row['query_id']}",
                    "top1_translation_error_m": row["top1_translation_error_m"],
                    "top1_rotation_error_deg": row["top1_rotation_error_deg"],
                    "tags": failure_tags(row),
                }
                for row in page
            ],
        }

    def query_detail(self, query_id: str) -> dict[str, object]:
        try:
            row = self.evaluation.query_by_id[query_id]
        except KeyError as error:
            raise KeyError(f"unknown evaluation query: {query_id}") from error
        return {
            **row,
            "tags": failure_tags(row),
            "query_image_url": f"/api/images/query/{query_id}",
            "retrievals": [
                {
                    **retrieval,
                    "image_url": (
                        f"/api/images/memory/{retrieval['observation_id']}"
                    ),
                }
                for retrieval in row["retrievals"]
            ],
        }
