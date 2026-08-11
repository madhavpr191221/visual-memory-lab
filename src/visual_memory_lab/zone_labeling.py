"""VLM-assisted semantic place-zone curation for 7-Scenes Office."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from visual_memory_lab.evaluation import pose_matrix, rotation_errors_deg

PROMPT_VERSION = "phase3-zones-v1"
DEFAULT_MODEL = "gpt-5.6-terra"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameObservation(StrictModel):
    observation_id: str
    suggested_zone: str
    stable_landmarks: list[str]
    confidence: Literal["low", "medium", "high"]


class DiscoveryBatch(StrictModel):
    frames: list[FrameObservation]


class ZonePrompts(StrictModel):
    name: str
    landmarks: str
    technician_question: str


class ZoneDefinition(StrictModel):
    slug: str
    name: str
    description: str
    stable_landmarks: list[str]
    prompts: ZonePrompts


class ZoneOntology(StrictModel):
    zones: list[ZoneDefinition] = Field(min_length=5, max_length=10)


class FrameAssignment(StrictModel):
    observation_id: str
    zone_slug: str
    visible_landmarks: list[str]
    confidence: Literal["low", "medium", "high"]


class VerificationBatch(StrictModel):
    assignments: list[FrameAssignment]


def _read_manifest(source: Path) -> tuple[dict[str, object], list[dict[str, object]], Path]:
    try:
        manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (source / "observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read source manifest {source}: {error}") from error
    if not isinstance(manifest, dict) or not records:
        raise ValueError("zone labeling requires a non-empty observation manifest")
    image_root_value = manifest.get("image_root", ".")
    if not isinstance(image_root_value, str):
        raise ValueError("manifest image_root must be a string")
    return manifest, records, (source / image_root_value).resolve()


def select_keyframes(
    records: list[dict[str, object]],
    *,
    translation_m: float = 0.5,
    rotation_deg: float = 30.0,
) -> list[dict[str, object]]:
    """Select deterministic pose-spaced frames within each sequence."""

    if translation_m <= 0.0 or rotation_deg <= 0.0:
        raise ValueError("keyframe thresholds must be positive")
    last_by_sequence: dict[str, np.ndarray] = {}
    selected: list[dict[str, object]] = []
    for record in records:
        sequence = str(record.get("sequence_id", record.get("episode_id", "")))
        current = pose_matrix(record)
        previous = last_by_sequence.get(sequence)
        if previous is None:
            selected.append(record)
            last_by_sequence[sequence] = current
            continue
        distance = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
        angle = float(rotation_errors_deg(previous[:3, :3], current[None, :3, :3])[0])
        if distance >= translation_m or angle >= rotation_deg:
            selected.append(record)
            last_by_sequence[sequence] = current
    return selected


def _image_part(path: Path) -> tuple[dict[str, str], str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode("ascii")
    return (
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{encoded}",
            "detail": "high",
        },
        digest,
    )


def _cache_key(
    *, model: str, schema: type[BaseModel], prompt: str, image_hashes: list[str]
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "schema": schema.model_json_schema(),
            "prompt": prompt,
            "image_hashes": image_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parsed_response(
    *,
    client: object,
    model: str,
    schema: type[SchemaT],
    prompt: str,
    image_paths: list[Path],
    cache_dir: Path,
) -> tuple[SchemaT, str]:
    image_parts: list[dict[str, str]] = []
    image_hashes: list[str] = []
    for path in image_paths:
        part, digest = _image_part(path)
        image_parts.append(part)
        image_hashes.append(digest)
    key = _cache_key(
        model=model,
        schema=schema,
        prompt=prompt,
        image_hashes=image_hashes,
    )
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return schema.model_validate(cached["parsed"]), str(cached["response_model"])

    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    content.extend(image_parts)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            responses = getattr(client, "responses")
            response = responses.parse(
                model=model,
                input=[{"role": "user", "content": content}],
                text_format=schema,
                store=False,
            )
            parsed = response.output_parsed
            if not isinstance(parsed, schema):
                raise ValueError("VLM response did not contain the required structured output")
            response_model = str(getattr(response, "model", model))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "cache_key": key,
                        "model_requested": model,
                        "response_model": response_model,
                        "prompt_version": PROMPT_VERSION,
                        "image_hashes": image_hashes,
                        "parsed": parsed.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return parsed, response_model
        except Exception as error:  # SDK errors vary by transport/status.
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"VLM labeling request failed after three attempts: {last_error}") from last_error


def _batched[T](items: list[T], size: int) -> list[list[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _require_exact_ids(expected: list[str], actual: list[str], stage: str) -> None:
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f"{stage} response IDs do not exactly match the requested keyframes")


def propagate_assignments(
    records: list[dict[str, object]],
    representatives: list[dict[str, object]],
    representative_labels: dict[str, str],
    *,
    distance_m: float = 0.5,
    angle_deg: float = 45.0,
) -> dict[str, str]:
    """Assign frames from their nearest verified representative in pose space."""

    rep_poses = np.stack([pose_matrix(record) for record in representatives])
    labels = [representative_labels.get(str(record["observation_id"]), "unassigned") for record in representatives]
    assignments: dict[str, str] = {}
    for record in records:
        current = pose_matrix(record)
        distances = np.linalg.norm(rep_poses[:, :3, 3] - current[:3, 3], axis=1)
        angles = rotation_errors_deg(current[:3, :3], rep_poses[:, :3, :3])
        qualifies = (distances <= distance_m) & (angles <= angle_deg)
        candidate_indices = [
            index for index in np.flatnonzero(qualifies) if labels[int(index)] != "unassigned"
        ]
        if not candidate_indices:
            assignments[str(record["observation_id"])] = "unassigned"
            continue
        best = min(
            candidate_indices,
            key=lambda index: (distances[index] / distance_m) ** 2
            + (angles[index] / angle_deg) ** 2,
        )
        assignments[str(record["observation_id"])] = labels[int(best)]
    return assignments


def label_zones(
    *,
    source: Path,
    output: Path,
    cache_dir: Path,
    model: str = DEFAULT_MODEL,
    client: object | None = None,
) -> dict[str, object]:
    """Discover, verify, propagate, and freeze Office place-zone labels."""

    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"zone artifact already exists: {output}")
    manifest, records, image_root = _read_manifest(source)
    if manifest.get("split") != "train":
        raise ValueError("place zones must be curated from the training split only")
    representatives = select_keyframes(records)
    if len(representatives) < 5:
        raise ValueError("too few pose-spaced keyframes to curate place zones")

    if client is None:
        load_dotenv()
        from openai import OpenAI

        client = OpenAI()

    discoveries: list[FrameObservation] = []
    response_models: set[str] = set()
    for batch in _batched(representatives, 6):
        ids = [str(record["observation_id"]) for record in batch]
        prompt = (
            "You are curating stable place zones in one cluttered office. The following "
            f"images correspond in order to these observation IDs: {ids}. For every image, "
            "identify durable landmarks and suggest a concise physical-area name. Ignore "
            "temporary details such as people or movable papers. Return exactly one frame "
            "record per supplied ID."
        )
        parsed, response_model = _parsed_response(
            client=client,
            model=model,
            schema=DiscoveryBatch,
            prompt=prompt,
            image_paths=[image_root / str(record["image_path"]) for record in batch],
            cache_dir=cache_dir,
        )
        _require_exact_ids(ids, [frame.observation_id for frame in parsed.frames], "discovery")
        discoveries.extend(parsed.frames)
        response_models.add(response_model)

    discovery_payload = [item.model_dump(mode="json") for item in discoveries]
    ontology_prompt = (
        "Consolidate these frame-level office observations into 5 to 10 mutually useful "
        "place zones. Zones should describe physical areas a maintenance technician could "
        "recognize, use lowercase hyphenated slugs, and rely on stable landmarks. Create "
        "three natural CLIP text prompts for each zone: its name, a landmark description, "
        "and a technician question beginning with 'Where'. Observations:\n"
        + json.dumps(discovery_payload, separators=(",", ":"))
    )
    ontology, response_model = _parsed_response(
        client=client,
        model=model,
        schema=ZoneOntology,
        prompt=ontology_prompt,
        image_paths=[],
        cache_dir=cache_dir,
    )
    response_models.add(response_model)
    slugs = [zone.slug for zone in ontology.zones]
    if len(slugs) != len(set(slugs)) or any(
        not slug or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in slug)
        for slug in slugs
    ):
        raise ValueError("VLM ontology returned invalid or duplicate zone slugs")

    ontology_text = json.dumps(
        [zone.model_dump(mode="json") for zone in ontology.zones], separators=(",", ":")
    )
    representative_labels: dict[str, str] = {}
    representative_evidence: dict[str, dict[str, object]] = {}
    for batch in _batched(representatives, 6):
        ids = [str(record["observation_id"]) for record in batch]
        prompt = (
            "Assign every supplied office image to exactly one zone from the frozen ontology "
            "or use the literal slug 'unassigned' when evidence is weak. Low-confidence "
            "assignments must be unassigned. Return exactly one assignment per ID. "
            f"IDs in image order: {ids}. Ontology: {ontology_text}"
        )
        parsed, response_model = _parsed_response(
            client=client,
            model=model,
            schema=VerificationBatch,
            prompt=prompt,
            image_paths=[image_root / str(record["image_path"]) for record in batch],
            cache_dir=cache_dir,
        )
        _require_exact_ids(
            ids,
            [assignment.observation_id for assignment in parsed.assignments],
            "verification",
        )
        response_models.add(response_model)
        for assignment in parsed.assignments:
            slug = assignment.zone_slug
            if assignment.confidence == "low":
                slug = "unassigned"
            if slug != "unassigned" and slug not in slugs:
                raise ValueError(f"verification returned unknown zone slug: {slug}")
            representative_labels[assignment.observation_id] = slug
            representative_evidence[assignment.observation_id] = {
                "zone_slug": slug,
                "visible_landmarks": assignment.visible_landmarks,
                "confidence": assignment.confidence,
            }

    assignments = propagate_assignments(records, representatives, representative_labels)
    used_slugs = {slug for slug in assignments.values() if slug != "unassigned"}
    zones = [zone.model_dump(mode="json") for zone in ontology.zones if zone.slug in used_slugs]
    if not zones:
        raise ValueError("VLM verification produced no usable place zones")
    counts: dict[str, int] = defaultdict(int)
    for slug in assignments.values():
        counts[slug] += 1
    source_digest = hashlib.sha256()
    source_digest.update((source / "run.json").read_bytes())
    source_digest.update((source / "observations.jsonl").read_bytes())
    artifact: dict[str, object] = {
        "schema_version": "1.0",
        "dataset_id": manifest.get("dataset_id"),
        "source": {
            "split": manifest.get("split"),
            "manifest_sha256": source_digest.hexdigest(),
        },
        "method": {
            "kind": "vlm_assisted_silver_labels",
            "model_requested": model,
            "response_models": sorted(response_models),
            "prompt_version": PROMPT_VERSION,
            "keyframe_translation_m": 0.5,
            "keyframe_rotation_deg": 30.0,
            "propagation_translation_m": 0.5,
            "propagation_rotation_deg": 45.0,
            "representative_count": len(representatives),
        },
        "zones": zones,
        "assignments": assignments,
        "assignment_counts": dict(sorted(counts.items())),
        "representative_evidence": representative_evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
