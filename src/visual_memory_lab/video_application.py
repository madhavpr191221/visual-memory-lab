"""Application-facing timeline and evidence helpers for prepared videos."""

from __future__ import annotations

from collections import OrderedDict
import json
from typing import Iterable


def _windows_for_video(windows: Iterable[dict[str, object]], video_id: str) -> list[dict[str, object]]:
    return sorted(
        [item for item in windows if str(item.get("video_id")) == video_id],
        key=lambda item: (float(item.get("start_s", 0.0)), float(item.get("end_s", 0.0))),
    )


def context_interval(start_s: float, end_s: float, duration_s: float, padding_s: float = 2.0) -> dict[str, float]:
    """Return a bounded playback interval around an event."""
    before = max(0.0, float(start_s) - padding_s)
    after = min(float(duration_s), float(end_s) + padding_s)
    return {"start_s": before, "end_s": max(before, after)}


def video_catalog(windows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    catalog: OrderedDict[str, dict[str, object]] = OrderedDict()
    for item in sorted(windows, key=lambda row: str(row.get("video_id", ""))):
        video_id = str(item.get("video_id", ""))
        if video_id not in catalog:
            catalog[video_id] = {
                "video_id": video_id,
                "video_url": f"/api/video-memory/videos/{video_id}",
                "duration_s": max(float(item.get("duration_s", item.get("end_s", 0.0))), 0.0),
                "description": str(item.get("description", "")),
                "script": str(item.get("script", "")),
                "scene": str(item.get("scene", "")),
                "subject": str(item.get("subject", "")),
                "objects": list(item.get("objects", [])),
                "actions": [json.loads(value) for value in sorted({
                    json.dumps(action, sort_keys=True)
                    for action in item.get("actions", [])
                    if isinstance(action, dict) and str(action.get("name", "")).strip()
                })],
            }
        else:
            catalog[video_id]["duration_s"] = max(
                float(catalog[video_id]["duration_s"]), float(item.get("duration_s", item.get("end_s", 0.0)))
            )
            existing = {
                json.dumps(action, sort_keys=True)
                for action in catalog[video_id].get("actions", [])
                if isinstance(action, dict)
            }
            existing |= {
                json.dumps(action, sort_keys=True)
                for action in item.get("actions", [])
                if isinstance(action, dict) and str(action.get("name", "")).strip()
            }
            catalog[video_id]["actions"] = [json.loads(value) for value in sorted(existing)]
            for field in ("script", "scene", "subject"):
                if not catalog[video_id].get(field):
                    catalog[video_id][field] = str(item.get(field, ""))
            catalog[video_id]["objects"] = sorted(set(catalog[video_id].get("objects", [])) | set(item.get("objects", [])))
    for item in catalog.values():
        if item.get("actions") and isinstance(item["actions"][0], str):
            item["actions"] = [json.loads(value) for value in item["actions"]]
        actions = [action for action in item.get("actions", []) if isinstance(action, dict)]
        overlap_groups: list[list[str]] = []
        for action in actions:
            overlaps = [
                str(other.get("name", ""))
                for other in actions
                if other is not action
                and float(other.get("start_s", 0.0)) < float(action.get("end_s", 0.0))
                and float(other.get("end_s", 0.0)) > float(action.get("start_s", 0.0))
            ]
            if overlaps:
                group = sorted(set([str(action.get("name", "")), *overlaps]))
                if group not in overlap_groups:
                    overlap_groups.append(group)
        item["overlap_groups"] = overlap_groups
    return list(catalog.values())


def summarize_video(windows: Iterable[dict[str, object]], video_id: str) -> dict[str, object]:
    """Create an auditable timeline from official action intervals.

    This is the no-cloud fallback and the evaluation reference. It never claims
    more than the dataset's action labels establish.
    """

    selected = _windows_for_video(windows, video_id)
    if not selected:
        raise KeyError(video_id)
    events: dict[tuple[str, float, float], dict[str, object]] = {}
    for window in selected:
        for action in window.get("actions", []):
            if not isinstance(action, dict):
                continue
            name = str(action.get("name", "")).strip()
            start = float(action.get("start_s", 0.0))
            end = float(action.get("end_s", 0.0))
            if not name or end <= start:
                continue
            key = (name, start, end)
            events[key] = {
                "start_s": start,
                "end_s": end,
                "label": name,
                "evidence_window_id": str(window.get("window_id", "")),
                "confidence": "dataset annotation",
                "limitations": ["The action interval is an annotation reference, not a VLM judgment."],
            }
    ordered = sorted(events.values(), key=lambda item: (float(item["start_s"]), float(item["end_s"])))
    groups: list[dict[str, object]] = []
    for event in ordered:
        if groups and float(event["start_s"]) <= float(groups[-1]["end_s"]) + 0.5:
            group = groups[-1]
            labels = list(group["labels"])
            if str(event["label"]) not in labels:
                labels.append(str(event["label"]))
            group["labels"] = labels
            group["label"] = " · ".join(labels)
            group["end_s"] = max(float(group["end_s"]), float(event["end_s"]))
            group["source_events"].append(event)
        else:
            groups.append({
                "start_s": float(event["start_s"]),
                "end_s": float(event["end_s"]),
                "label": str(event["label"]),
                "labels": [str(event["label"])],
                "source_events": [event],
            })
    overview = str(selected[0].get("description", "")).split(";", 1)[0].strip()
    if overview.isupper():
        overview = overview.capitalize()
    return {
        "video_id": video_id,
        "video_url": f"/api/video-memory/videos/{video_id}",
        "overview": overview,
        "events": groups,
        "raw_events": ordered,
        "objects": list(selected[0].get("objects", [])),
        "source": "official_charades_annotations",
        "vlm_used": False,
    }


def video_finding_payload(*, video_id: str, question: str, start_s: float, end_s: float, answer: str,
                          evidence_window_ids: list[str], status: str, note: str = "",
                          limitations: list[str] | None = None) -> dict[str, object]:
    return {
        "video_id": video_id,
        "question": question,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "answer": answer,
        "evidence_window_ids": evidence_window_ids,
        "status": status,
        "note": note,
        "limitations": limitations or [],
        "source": "official_charades_annotations",
    }


def answer_follow_up(
    windows: Iterable[dict[str, object]], video_id: str, question: str, *, start_s: float, end_s: float
) -> dict[str, object]:
    selected = _windows_for_video(windows, video_id)
    if not selected:
        raise KeyError(video_id)
    actions: list[dict[str, object]] = []
    for window in selected:
        for action in window.get("actions", []):
            if not isinstance(action, dict):
                continue
            if float(action.get("end_s", 0.0)) > start_s and float(action.get("start_s", 0.0)) < end_s:
                if action not in actions:
                    actions.append(action)
    actions.sort(key=lambda item: float(item.get("start_s", 0.0)))
    if actions:
        answer = "The selected evidence contains: " + ", ".join(str(item.get("name")) for item in actions) + "."
        supported = True
    else:
        answer = "The selected evidence does not contain an annotated action that answers this question."
        supported = False
    return {
        "video_id": video_id,
        "question": question,
        "answer": answer,
        "supported": supported,
        "evidence_window_ids": [str(item.get("window_id", "")) for item in selected if float(item.get("end_s", 0.0)) > start_s and float(item.get("start_s", 0.0)) < end_s],
        "limitations": ["This answer is grounded in selected evidence and official action annotations; it is not a VLM judgment."],
        "source": "official_charades_annotations",
    }
