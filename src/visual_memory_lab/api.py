"""FastAPI application for the local Office visual-memory explorer."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from visual_memory_lab import __version__
from visual_memory_lab.object_showcase import ObjectShowcase
from visual_memory_lab.association_showcase import AssociationShowcase
from visual_memory_lab.rgbd_showcase import RgbdShowcase
from visual_memory_lab.api_models import (
    AnalysisRequest,
    AnalysisResponse,
    InspectionCompareRequest,
    InspectionCreateRequest,
    InspectionReportRequest,
    InspectionSummaryRequest,
    VideoFollowUpRequest,
    VideoObjectEvidenceRequest,
    VideoSynthesisRequest,
    VideoFindingCreateRequest,
    VideoSummaryRequest,
    SearchResponse,
    TextSearchRequest,
)
from visual_memory_lab.encoder import ClipEncoder
from visual_memory_lab.memory_store import MemoryStore, NumpyMemoryStore
from visual_memory_lab.ui_service import (
    EvaluationCatalog,
    OfficeMemoryService,
    QueryEncoder,
    ZoneCatalog,
)
from visual_memory_lab.vlm_analysis import EvidenceAnalyzer
from visual_memory_lab.inspection_store import InspectionStore
from visual_memory_lab.technician_benchmark import load_questions
from visual_memory_lab.charades import load_manifest, search_windows
from visual_memory_lab.learned_video import LearnedVideoIndex, LearnedVideoRetriever, VideoActionResolver, group_video_events
from visual_memory_lab.video_application import answer_follow_up, context_interval, summarize_video, video_catalog
from visual_memory_lab.video_object_evidence import VideoObjectEvidence
from visual_memory_lab.local_video import LocalVideoManager

logger = logging.getLogger("visual_memory_lab")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}


@dataclass(frozen=True)
class AppConfig:
    memory_index: Path = Path("outputs/phase3/train-index")
    query_index: Path = Path("outputs/phase3/test-index")
    zones: Path = Path("artifacts/phase3/office-zones.json")
    evaluation: Path = Path("outputs/phase3/evaluation")
    web_dist: Path = Path("web/dist")
    device: str = "auto"
    verify_source: bool = False
    analysis_model: str = "gpt-5.6-terra"
    analysis_cache: Path = Path("outputs/phase4/vlm-cache")
    object_localization: Path = Path("outputs/phase6b1/object-localization")
    object_audit: Path = Path("outputs/phase6b1/vlm-audit")
    rgbd_evidence: Path = Path("outputs/phase612/rgbd-evidence")
    associations: Path = Path("outputs/phase613/associations")
    association_audit: Path = Path("outputs/phase613/vlm-audit")
    inspection_db: Path = Path("outputs/phase8/inspections.sqlite3")
    technician_questions: Path = Path("data/phase7/technician_questions.jsonl")
    technician_output: Path = Path("outputs/phase7/technician-benchmark")
    charades_windows: Path = Path("outputs/charades/learned/windows/windows.jsonl")
    # The application index covers every prepared recording. Evaluation keeps
    # using a separate train-only index and held-out test manifest.
    charades_learned_index: Path = Path("outputs/charades/learned/application/index")
    local_video_root: Path = Path("outputs/local-video-sessions")


@dataclass
class AppResources:
    service: OfficeMemoryService
    memory: MemoryStore
    queries: MemoryStore
    analysis: object | None = None
    objects: ObjectShowcase | None = None
    rgbd: RgbdShowcase | None = None
    associations: AssociationShowcase | None = None
    inspections: InspectionStore | None = None
    charades_windows: list[dict[str, object]] | None = None
    charades_video: LearnedVideoRetriever | None = None
    video_action_resolver: VideoActionResolver | None = None
    video_objects: VideoObjectEvidence | None = None
    local_videos: LocalVideoManager | None = None


def load_resources(config: AppConfig) -> AppResources:
    logger.info("Loading prepared memory artifacts")
    memory = NumpyMemoryStore.load(
        config.memory_index, verify_source=config.verify_source
    )
    queries = NumpyMemoryStore.load(
        config.query_index, verify_source=config.verify_source
    )
    logger.info("Loading CLIP model (requested_device=%s)", config.device)
    encoder = ClipEncoder(device=config.device)
    logger.info("CLIP model ready (device=%s)", encoder.device)
    service = OfficeMemoryService(
        memory=memory,
        queries=queries,
        encoder=encoder,
        zones=ZoneCatalog.load(config.zones),
        evaluation=EvaluationCatalog.load(config.evaluation),
    )
    analysis = None
    if os.getenv("OPENAI_API_KEY"):
        analysis = EvidenceAnalyzer(
            model=config.analysis_model,
            cache_dir=config.analysis_cache,
        )
    associations = None
    try:
        associations = AssociationShowcase.load(
            associations=config.associations,
            localization=config.object_localization,
            audit=config.association_audit,
        )
    except FileNotFoundError:
        pass
    objects = None
    try:
        objects = ObjectShowcase.load(
            localization=config.object_localization,
            audit=config.object_audit,
        )
    except FileNotFoundError:
        pass
    rgbd = None
    try:
        rgbd = RgbdShowcase.load(
            evidence=config.rgbd_evidence,
            localization=config.object_localization,
        )
    except FileNotFoundError:
        pass
    charades_windows = None
    if config.charades_windows.is_file():
        charades_windows = load_manifest(config.charades_windows)
        # Older prepared window artifacts did not carry all recording-level
        # provenance fields. Enrich them from the sibling manifest at load
        # time so the UI remains trustworthy without rewriting the MP4 index.
        source_manifest = config.charades_windows.parent.parent / "manifest.jsonl"
        if source_manifest.is_file():
            by_video = {str(row.get("video_id")): row for row in load_manifest(source_manifest)}
            for window in charades_windows:
                source = by_video.get(str(window.get("video_id")), {})
                for field in ("script", "scene", "subject", "description", "objects", "duration_s"):
                    if field not in window or not window.get(field):
                        if field in source:
                            window[field] = source[field]
    charades_video = None
    video_action_resolver = None
    if os.getenv("OPENAI_API_KEY"):
        video_action_resolver = VideoActionResolver(model=config.analysis_model, cache_dir=config.analysis_cache / "video-action-resolver")
    try:
        if (config.charades_learned_index / "metadata.json").is_file():
            charades_video = LearnedVideoRetriever(
                LearnedVideoIndex.load(config.charades_learned_index),
                device=config.device,
                action_resolver=video_action_resolver,
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        charades_video = None
    resources = AppResources(
        service=service,
        memory=memory,
        queries=queries,
        analysis=analysis,
        objects=objects,
        rgbd=rgbd,
        associations=associations,
        inspections=InspectionStore(config.inspection_db),
        charades_windows=charades_windows,
        charades_video=charades_video,
        video_action_resolver=video_action_resolver,
        video_objects=VideoObjectEvidence(device=config.device),
        local_videos=LocalVideoManager(config.local_video_root, service.encoder),
    )
    logger.info("Application resources ready (charades=%s, local_video_device=%s)", charades_windows is not None, encoder.device)
    return resources


async def _read_upload(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="upload must be a PNG or JPEG image")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload must not exceed 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise HTTPException(status_code=422, detail="upload is not a readable image") from error


def _sample_video_frames(path: Path, start_s: float, end_s: float, count: int = 6) -> list[tuple[str, float, Image.Image]]:
    """Decode a small, deterministic RGB evidence set from an MP4."""
    import av

    if not path.is_file() or end_s <= start_s:
        raise ValueError("video evidence interval is unavailable")
    timestamps = [start_s + (index + 0.5) * (end_s - start_s) / count for index in range(count)]
    output: list[tuple[str, float, Image.Image]] = []
    target_index = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame_time = float(frame.time or 0.0)
            while target_index < len(timestamps) and frame_time >= timestamps[target_index]:
                timestamp = timestamps[target_index]
                output.append((f"frame-{target_index:02d}-{timestamp:.3f}", timestamp, frame.to_image().convert("RGB")))
                target_index += 1
            if target_index >= len(timestamps):
                break
    if not output:
        raise ValueError("could not decode video evidence frames")
    while len(output) < len(timestamps):
        index = len(output)
        timestamp = timestamps[index]
        output.append((f"frame-{index:02d}-{timestamp:.3f}", timestamp, output[-1][2].copy()))
    return output


def create_app(
    config: AppConfig | None = None,
    *,
    resources: AppResources | None = None,
) -> FastAPI:
    resolved_config = config or AppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        load_dotenv()
        app.state.resources = resources or load_resources(resolved_config)
        yield
        app.state.resources = None

    app = FastAPI(
        title="Visual Memory Lab API",
        version=__version__,
        lifespan=lifespan,
    )

    def current() -> AppResources:
        value = getattr(app.state, "resources", None)
        if not isinstance(value, AppResources):
            raise HTTPException(status_code=503, detail="visual memory is not loaded")
        return value

    @app.get("/api/health")
    def health() -> dict[str, str]:
        current()
        return {"status": "ready"}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, object]:
        return current().service.capabilities(
            analysis_available=current().analysis is not None
        )

    @app.get("/api/memory")
    def memory_summary() -> dict[str, object]:
        return {
            "memory_count": len(current().memory),
            "query_count": len(current().queries),
            "model_id": current().memory.model_id,
            "model_revision": current().memory.model_revision,
        }

    @app.get("/api/video-memory")
    def video_memory(
        q: str = Query(default=""),
        top_k: int = Query(default=8, ge=1, le=24),
        video_id: str | None = Query(default=None, min_length=1, max_length=32),
    ) -> dict[str, object]:
        if video_id and current().local_videos and current().local_videos.get(video_id):
            if not q.strip():
                return {"dataset": "local", "window_count": 0, "catalog_window_count": 0, "indexed_window_count": 0, "query": q, "video_id": video_id, "retrieval_mode": "local_clip_window", "support_status": "unsupported", "matched_actions": [], "message": "Ask a question about this local recording.", "results": []}
            local_results = current().local_videos.search(video_id, q, top_k=top_k)
            for item in local_results:
                item["video_url"] = f"/api/video-memory/videos/{video_id}"
                item["evidence_start_s"] = item["context_start_s"]
                item["evidence_end_s"] = item["context_end_s"]
            return {
                "dataset": "local",
                "window_count": len(local_results),
                "catalog_window_count": len(local_results),
                "indexed_window_count": len(local_results),
                "query": q,
                "video_id": video_id,
                "retrieval_mode": "local_clip_window_grouped",
                "support_status": "visual_candidate",
                "matched_actions": [],
                "message": "Retrieved visually similar candidate moments. This private video has no ground-truth action labels.",
                "results": local_results,
            }
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        if q.strip() and current().charades_video is not None:
            results, support = current().charades_video.search_with_metadata(q, top_k=top_k, video_id=video_id)
            retrieval_mode = "learned_temporal_clip_action_guard"
            # The learned index is built from the training split, while the
            # catalogue intentionally exposes held-out recordings as well.
            # In that case the retriever has no indexed rows from which to
            # derive the action vocabulary. Use the prepared annotation windows
            # for that recording instead of reporting that it has no actions.
            if not results and video_id:
                catalog_matches = [
                    item for item in windows
                    if str(item.get("video_id", "")) == video_id
                ]
                fallback = search_windows(catalog_matches, q, top_k=top_k)
                if fallback:
                    results = fallback
                    support = {
                        "status": "supported",
                        "matched_actions": [],
                        "reason": "learned vectors are unavailable for this recording; matching annotated windows are shown",
                        "fallback": True,
                    }
                    retrieval_mode = "annotation_fallback_unindexed_recording"
        else:
            eligible = (
                [item for item in windows if str(item.get("video_id")) == video_id]
                if video_id
                else windows
            )
            results = search_windows(eligible, q, top_k=top_k) if q.strip() else []
            support = {"status": "supported" if results else "unsupported", "matched_actions": [], "reason": "lexical fallback"}
            retrieval_mode = "annotation_lexical_baseline"
        results = group_video_events(results, top_k=min(top_k, 3))
        matched_actions = [str(value) for value in support.get("matched_actions", []) if str(value).strip()]
        for result in results:
            actions = [
                action for action in result.get("actions", [])
                if isinstance(action, dict) and str(action.get("name", "")).strip()
            ]
            matched = [action for action in actions if str(action.get("name")) in matched_actions]
            primary = matched[0] if matched else (actions[0] if actions else None)
            primary_name = str(primary.get("name")) if primary else (matched_actions[0] if matched_actions else "Relevant event")
            context_actions = [
                str(action.get("name")) for action in actions
                if str(action.get("name")) != primary_name
            ]
            result["primary_action"] = primary_name
            result["context_actions"] = list(dict.fromkeys(context_actions))
            result["recorded_action"] = {
                "label": primary_name,
                "start_s": float(primary.get("start_s", result.get("start_s", 0.0))) if primary else float(result.get("start_s", 0.0)),
                "end_s": float(primary.get("end_s", result.get("end_s", 0.0))) if primary else float(result.get("end_s", 0.0)),
                "source_window_ids": list(result.get("evidence_window_ids", [])),
                "note": "The recorded action comes from the dataset annotation; it is not independent visual proof.",
            }
            # Keep three intervals distinct: the exact annotation, the
            # retrieved/index window, and the padded playback context.
            result["annotation_start_s"] = float(result["recorded_action"]["start_s"])
            result["annotation_end_s"] = float(result["recorded_action"]["end_s"])
            refined_start = result.get("refined_start_s", result.get("predicted_start_s"))
            refined_end = result.get("refined_end_s", result.get("predicted_end_s"))
            has_refined_interval = refined_start is not None and refined_end is not None
            result["action_start_s"] = float(refined_start) if has_refined_interval else result["annotation_start_s"]
            result["action_end_s"] = float(refined_end) if has_refined_interval else result["annotation_end_s"]
            result["interval_source"] = "temporal_refinement" if has_refined_interval else "dataset_annotation"
            if result.get("frame_timestamps_s"):
                result["frame_timestamps_s"] = [float(value) for value in result["frame_timestamps_s"]]
            if result.get("refinement_confidence") is not None:
                result["refinement_confidence"] = float(result["refinement_confidence"])
            result["evidence_start_s"] = float(result.get("context_start_s", result.get("start_s", 0.0)))
            result["evidence_end_s"] = float(result.get("context_end_s", result.get("end_s", 0.0)))
            duration_s = max(
                float(item.get("duration_s", item.get("end_s", 0.0)))
                for item in windows
                if str(item.get("video_id", "")) == str(result.get("video_id", ""))
            )
            # A few Charades annotations extend a fraction beyond the encoded
            # MP4 duration. Never display or seek past playable evidence.
            result["recorded_action"]["start_s"] = max(
                0.0, min(float(result["recorded_action"]["start_s"]), duration_s)
            )
            result["recorded_action"]["end_s"] = max(
                result["recorded_action"]["start_s"],
                min(float(result["recorded_action"]["end_s"]), duration_s),
            )
            if not has_refined_interval:
                result["action_start_s"] = result["recorded_action"]["start_s"]
                result["action_end_s"] = result["recorded_action"]["end_s"]
            result["action_start_s"] = max(0.0, min(float(result["action_start_s"]), duration_s))
            result["action_end_s"] = max(
                result["action_start_s"], min(float(result["action_end_s"]), duration_s)
            )
            context = context_interval(
                result["action_start_s"], result["action_end_s"], duration_s, padding_s=2.0
            )
            result["context_start_s"] = context["start_s"]
            result["context_end_s"] = context["end_s"]
            result["result_limitations"] = [
                "The action label is an annotation reference, not proof that every detail is visually identifiable.",
                "The sampled frames may be too small, blurred, or occluded to verify the object or action fully.",
            ]
        for result in results:
            result["video_url"] = f"/api/video-memory/videos/{result['video_id']}"
        return {
            "dataset": "charades",
            "window_count": len(windows),
            "catalog_window_count": len(windows),
            "indexed_window_count": (
                len(current().charades_video.index.records)
                if current().charades_video is not None
                else 0
            ),
            "query": q,
            "video_id": video_id,
            "retrieval_mode": retrieval_mode,
            "support_status": support.get("status", "unsupported"),
            "matched_actions": matched_actions,
            "message": (
                "The question matched an action available in this recording."
                if support.get("status") == "supported"
                else str(support.get("reason", "No supported event was found in this recording."))
            ),
            "results": results,
        }

    @app.get("/api/video-memory/frame/{video_id}")
    def video_memory_frame(video_id: str, timestamp_s: float = Query(..., ge=0.0)) -> Response:
        """Return one decoded evidence frame without writing a derived file."""
        local = current().local_videos.get(video_id) if current().local_videos else None
        windows = current().charades_windows
        if local is not None:
            matches = local.records
            path = local.path
            duration_s = local.duration_s
        else:
            matches = [item for item in (windows or []) if str(item.get("video_id")) == video_id]
            if not matches:
                raise HTTPException(status_code=404, detail="video not found")
            path = Path(str(matches[0].get("video_path", ""))).resolve()
            duration_s = max(float(item.get("duration_s", item.get("end_s", 0.0))) for item in matches)
        if windows is None and local is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        timestamp = min(timestamp_s, max(duration_s - 1e-3, 0.0))
        try:
            frame = _sample_video_frames(path, timestamp, min(duration_s, timestamp + 0.01), count=1)[0][2]
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=88)
            frame.close()
            return Response(content=buffer.getvalue(), media_type="image/jpeg")
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail="could not decode evidence frame") from error

    @app.get("/api/video-memory/catalog")
    def video_memory_catalog() -> dict[str, object]:
        windows = current().charades_windows
        if windows is None:
            catalog = []
        else:
            catalog = video_catalog(windows)
        if current().local_videos:
            catalog.extend(current().local_videos.catalog())
        if not catalog:
            raise HTTPException(status_code=404, detail="No video recordings are prepared")
        return {"dataset": "charades+local", "videos": catalog}

    @app.post("/api/video-memory/uploads")
    async def video_memory_upload(video: UploadFile = File(...)) -> dict[str, object]:
        manager = current().local_videos
        if manager is None:
            raise HTTPException(status_code=503, detail="local video import is unavailable")
        if video.content_type not in {"video/mp4", "video/quicktime", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="upload must be an MP4 video")
        upload_id = f"upload-{secrets.token_hex(8)}"
        device = str(manager.encoder.device)
        manager.create_job(upload_id, device=device)
        logger.info("Local upload received: name=%s device=%s", video.filename or "video.mp4", device)
        session_dir = manager.root / f"incoming-{secrets.token_hex(8)}"
        session_dir.mkdir(parents=True, exist_ok=False)
        source = session_dir / "upload.mp4"
        total = 0
        try:
            with source.open("wb") as handle:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)
                    if total > 500 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="video must not exceed 500 MB")
                    handle.write(chunk)
                    manager.update_job(upload_id, progress=min(0.10, total / (500 * 1024 * 1024) * 0.10), message="Uploading your video locally…")
            manager.start_import(source, original_name=video.filename or "video.mp4", upload_id=upload_id)
            return {"upload_id": upload_id, "video_id": None, "status": "processing", "progress": 0.10, "stage": "decoding", "device": device, "message": "Upload complete. Checking the video locally…"}
        except HTTPException:
            source.unlink(missing_ok=True); session_dir.rmdir()
            manager.update_job(upload_id, status="failed", stage="failed", progress=1.0, error="Upload rejected.", message="The video could not be uploaded.")
            raise
        except Exception as error:
            source.unlink(missing_ok=True); session_dir.rmdir()
            manager.update_job(upload_id, status="failed", stage="failed", progress=1.0, error=str(error), message="The video could not be uploaded.")
            raise HTTPException(status_code=422, detail=f"could not process video: {error}") from error

    @app.get("/api/video-memory/uploads/{upload_id}")
    def video_memory_upload_status(upload_id: str) -> dict[str, object]:
        manager = current().local_videos
        status = manager.job(upload_id) if manager is not None else None
        if status is None:
            raise HTTPException(status_code=404, detail="upload job not found")
        return status

    @app.post("/api/video-memory/summarize")
    def video_memory_summary(request: VideoSummaryRequest) -> dict[str, object]:
        local = current().local_videos.get(request.video_id) if current().local_videos else None
        if local is not None:
            return {
                "video_id": local.video_id,
                "video_url": f"/api/video-memory/videos/{local.video_id}",
                "overview": "A locally uploaded recording. Ask a question to retrieve visually similar moments.",
                "events": [],
                "raw_events": [],
                "objects": [],
                "source": "local_upload",
                "vlm_used": False,
                "limitations": ["No dataset action labels or verified event intervals are available for this upload."],
            }
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        try:
            return summarize_video(windows, request.video_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="video not found") from error

    @app.post("/api/video-memory/follow-up")
    def video_memory_follow_up(request: VideoFollowUpRequest) -> dict[str, object]:
        if request.end_s <= request.start_s:
            raise HTTPException(status_code=422, detail="end_s must be greater than start_s")
        local = current().local_videos.get(request.video_id) if current().local_videos else None
        if local is not None:
            interval = context_interval(request.start_s, request.end_s, local.duration_s, padding_s=0.0)
            evidence_ids = [
                str(item.get("window_id", ""))
                for item in local.records
                if float(item.get("end_s", 0.0)) > interval["start_s"]
                and float(item.get("start_s", 0.0)) < interval["end_s"]
            ]
            return {
                "video_id": request.video_id,
                "question": request.question,
                "answer": "This local recording has no verified action labels. Review the selected playback and object evidence before drawing a conclusion.",
                "supported": False,
                "evidence_window_ids": evidence_ids,
                "limitations": ["The local path retrieves visual candidates with frozen CLIP; it does not establish event identity or exact boundaries."],
                "source": "local_visual_retrieval",
            }
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        try:
            matching = [item for item in windows if str(item.get("video_id")) == request.video_id]
            if not matching:
                raise KeyError(request.video_id)
            duration_s = max(float(item.get("end_s", 0.0)) for item in matching)
            interval = context_interval(request.start_s, request.end_s, duration_s, padding_s=0.0)
            return answer_follow_up(
                windows,
                request.video_id,
                request.question,
                start_s=interval["start_s"],
                end_s=interval["end_s"],
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="video not found") from error

    @app.post("/api/video-memory/synthesize")
    def video_memory_synthesize(request: VideoSynthesisRequest) -> dict[str, object]:
        local = current().local_videos.get(request.video_id) if current().local_videos else None
        if local is not None:
            evidence = [
                item for item in local.records
                if str(item.get("window_id")) in request.evidence_window_ids
            ]
            if not evidence:
                evidence = [
                    item for item in local.records
                    if float(item.get("end_s", 0.0)) > request.start_s
                    and float(item.get("start_s", 0.0)) < request.end_s
                ]
            if not evidence:
                raise HTTPException(status_code=422, detail="no local evidence windows overlap the selected event")
            return {
                "video_id": request.video_id,
                "event_label": request.event_label,
                "start_s": request.start_s,
                "end_s": request.end_s,
                "answer": "The selected interval is a visually retrieved candidate from a private local video. No annotation or model judgment verifies the named event.",
                "supported": False,
                "confidence": "low",
                "evidence_window_ids": [str(item.get("window_id", "")) for item in evidence],
                "evidence_citations": [{"observation_id": str(item.get("window_id", "")), "claim": "Retrieved visual candidate window."} for item in evidence],
                "limitations": ["This local recording has no ground-truth action labels.", "Candidate timestamps come from CLIP similarity and are not verified event boundaries."],
                "cached": False,
                "model": None,
                "source": "local_visual_retrieval",
                "visible_evidence": "Inspect the playable interval and optional object overlays directly.",
                "visual_evidence_supported": None,
                "visual_support_status": "candidate_only",
            }
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        matching = [item for item in windows if str(item.get("video_id")) == request.video_id]
        if not matching:
            raise HTTPException(status_code=404, detail="video not found")
        evidence = [item for item in matching if str(item.get("window_id")) in request.evidence_window_ids]
        if not evidence:
            evidence = [item for item in matching if float(item.get("end_s", 0.0)) > request.start_s and float(item.get("start_s", 0.0)) < request.end_s]
        if not evidence:
            raise HTTPException(status_code=422, detail="no evidence windows overlap the selected event")
        actions = sorted({str(action.get("name", "")) for item in evidence for action in item.get("actions", []) if isinstance(action, dict) and action.get("name")})
        objects = sorted({str(value) for item in evidence for value in item.get("objects", [])})
        fallback = answer_follow_up(windows, request.video_id, request.question, start_s=request.start_s, end_s=request.end_s)
        fallback.update({
            "event_label": request.event_label,
            "confidence": "medium" if fallback["supported"] else "low",
            "evidence_citations": [{"observation_id": item, "claim": "Overlapping annotated evidence window."} for item in fallback["evidence_window_ids"]],
            "cached": False,
            "model": None,
            "source": "annotation_fallback",
            "visible_evidence": "The selected frames were not analyzed by the visual model.",
            "visual_evidence_supported": None,
            "visual_support_status": "not_visibly_confirmed",
        })
        analyzer = current().analysis
        if analyzer is None:
            return fallback
        try:
            path = Path(str(evidence[0].get("video_path", ""))).resolve()
            frames = _sample_video_frames(path, request.start_s, request.end_s)
            result = analyzer.synthesize_video(
                question=request.question,
                video_id=request.video_id,
                event_label=request.event_label,
                start_s=request.start_s,
                end_s=request.end_s,
                frames=frames,
                actions=actions,
                objects=objects,
                mode=request.mode,
            )
            result.update({"video_id": request.video_id, "event_label": request.event_label, "start_s": request.start_s, "end_s": request.end_s})
            return result
        except Exception:
            return fallback

    @app.post("/api/video-memory/object-evidence")
    def video_memory_object_evidence(request: VideoObjectEvidenceRequest) -> dict[str, object]:
        """Inspect objects only in the selected event's RGB evidence frames."""
        local = current().local_videos.get(request.video_id) if current().local_videos else None
        windows = local.records if local is not None else current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        matching = [item for item in windows if str(item.get("video_id")) == request.video_id]
        if not matching:
            raise HTTPException(status_code=404, detail="video not found")
        duration_s = max(float(item.get("duration_s", item.get("end_s", 0.0))) for item in matching)
        if request.end_s <= request.start_s:
            raise HTTPException(status_code=422, detail="end_s must be greater than start_s")
        end_s = min(request.end_s, duration_s)
        start_s = min(max(request.start_s, 0.0), end_s)
        frame_count = min(max(len(request.frame_timestamps_s), 8), 24)
        try:
            frames = _sample_video_frames(
                Path(str(matching[0].get("video_path", ""))).resolve(),
                start_s,
                end_s,
                count=frame_count,
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail="could not decode object evidence frames") from error
        prompts = [str(value).strip(" \t\r\n.,;:!?()[]{}\"'").lower() for value in request.object_prompts]
        query_terms = {
            token.strip(".,?!:;()[]{}").lower()
            for token in request.query.split()
            if len(token.strip(".,?!:;()[]{}")) >= 4
        }
        prompts.extend(
            str(value)
            for item in matching
            for value in str(item.get("objects", "")).split()
            if any(term in str(value).lower() or str(value).lower() in request.query.lower() for term in query_terms)
        )
        stopwords = {
            "when", "what", "where", "which", "does", "did", "someone", "person",
            "people", "they", "them", "this", "that", "from", "into", "with", "about",
            "show", "find", "happen", "happened", "there", "their", "have", "holding",
            "opening", "closing", "taking", "putting", "some", "the", "and", "near",
            "relevant", "event", "action", "actions", "visible", "visibility", "eating",
            "eat", "ate", "someone", "anything", "something",
        }
        prompts.extend(
            token.strip(".,?!:;()[]{}").lower()
            for token in request.query.split()
            if len(token.strip(".,?!:;()[]{}")) >= 4
            and token.strip(".,?!:;()[]{}").lower() not in stopwords
        )
        if current().video_objects is None:
            raise HTTPException(status_code=503, detail="object inspection is unavailable")
        try:
            result = current().video_objects.inspect(frames, object_prompts=prompts)
        finally:
            for _, _, image in frames:
                image.close()
        prompts = sorted({value for value in prompts if value})
        return {
            "video_id": request.video_id,
            "event_label": request.event_label,
            "query": request.query,
            "start_s": start_s,
            "end_s": end_s,
            "target_objects": sorted(set(prompts)),
            "prompt_terms": sorted(set(prompts)),
            **result,
        }

    @app.post("/api/video-memory/findings")
    def create_video_finding(request: VideoFindingCreateRequest) -> dict[str, object]:
        local = current().local_videos.get(request.video_id) if current().local_videos else None
        if local is not None:
            if request.end_s <= request.start_s:
                raise HTTPException(status_code=422, detail="end_s must be greater than start_s")
            if current().inspections is None:
                raise HTTPException(status_code=503, detail="finding storage is unavailable")
            payload = request.model_dump(mode="json")
            payload["source"] = "local_visual_retrieval"
            payload["limitations"] = list(payload.get("limitations", [])) + ["Local video has no verified annotations; saved finding records review state only."]
            return current().inspections.create_video_finding(payload)
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        if request.end_s <= request.start_s:
            raise HTTPException(status_code=422, detail="end_s must be greater than start_s")
        if not any(str(item.get("video_id")) == request.video_id for item in windows):
            raise HTTPException(status_code=404, detail="video not found")
        if current().inspections is None:
            raise HTTPException(status_code=503, detail="finding storage is unavailable")
        return current().inspections.create_video_finding(request.model_dump(mode="json"))

    @app.get("/api/video-memory/findings")
    def list_video_findings() -> list[dict[str, object]]:
        if current().inspections is None:
            return []
        return current().inspections.list_video_findings()

    @app.get("/api/video-memory/findings/{finding_id}")
    def video_finding_detail(finding_id: str) -> dict[str, object]:
        try:
            if current().inspections is None:
                raise KeyError(finding_id)
            return current().inspections.get_video_finding(finding_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="video finding not found") from error

    @app.get("/api/video-memory/videos/{video_id}")
    def video_memory_file(video_id: str) -> FileResponse:
        local = current().local_videos.get(video_id) if current().local_videos else None
        if local is not None:
            return FileResponse(local.path, media_type="video/mp4", filename=local.path.name)
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        matches = [item for item in windows if str(item.get("video_id")) == video_id]
        if not matches:
            raise HTTPException(status_code=404, detail="video not found")
        path = Path(str(matches[0].get("video_path", ""))).resolve()
        if path.suffix.lower() != ".mp4" or not path.is_file():
            raise HTTPException(status_code=404, detail="video file is unavailable")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/guided-demo")
    def guided_demo() -> dict[str, object]:
        """Return a deterministic, evidence-backed hiring showcase case."""
        result = current().service.search_text(
            "workstation beside a window", display_k=5
        )
        evidence = result.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) < 2:
            raise HTTPException(
                status_code=404,
                detail="guided demo requires at least two office evidence frames",
            )
        return {
            "case_id": "window-side-workstation",
            "title": "Is this the workstation beside the window?",
            "question": "Where is the workstation beside a window?",
            "current": evidence[0],
            "earlier": evidence[1],
            "supporting_evidence": evidence[:5],
            "outcome": "The retrieved views point to a window-side workstation.",
            "explanation": "The strongest views show a desk with two monitors directly beside a bright window. The images support the area description, but they do not establish a persistent object identity or calendar date.",
            "manual_check": "Confirm the workstation, monitor power, cable routing, and desk stability on site.",
            "limitations": [
                "The public recordings provide logical sequence order, not calendar time.",
                "Visual similarity is not proof that two views show the same physical workstation.",
            ],
        }

    @app.get("/api/inspections")
    def inspections() -> list[dict[str, object]]:
        return current().inspections.list() if current().inspections else []

    @app.get("/api/inspections/{inspection_id}")
    def inspection_detail(inspection_id: str) -> dict[str, object]:
        try:
            return current().inspections.get(inspection_id)  # type: ignore[union-attr]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="inspection not found") from error

    @app.post("/api/inspections")
    def create_inspection(request: InspectionCreateRequest) -> dict[str, object]:
        if len(request.evidence_ids) != len(set(request.evidence_ids)):
            raise HTTPException(status_code=422, detail="evidence IDs must be unique")
        evidence: list[dict[str, object]] = []
        for rank, observation_id in enumerate(request.evidence_ids, start=1):
            try:
                record = next((item for item in current().memory.records() if item.get("observation_id") == observation_id), None)
                if record is None:
                    raise KeyError(observation_id)
            except (KeyError, ValueError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            evidence.append({"observation_id": observation_id, "collection": "memory", "rank": rank, "score": None, "role": "supporting"})
        result = current().inspections.create(  # type: ignore[union-attr]
            title=request.title,
            question=request.question,
            result_text=("Inspection saved with selected visual evidence." if evidence else "Inspection saved; select evidence before drawing a conclusion."),
            status=("supported_with_limits" if evidence else "insufficient_evidence"),
            limitations=["Saved evidence does not establish persistent object identity or movement."],
            current_image_path=None,
            evidence=evidence,
        )
        return result

    @app.post("/api/inspections/with-image")
    async def create_inspection_with_image(
        title: Annotated[str, Form()] = "Office inspection",
        question: Annotated[str, Form()] = "Where was this office area seen before?",
        evidence_ids: Annotated[str, Form()] = "[]",
        image: UploadFile = File(...),
    ) -> dict[str, object]:
        try:
            parsed_ids = json.loads(evidence_ids)
            request = InspectionCreateRequest(title=title, question=question, evidence_ids=parsed_ids)
        except (json.JSONDecodeError, ValidationError) as error:
            raise HTTPException(status_code=422, detail="invalid inspection form") from error
        decoded = await _read_upload(image)
        try:
            result = create_inspection(request)
            upload_root = (resolved_config.inspection_db.parent / "uploads").resolve()
            upload_root.mkdir(parents=True, exist_ok=True)
            image_path = upload_root / f"{result['id']}.jpg"
            decoded.save(image_path, format="JPEG", quality=92)
            return current().inspections.set_current_image(str(result["id"]), str(image_path))  # type: ignore[union-attr]
        finally:
            decoded.close()

    @app.get("/api/inspections/{inspection_id}/current-image")
    def inspection_current_image(inspection_id: str) -> FileResponse:
        try:
            inspection = current().inspections.get(inspection_id)  # type: ignore[union-attr]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="inspection not found") from error
        path = inspection.get("current_image_path")
        if not path or not Path(str(path)).is_file():
            raise HTTPException(status_code=404, detail="inspection has no current image")
        return FileResponse(Path(str(path)))

    @app.post("/api/inspections/{inspection_id}/compare")
    def compare_inspection(inspection_id: str, request: InspectionCompareRequest) -> dict[str, object]:
        try:
            inspection = current().inspections.get(inspection_id)  # type: ignore[union-attr]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="inspection not found") from error
        try:
            earlier = next((item for item in current().memory.records() if item.get("observation_id") == request.earlier_observation_id), None)
            if earlier is None:
                raise KeyError(request.earlier_observation_id)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        current_evidence = inspection.get("evidence", [])
        current_path = inspection.get("current_image_path")
        current_item = next((item for item in current().memory.records() if item.get("observation_id") == (current_evidence[0].get("observation_id") if current_evidence else None)), None)
        current_side = {
            "image_url": f"/api/inspections/{inspection_id}/current-image" if current_path else (f"/api/images/memory/{current_item['observation_id']}" if current_item else None),
            "observation_id": current_item.get("observation_id") if current_item else None,
            "sequence_id": current_item.get("sequence_id", current_item.get("episode_id")) if current_item else None,
            "frame": current_item.get("frame") if current_item else None,
            "zone": current_item.get("zone") if current_item else None,
            "label": "Current view",
        }
        earlier_side = {
            "image_url": f"/api/images/memory/{request.earlier_observation_id}",
            "observation_id": request.earlier_observation_id,
            "sequence_id": earlier.get("sequence_id", earlier.get("episode_id")),
            "frame": earlier.get("frame"),
            "zone": earlier.get("zone"),
            "label": "Earlier view",
        }
        limitations = ["The comparison does not establish persistent object identity or prove movement."]
        result_text = "The two views are ready for side-by-side inspection. Any difference may be caused by viewpoint, visibility, or a real scene change."
        if not current_side["image_url"]:
            limitations.append("No current image or selected current memory was available.")
        updated = current().inspections.update_comparison(inspection_id, earlier_observation_id=request.earlier_observation_id, result_text=result_text, status="manual_review_required", limitations=limitations)  # type: ignore[union-attr]
        earlier_observation = earlier.get("observation", earlier.get("observation_id"))
        association_matches = []
        if current().associations is not None:
            association_matches = [
                pair for pair in current().associations.payload.get("pairs", [])
                if isinstance(pair, dict) and str(pair.get("earlier_observation")) == str(earlier_observation)
            ]
        rgbd_matches = []
        if current().rgbd is not None:
            rgbd_matches = [
                item for item in current().rgbd.payload.get("comparisons", [])
                if isinstance(item, dict) and str(item.get("earlier_observation")) == str(earlier_observation)
            ]
        return {
            "inspection_id": inspection_id,
            "current": current_side,
            "earlier": earlier_side,
            "current_evidence": current_evidence,
            "updated_inspection": updated,
            "status": "manual_review_required",
            "explanation": result_text,
            "limitations": limitations,
            "supporting_artifacts": {
                "association_candidates": association_matches[:10],
                "rgbd_candidates": rgbd_matches[:10],
                "note": "These artifacts are supporting candidates only; they do not establish identity or movement.",
            },
        }

    @app.post("/api/inspection-summary/image")
    async def inspection_summary_image(image: UploadFile = File(...)) -> dict[str, object]:
        decoded = await _read_upload(image)
        try:
            try:
                return analyzer().summarize_image(decoded)  # type: ignore[no-any-return]
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            except RuntimeError as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            decoded.close()

    @app.post("/api/inspections/{inspection_id}/summary")
    def save_inspection_summary(inspection_id: str, request: InspectionSummaryRequest) -> dict[str, object]:
        try:
            current().inspections.get(inspection_id)  # type: ignore[union-attr]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="inspection not found") from error
        summary = request.model_dump(mode="json")
        return current().inspections.set_summary(inspection_id, summary)  # type: ignore[union-attr]

    @app.post("/api/inspections/{inspection_id}/report")
    def inspection_report(inspection_id: str, request: InspectionReportRequest) -> dict[str, object]:
        try:
            inspection = current().inspections.get(inspection_id)  # type: ignore[union-attr]
            earlier = next((item for item in current().memory.records() if item.get("observation_id") == request.earlier_observation_id), None)
            if earlier is None:
                raise KeyError(request.earlier_observation_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="inspection or earlier observation not found") from error
        evidence_ids = [request.earlier_observation_id]
        for item in inspection.get("evidence", []):
            candidate = str(item.get("observation_id"))
            if candidate not in evidence_ids:
                evidence_ids.append(candidate)
            if len(evidence_ids) == 5:
                break
        try:
            evidence = selected_evidence(evidence_ids)
            current_image = None
            current_path = inspection.get("current_image_path")
            if current_path and Path(str(current_path)).is_file():
                current_image = Image.open(Path(str(current_path))).convert("RGB")
            try:
                report = analyzer().report(question=request.question, evidence=evidence, query_image=current_image)  # type: ignore[union-attr]
            finally:
                if current_image is not None:
                    current_image.close()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        updated = current().inspections.set_report(  # type: ignore[union-attr]
            inspection_id,
            report,
            result_text=str(report["summary"]),
            status=str(report["status"]),
            limitations=[str(item) for item in report.get("limitations", [])],
        )
        return {**report, "inspection": updated}

    @app.get("/api/technician-benchmark")
    def technician_benchmark() -> dict[str, object]:
        try:
            questions = load_questions(resolved_config.technician_questions)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        summary_path = resolved_config.technician_output / "summary.json"
        summary: dict[str, object] | None = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = None
        return {
            "question_count": len(questions),
            "questions": [
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "category": item.category,
                    "dataset": item.dataset,
                    "source_observation_id": item.source_observation_id,
                    "answerability": item.answerability,
                    "expected_zone": item.expected_zone,
                    "expected_visit": item.expected_visit,
                    "expected_object_class": item.expected_object_class,
                    "expected_artifact": item.expected_artifact,
                    "rationale": item.rationale,
                }
                for item in questions
            ],
            "summary": summary,
        }

    @app.post("/api/search/text", response_model=SearchResponse)
    def search_text(request: TextSearchRequest) -> dict[str, object]:
        try:
            return current().service.search_text(
                request.question, display_k=request.display_k
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/search/image", response_model=SearchResponse)
    async def search_image(
        image: Annotated[UploadFile, File()],
        display_k: Annotated[int, Form()] = 5,
    ) -> dict[str, object]:
        decoded = await _read_upload(image)
        try:
            try:
                return current().service.search_image(decoded, display_k=display_k)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            decoded.close()

    def selected_evidence(ids: list[str]) -> list[tuple[str, Path]]:
        if len(ids) != len(set(ids)):
            raise HTTPException(status_code=422, detail="evidence IDs must be unique")
        selected: list[tuple[str, Path]] = []
        for observation_id in ids:
            try:
                selected.append((observation_id, current().memory.image_path(observation_id)))
            except (KeyError, FileNotFoundError, ValueError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return selected

    def analyzer() -> EvidenceAnalyzer:
        value = current().analysis
        if value is None or not hasattr(value, "analyze"):
            raise HTTPException(
                status_code=503,
                detail="cloud analysis is unavailable; configure OPENAI_API_KEY",
            )
        return value  # type: ignore[return-value]

    @app.post("/api/analyze/text", response_model=AnalysisResponse)
    def analyze_text(request: AnalysisRequest) -> dict[str, object]:
        try:
            return analyzer().analyze(
                question=request.question,
                evidence=selected_evidence(request.evidence_ids),
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/api/analyze/image", response_model=AnalysisResponse)
    async def analyze_image(
        image: Annotated[UploadFile, File()],
        question: Annotated[str, Form()],
        evidence_ids: Annotated[str, Form()],
    ) -> dict[str, object]:
        try:
            parsed_ids = json.loads(evidence_ids)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=422, detail="evidence_ids must be a JSON list") from error
        if not isinstance(parsed_ids, list) or not all(isinstance(item, str) for item in parsed_ids):
            raise HTTPException(status_code=422, detail="evidence_ids must be a JSON list of strings")
        try:
            request = AnalysisRequest(question=question, evidence_ids=parsed_ids)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="invalid analysis request") from error
        decoded = await _read_upload(image)
        try:
            try:
                return analyzer().analyze(
                    question=request.question,
                    evidence=selected_evidence(request.evidence_ids),
                    query_image=decoded,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            except RuntimeError as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
        finally:
            decoded.close()

    @app.get("/api/images/{collection}/{observation_id}")
    def observation_image(
        collection: Literal["memory", "query"], observation_id: str
    ) -> FileResponse:
        store = current().memory if collection == "memory" else current().queries
        try:
            path = store.image_path(observation_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return FileResponse(path)

    @app.get("/api/zones")
    def zones() -> list[dict[str, object]]:
        return current().service.zone_list()

    @app.get("/api/zones/{slug}")
    def zone_detail(slug: str) -> dict[str, object]:
        try:
            return current().service.zone_detail(slug)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/memory/evaluation")
    def evaluation() -> dict[str, object]:
        return current().service.evaluation.metrics

    @app.get("/api/objects")
    def objects() -> dict[str, object]:
        objects = current().objects
        if objects is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Object artifacts are unavailable; run localize-eth-objects first"
                ),
            )
        return objects.payload

    @app.get("/api/objects/images/{image_id}")
    def object_image(image_id: str) -> FileResponse:
        objects = current().objects
        if objects is None:
            raise HTTPException(status_code=404, detail="Object artifacts are unavailable")
        try:
            path = objects.image_path(image_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, content_disposition_type="inline")

    @app.get("/api/evidence")
    def evidence() -> dict[str, object]:
        evidence = current().rgbd
        if evidence is None:
            raise HTTPException(status_code=404, detail="RGB-D evidence is unavailable")
        return evidence.payload

    @app.get("/api/associations")
    def associations() -> dict[str, object]:
        associations = current().associations
        if associations is None:
            raise HTTPException(status_code=404, detail="Association artifacts are unavailable")
        return associations.payload

    @app.get("/api/queries")
    def queries(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 24,
        sequence: str | None = None,
        tag: str | None = None,
    ) -> dict[str, object]:
        return current().service.query_page(
            offset=offset,
            limit=limit,
            sequence=sequence,
            tag=tag,
        )

    @app.get("/api/queries/{query_id}")
    def query_detail(query_id: str) -> dict[str, object]:
        try:
            return current().service.query_detail(query_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    dist = resolved_config.web_dist.resolve()
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    if (dist / "index.html").is_file():
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file():
                try:
                    candidate.relative_to(dist)
                except ValueError:
                    pass
                else:
                    return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
