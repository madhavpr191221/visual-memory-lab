"""FastAPI application for the local Office visual-memory explorer."""

from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
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
from visual_memory_lab.learned_video import LearnedVideoIndex, LearnedVideoRetriever
from visual_memory_lab.video_application import answer_follow_up, context_interval, summarize_video, video_catalog

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
    charades_learned_index: Path = Path("outputs/charades/learned/full/index")


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


def load_resources(config: AppConfig) -> AppResources:
    memory = NumpyMemoryStore.load(
        config.memory_index, verify_source=config.verify_source
    )
    queries = NumpyMemoryStore.load(
        config.query_index, verify_source=config.verify_source
    )
    encoder = ClipEncoder(device=config.device)
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
    charades_video = None
    try:
        if (config.charades_learned_index / "metadata.json").is_file():
            charades_video = LearnedVideoRetriever(
                LearnedVideoIndex.load(config.charades_learned_index),
                device=config.device,
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        charades_video = None
    return AppResources(
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
    )


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
    ) -> dict[str, object]:
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        if q.strip() and current().charades_video is not None:
            results = current().charades_video.search(q, top_k=top_k)
            retrieval_mode = "learned_temporal_clip"
        else:
            results = search_windows(windows, q, top_k=top_k) if q.strip() else []
            retrieval_mode = "annotation_lexical_baseline"
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
            "retrieval_mode": retrieval_mode,
            "results": results,
        }

    @app.get("/api/video-memory/catalog")
    def video_memory_catalog() -> dict[str, object]:
        windows = current().charades_windows
        if windows is None:
            raise HTTPException(status_code=404, detail="Charades temporal memory is not prepared")
        return {"dataset": "charades", "videos": video_catalog(windows)}

    @app.post("/api/video-memory/summarize")
    def video_memory_summary(request: VideoSummaryRequest) -> dict[str, object]:
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

    @app.post("/api/video-memory/findings")
    def create_video_finding(request: VideoFindingCreateRequest) -> dict[str, object]:
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
