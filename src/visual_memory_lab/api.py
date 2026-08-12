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
from visual_memory_lab.technician_benchmark import load_questions

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
    technician_questions: Path = Path("data/phase7/technician_questions.jsonl")
    technician_output: Path = Path("outputs/phase7/technician-benchmark")


@dataclass
class AppResources:
    service: OfficeMemoryService
    memory: MemoryStore
    queries: MemoryStore
    analysis: object | None = None
    objects: ObjectShowcase | None = None
    rgbd: RgbdShowcase | None = None
    associations: AssociationShowcase | None = None


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
    return AppResources(
        service=service,
        memory=memory,
        queries=queries,
        analysis=analysis,
        objects=objects,
        rgbd=rgbd,
        associations=associations,
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
