"""API contract and security tests for the Office memory explorer."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from visual_memory_lab.api import AppConfig, AppResources, create_app
from visual_memory_lab.inspection_store import InspectionStore

from tests.test_ui_service import make_service
from tests.test_object_showcase import make_object_showcase


def _client(tmp_path: Path) -> TestClient:
    service = make_service(tmp_path)
    resources = AppResources(
        service=service,
        memory=service.memory,
        queries=service.queries,
    )
    return TestClient(
        create_app(AppConfig(web_dist=tmp_path / "missing-dist"), resources=resources)
    )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_capabilities_and_text_search(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/health").json() == {"status": "ready"}
        capabilities = client.get("/api/capabilities").json()
        assert capabilities["memory_count"] == 10
        assert capabilities["captured_at_available"] is False
        assert capabilities["analysis_available"] is False

        response = client.post(
            "/api/search/text",
            json={"question": "desk beside a window", "display_k": 5},
        )
        assert response.status_code == 200
        assert response.json()["likely_area"]["strength"] == "strong"
        assert "C:\\" not in response.text
        assert client.get("/api/evaluation").status_code == 404
        assert client.get("/api/phase6a").status_code == 404


def test_guided_demo_returns_evidence_backed_case(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/guided-demo")
        assert response.status_code == 200
        payload = response.json()
        assert payload["case_id"] == "window-side-workstation"
        assert payload["current"]["image_url"].startswith("/api/images/memory/")
        assert payload["earlier"]["observation_id"] != payload["current"]["observation_id"]
        assert len(payload["supporting_evidence"]) >= 2
        assert payload["limitations"]


def test_video_memory_search_returns_timestamped_windows(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    resources = AppResources(
        service=service,
        memory=service.memory,
        queries=service.queries,
        inspections=InspectionStore(tmp_path / "findings.sqlite3"),
        charades_windows=[
            {
                "window_id": "ABC12:0-4",
                "video_id": "ABC12",
                "video_path": str(tmp_path / "ABC12.mp4"),
                "split": "train",
                "start_s": 0.0,
                "end_s": 4.0,
                "actions": [{"action_id": "c008", "name": "Opening a door", "start_s": 1.0, "end_s": 2.0}],
                "objects": ["door"],
                "description": "A person opens a door.",
            }
        ],
    )
    with TestClient(create_app(AppConfig(web_dist=tmp_path / "missing"), resources=resources)) as client:
        response = client.get("/api/video-memory", params={"q": "open door"})
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["video_url"].endswith("/ABC12")
        assert result["primary_action"] == "Opening a door"
        assert result["recorded_action"]["label"] == "Opening a door"
        assert "annotation" in result["recorded_action"]["note"]
        scoped = client.get("/api/video-memory", params={"q": "open door", "video_id": "MISSING"})
        assert scoped.status_code == 200
        assert scoped.json()["results"] == []


def test_video_summary_and_follow_up_are_evidence_scoped(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    resources = AppResources(
        service=service,
        memory=service.memory,
        queries=service.queries,
        inspections=InspectionStore(tmp_path / "findings.sqlite3"),
        charades_windows=[
            {
                "window_id": "VID:0-4", "video_id": "VID", "video_path": str(tmp_path / "VID.mp4"),
                "split": "train", "start_s": 0.0, "end_s": 4.0,
                "actions": [{"action_id": "c1", "name": "Opening a door", "start_s": 1.0, "end_s": 2.0}],
                "objects": ["door"], "description": "A person opens a door.;Another view.",
            },
            {
                "window_id": "VID:2-6", "video_id": "VID", "video_path": str(tmp_path / "VID.mp4"),
                "split": "train", "start_s": 2.0, "end_s": 6.0,
                "actions": [{"action_id": "c2", "name": "Sitting in a chair", "start_s": 3.0, "end_s": 5.0}],
                "objects": ["chair"], "description": "A person sits.",
            },
        ],
    )
    with TestClient(create_app(AppConfig(web_dist=tmp_path / "missing"), resources=resources)) as client:
        summary = client.post("/api/video-memory/summarize", json={"video_id": "VID"})
        assert summary.status_code == 200
        assert [event["label"] for event in summary.json()["events"]] == ["Opening a door", "Sitting in a chair"]
        follow_up = client.post("/api/video-memory/follow-up", json={"video_id": "VID", "question": "What happened?", "start_s": 2.0, "end_s": 4.0})
        assert follow_up.status_code == 200
        assert "Sitting in a chair" in follow_up.json()["answer"]
        finding = client.post("/api/video-memory/findings", json={"video_id": "VID", "question": "What happened?", "start_s": 2.0, "end_s": 4.0, "answer": "A person sat down.", "evidence_window_ids": ["VID:2-6"], "status": "confirmed", "note": "Checked."})
        assert finding.status_code == 200
        finding_id = finding.json()["id"]
        assert client.get("/api/video-memory/findings").json()[0]["id"] == finding_id
        assert client.get(f"/api/video-memory/findings/{finding_id}").json()["status"] == "confirmed"


def test_image_search_validation_and_allowlisted_serving(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/search/image",
            data={"display_k": "3"},
            files={"image": ("query.png", _png_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["query"] == {"kind": "image", "question": None}

        invalid_type = client.post(
            "/api/search/image",
            files={"image": ("query.txt", b"no", "text/plain")},
        )
        assert invalid_type.status_code == 415

        corrupt = client.post(
            "/api/search/image",
            files={"image": ("query.png", b"not an image", "image/png")},
        )
        assert corrupt.status_code == 422

        image = client.get("/api/images/memory/memory:0")
        assert image.status_code == 200
        unknown = client.get("/api/images/memory/../../secret")
        assert unknown.status_code in {404, 422}


def test_evaluation_zone_and_failure_endpoints(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/api/memory/evaluation").json()["pose"]["query_count"] == 1
        assert len(client.get("/api/zones").json()) == 2
        assert client.get("/api/zones/zone-a").json()["name"] == "Window desk"
        assert client.get("/api/zones/missing").status_code == 404

        page = client.get("/api/queries", params={"tag": "large-translation"}).json()
        assert page["total"] == 1
        detail = client.get("/api/queries/query:0").json()
        assert detail["retrievals"][0]["image_url"].endswith("memory:0")
        assert client.get("/api/queries/missing").status_code == 404


def test_object_showcase_and_allowlisted_image(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    resources = AppResources(
        service=service,
        memory=service.memory,
        queries=service.queries,
        objects=make_object_showcase(tmp_path / "objects"),
    )
    with TestClient(create_app(AppConfig(web_dist=tmp_path / "missing"), resources=resources)) as client:
        response = client.get("/api/objects")
        assert response.status_code == 200
        assert response.json()["metrics"]["detection_count"] == 1
        image = client.get("/api/objects/images/eth-office-0-000001-raw")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert client.get("/api/objects/images/unknown").status_code == 404


class FakeAnalyzer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def analyze(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "question_type": "location",
            "supported": True,
            "answer": "The desk is beside the window.",
            "evidence_citations": [
                {"observation_id": "memory:0", "claim": "The desk and window are visible."}
            ],
            "evidence_strength": "high",
            "limitations": ["Calendar time is unavailable."],
            "model": "fake-model",
            "cached": False,
        }


def test_analysis_requires_available_analyzer_and_selected_memory(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        unavailable = client.post(
            "/api/analyze/text",
            json={"question": "Where is the desk?", "evidence_ids": ["memory:0"]},
        )
        assert unavailable.status_code == 503

    service = make_service(tmp_path)
    analyzer = FakeAnalyzer()
    resources = AppResources(
        service=service,
        memory=service.memory,
        queries=service.queries,
        analysis=analyzer,
    )
    with TestClient(create_app(AppConfig(web_dist=tmp_path / "missing"), resources=resources)) as client:
        assert client.get("/api/capabilities").json()["analysis_available"] is True
        response = client.post(
            "/api/analyze/text",
            json={"question": "Where is the desk?", "evidence_ids": ["memory:0"]},
        )
        assert response.status_code == 200
        assert response.json()["evidence_citations"][0]["observation_id"] == "memory:0"
        assert len(analyzer.calls) == 1
        unknown = client.post(
            "/api/analyze/text",
            json={"question": "Where?", "evidence_ids": ["unknown"]},
        )
        assert unknown.status_code == 422
