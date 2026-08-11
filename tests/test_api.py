"""API contract and security tests for the Office memory explorer."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from visual_memory_lab.api import AppConfig, AppResources, create_app

from tests.test_ui_service import make_service


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
        assert client.get("/api/evaluation").json()["pose"]["query_count"] == 1
        assert len(client.get("/api/zones").json()) == 2
        assert client.get("/api/zones/zone-a").json()["name"] == "Window desk"
        assert client.get("/api/zones/missing").status_code == 404

        page = client.get("/api/queries", params={"tag": "large-translation"}).json()
        assert page["total"] == 1
        detail = client.get("/api/queries/query:0").json()
        assert detail["retrievals"][0]["image_url"].endswith("memory:0")
        assert client.get("/api/queries/missing").status_code == 404


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
