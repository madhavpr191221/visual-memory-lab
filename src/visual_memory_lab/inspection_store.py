"""Local SQLite persistence for technician inspections."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class InspectionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_text TEXT NOT NULL,
                    limitations TEXT NOT NULL,
                    current_image_path TEXT,
                    selected_earlier_observation_id TEXT,
                    summary_json TEXT,
                    report_json TEXT
                );
                CREATE TABLE IF NOT EXISTS inspection_evidence (
                    inspection_id TEXT NOT NULL,
                    observation_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    rank INTEGER,
                    score REAL,
                    role TEXT NOT NULL,
                    PRIMARY KEY (inspection_id, observation_id),
                    FOREIGN KEY (inspection_id) REFERENCES inspections(id)
                );
                CREATE TABLE IF NOT EXISTS video_findings (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    answer TEXT NOT NULL,
                    evidence_window_ids TEXT NOT NULL,
                    status TEXT NOT NULL,
                    note TEXT NOT NULL,
                    limitations TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(inspections)").fetchall()}
            if "summary_json" not in columns:
                connection.execute("ALTER TABLE inspections ADD COLUMN summary_json TEXT")
            if "report_json" not in columns:
                connection.execute("ALTER TABLE inspections ADD COLUMN report_json TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(
        self,
        *,
        title: str,
        question: str,
        result_text: str,
        status: str,
        limitations: list[str],
        current_image_path: str | None,
        evidence: list[dict[str, object]],
    ) -> dict[str, object]:
        inspection_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO inspections (id, title, question, created_at, status, result_text, limitations, current_image_path, selected_earlier_observation_id, summary_json, report_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (inspection_id, title, question, created_at, status, result_text, json.dumps(limitations), current_image_path, None, None, None),
            )
            connection.executemany(
                "INSERT INTO inspection_evidence VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (inspection_id, str(item["observation_id"]), str(item.get("collection", "memory")), item.get("rank"), item.get("score"), str(item.get("role", "supporting")))
                    for item in evidence
                ],
            )
        return self.get(inspection_id)

    def list(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM inspections ORDER BY created_at DESC").fetchall()
        return [self._row(row) for row in rows]

    def get(self, inspection_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
            if row is None:
                raise KeyError(inspection_id)
            evidence = connection.execute("SELECT * FROM inspection_evidence WHERE inspection_id = ? ORDER BY rank", (inspection_id,)).fetchall()
        payload = self._row(row)
        payload["evidence"] = [dict(item) for item in evidence]
        return payload

    def update_comparison(
        self,
        inspection_id: str,
        *,
        earlier_observation_id: str,
        result_text: str,
        status: str,
        limitations: list[str],
    ) -> dict[str, object]:
        with self._connect() as connection:
            connection.execute(
                "UPDATE inspections SET selected_earlier_observation_id = ?, result_text = ?, status = ?, limitations = ? WHERE id = ?",
                (earlier_observation_id, result_text, status, json.dumps(limitations), inspection_id),
            )
        return self.get(inspection_id)

    def set_current_image(self, inspection_id: str, path: str) -> dict[str, object]:
        with self._connect() as connection:
            connection.execute("UPDATE inspections SET current_image_path = ? WHERE id = ?", (path, inspection_id))
        return self.get(inspection_id)

    def set_summary(self, inspection_id: str, summary: dict[str, object]) -> dict[str, object]:
        with self._connect() as connection:
            connection.execute("UPDATE inspections SET summary_json = ? WHERE id = ?", (json.dumps(summary), inspection_id))
        return self.get(inspection_id)

    def set_report(self, inspection_id: str, report: dict[str, object], *, result_text: str, status: str, limitations: list[str]) -> dict[str, object]:
        with self._connect() as connection:
            connection.execute("UPDATE inspections SET report_json = ?, result_text = ?, status = ?, limitations = ? WHERE id = ?", (json.dumps(report), result_text, status, json.dumps(limitations), inspection_id))
        return self.get(inspection_id)

    def create_video_finding(self, payload: dict[str, object]) -> dict[str, object]:
        finding_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO video_findings (id, video_id, question, start_s, end_s, answer, evidence_window_ids, status, note, limitations, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (finding_id, str(payload["video_id"]), str(payload["question"]), float(payload["start_s"]), float(payload["end_s"]), str(payload["answer"]), json.dumps(payload.get("evidence_window_ids", [])), str(payload.get("status", "unclear")), str(payload.get("note", "")), json.dumps(payload.get("limitations", [])), str(payload.get("source", "official_charades_annotations")), created_at),
            )
        return self.get_video_finding(finding_id)

    def list_video_findings(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM video_findings ORDER BY created_at DESC").fetchall()
        return [self._video_row(row) for row in rows]

    def get_video_finding(self, finding_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM video_findings WHERE id = ?", (finding_id,)).fetchone()
        if row is None:
            raise KeyError(finding_id)
        return self._video_row(row)

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, object]:
        payload = dict(row)
        payload["limitations"] = json.loads(str(payload["limitations"]))
        for key in ("summary_json", "report_json"):
            value = payload.get(key)
            payload[key] = json.loads(str(value)) if value else None
        return payload

    @staticmethod
    def _video_row(row: sqlite3.Row) -> dict[str, object]:
        payload = dict(row)
        payload["evidence_window_ids"] = json.loads(str(payload["evidence_window_ids"]))
        payload["limitations"] = json.loads(str(payload["limitations"]))
        return payload
