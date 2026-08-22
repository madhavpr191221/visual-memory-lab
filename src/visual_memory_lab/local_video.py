"""Ephemeral, local-only video sessions for bring-your-own-video testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import secrets
import threading
import time
from typing import Callable

import numpy as np

from visual_memory_lab.encoder import ClipEncoder
from visual_memory_lab.learned_video import sample_window_timestamps

logger = logging.getLogger("visual_memory_lab.local_video")


def group_local_results(results: list[dict[str, object]], *, adjacency_s: float = 0.25) -> list[dict[str, object]]:
    """Collapse overlapping CLIP windows into reviewable candidate moments.

    Local uploads have no action annotations, so these groups are deliberately
    described as visual candidates rather than events.  The best-scoring raw
    window remains the representative record while all contributing windows
    are retained in ``evidence_window_ids``.
    """
    if not results:
        return []
    ordered = sorted(results, key=lambda item: (float(item.get("start_s", 0.0)), -float(item.get("score", 0.0))))
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_end = 0.0
    for item in ordered:
        start = float(item.get("start_s", 0.0))
        end = float(item.get("end_s", start))
        if current and start > current_end + adjacency_s:
            groups.append(current)
            current = []
        current.append(item)
        current_end = max(current_end, end)
    if current:
        groups.append(current)

    grouped: list[dict[str, object]] = []
    for members in groups:
        representative = max(members, key=lambda item: float(item.get("score", 0.0)))
        item = dict(representative)
        start = min(float(member.get("start_s", 0.0)) for member in members)
        end = max(float(member.get("end_s", start)) for member in members)
        context_start = min(float(member.get("context_start_s", member.get("start_s", start))) for member in members)
        context_end = max(float(member.get("context_end_s", member.get("end_s", end))) for member in members)
        timestamps = sorted({
            round(float(timestamp), 3)
            for member in members
            for timestamp in member.get("frame_timestamps_s", [])
        })
        # Object inspection accepts at most 32 timestamps.  Preserve temporal
        # coverage across the whole grouped moment rather than keeping only
        # the first raw window's frames.
        if len(timestamps) > 32:
            sample_indices = np.linspace(0, len(timestamps) - 1, 32, dtype=int)
            timestamps = [timestamps[int(index)] for index in sample_indices]
        ids = [str(member.get("window_id")) for member in members if member.get("window_id")]
        item.update({
            "start_s": start,
            "end_s": end,
            "context_start_s": context_start,
            "context_end_s": context_end,
            "action_start_s": start,
            "action_end_s": end,
            "evidence_start_s": context_start,
            "evidence_end_s": context_end,
            "frame_timestamps_s": timestamps,
            "evidence_window_ids": ids,
            "candidate_kind": "grouped_visual_moment",
            "grouped_window_count": len(members),
            "retrieved_window_start_s": float(representative.get("start_s", start)),
            "retrieved_window_end_s": float(representative.get("end_s", end)),
            "primary_action": "Visually similar moment",
            "interval_source": "visual_retrieval_group",
            "result_limitations": [
                "This private video has no ground-truth action annotation.",
                "The displayed interval groups overlapping CLIP windows; it is not a verified event boundary.",
            ],
        })
        grouped.append(item)
    return sorted(grouped, key=lambda item: -float(item.get("score", 0.0)))


@dataclass
class LocalVideoSession:
    video_id: str
    path: Path
    duration_s: float
    records: list[dict[str, object]]
    vectors: np.ndarray
    created_at: float = field(default_factory=time.time)
    status: str = "ready"
    error: str | None = None


@dataclass
class LocalVideoJob:
    upload_id: str
    status: str = "uploading"
    stage: str = "uploading"
    progress: float = 0.0
    message: str = "Uploading your video locally…"
    device: str = "cpu"
    duration_s: float | None = None
    windows_done: int = 0
    windows_total: int = 0
    video_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


def _duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return max(float(stream.duration * stream.time_base), 0.01)
        return max(float(container.duration or 0) / 1_000_000.0, 0.01)


def _frames(path: Path, timestamps: list[float]):
    import av

    wanted = sorted(timestamps)
    output: list[object] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            current = float(frame.time or 0.0)
            while wanted and current >= wanted[0]:
                output.append(frame.to_image().convert("RGB"))
                wanted.pop(0)
            if not wanted:
                break
    if not output:
        raise ValueError("video contains no decodable RGB frames")
    while len(output) < len(timestamps):
        output.append(output[-1].copy())
    return output


class LocalVideoManager:
    """Process user videos in memory and under an ephemeral local directory."""

    def __init__(self, root: Path, encoder: ClipEncoder, *, ttl_s: int = 86_400) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.ttl_s = ttl_s
        self.sessions: dict[str, LocalVideoSession] = {}
        self.jobs: dict[str, LocalVideoJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-video")
        self.cleanup()

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl_s
        for path in self.root.glob("*"):
            if path.is_dir() and path.stat().st_mtime < cutoff:
                for child in path.iterdir():
                    child.unlink(missing_ok=True)
                path.rmdir()
        self.sessions = {key: value for key, value in self.sessions.items() if value.created_at >= cutoff}
        self.jobs = {key: value for key, value in self.jobs.items() if value.created_at >= cutoff}

    def create_job(self, upload_id: str, *, device: str) -> LocalVideoJob:
        job = LocalVideoJob(upload_id=upload_id, device=device)
        with self._lock:
            self.jobs[upload_id] = job
        return job

    def update_job(self, upload_id: str, **changes: object) -> None:
        with self._lock:
            job = self.jobs.get(upload_id)
            if job is not None:
                for key, value in changes.items():
                    setattr(job, key, value)

    def job(self, upload_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self.jobs.get(upload_id)
            if job is None:
                return None
            return {
                "upload_id": job.upload_id,
                "status": job.status,
                "stage": job.stage,
                "progress": round(float(job.progress), 4),
                "message": job.message,
                "device": job.device,
                "duration_s": job.duration_s,
                "windows_done": job.windows_done,
                "windows_total": job.windows_total,
                "video_id": job.video_id,
                "error": job.error,
            }

    def start_import(self, source: Path, *, original_name: str, upload_id: str) -> None:
        self._executor.submit(self._run_import, source, original_name, upload_id)

    def _run_import(self, source: Path, original_name: str, upload_id: str) -> None:
        try:
            self.update_job(upload_id, status="decoding", stage="decoding", progress=0.10, message="Checking the video and reading its duration…")
            duration = _duration(source)
            window_s, stride_s, frame_count = 4.0, 2.0, 16
            starts = list(np.arange(0.0, max(duration - 1e-6, 0.0), stride_s)) or [0.0]
            self.update_job(upload_id, duration_s=duration, windows_total=len(starts), progress=0.15, message=f"Video checked: {duration:.1f} seconds. Preparing visual memory…")
            logger.info("Local video probe complete: upload_id=%s duration=%.1fs windows=%d device=%s", upload_id, duration, len(starts), self.encoder.device)

            def report(done: int, total: int) -> None:
                self.update_job(
                    upload_id,
                    status="embedding",
                    stage="embedding",
                    progress=0.15 + 0.80 * (done / max(total, 1)),
                    windows_done=done,
                    windows_total=total,
                    message=f"Building visual memory: window {done} of {total}…",
                )
                if done == 1 or done == total or done % 5 == 0:
                    logger.info("Local visual memory: upload_id=%s window=%d/%d", upload_id, done, total)

            session = self.import_video(
                source,
                original_name=original_name,
                progress=report,
            )
            try:
                source.parent.rmdir()
            except OSError:
                logger.debug("Temporary upload directory was not empty: %s", source.parent)
            self.update_job(upload_id, status="finalizing", stage="finalizing", progress=0.97, message="Finishing local setup…")
            self.update_job(upload_id, status="ready", stage="ready", progress=1.0, video_id=session.video_id, message="Ready. Your local video is available for search.")
            logger.info("Local video ready: upload_id=%s video_id=%s windows=%d", upload_id, session.video_id, len(session.records))
        except Exception as error:
            source.unlink(missing_ok=True)
            try:
                source.parent.rmdir()
            except OSError:
                pass
            self.update_job(upload_id, status="failed", stage="failed", progress=1.0, error=str(error), message="The video could not be prepared.")
            logger.exception("Local video failed: upload_id=%s", upload_id)

    def import_video(self, source: Path, *, original_name: str = "video.mp4", progress: Callable[[int, int], None] | None = None) -> LocalVideoSession:
        self.cleanup()
        video_id = f"local-{secrets.token_hex(8)}"
        session_dir = self.root / video_id
        session_dir.mkdir(parents=True, exist_ok=False)
        target = session_dir / "video.mp4"
        source.replace(target)
        duration = _duration(target)
        window_s, stride_s, frame_count = 4.0, 2.0, 16
        records: list[dict[str, object]] = []
        vectors: list[np.ndarray] = []
        starts = list(np.arange(0.0, max(duration - 1e-6, 0.0), stride_s)) or [0.0]
        total_windows = len(starts)
        for index, start in enumerate(starts, start=1):
            end = min(start + window_s, duration)
            timestamps = sample_window_timestamps(start, end, frame_count)
            images = _frames(target, timestamps)
            vector = self.encoder.encode_pil_images(images).mean(axis=0)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
            vectors.append(vector.astype(np.float32))
            records.append({
                "window_id": f"{video_id}:{start:.2f}-{end:.2f}",
                "video_id": video_id,
                "video_path": str(target),
                "duration_s": duration,
                "start_s": start,
                "end_s": end,
                "context_start_s": max(0.0, start - 1.0),
                "context_end_s": min(duration, end + 1.0),
                "frame_timestamps_s": timestamps,
                "actions": [],
                "objects": [],
                "description": "User-provided local video; no dataset annotations are available.",
                "source": "local_upload",
                "original_name": original_name,
            })
            for image in images:
                image.close()
            if progress is not None:
                progress(index, total_windows)
        session = LocalVideoSession(video_id, target, duration, records, np.stack(vectors))
        self.sessions[video_id] = session
        (session_dir / "metadata.json").write_text(json.dumps({"video_id": video_id, "duration_s": duration, "source": "local_upload"}), encoding="utf-8")
        return session

    def get(self, video_id: str) -> LocalVideoSession | None:
        return self.sessions.get(video_id)

    def catalog(self) -> list[dict[str, object]]:
        return [{
            "video_id": item.video_id,
            "video_url": f"/api/video-memory/videos/{item.video_id}",
            "duration_s": item.duration_s,
            "description": "Local video; annotations are not available.",
            "objects": [],
            "actions": [],
            "source": "local_upload",
        } for item in self.sessions.values()]

    def search(self, video_id: str, query: str, top_k: int = 8) -> list[dict[str, object]]:
        session = self.sessions.get(video_id)
        if session is None:
            return []
        query_vector = self.encoder.encode_texts([query])[0]
        scores = session.vectors @ query_vector
        order = np.argsort(-scores)[:top_k]
        results = []
        for index in order:
            record = dict(session.records[int(index)])
            record["score"] = round(float(scores[int(index)]), 4)
            record["retrieval_mode"] = "local_clip_window"
            record["primary_action"] = "Visually similar moment"
            record["context_actions"] = []
            record["recorded_action"] = None
            record["action_start_s"] = record["start_s"]
            record["action_end_s"] = record["end_s"]
            record["interval_source"] = "visual_retrieval"
            record["candidate_kind"] = "raw_visual_window"
            record["result_limitations"] = ["This local video has no ground-truth action annotation.", "The timestamp is a retrieved visual candidate, not a verified event boundary."]
            results.append(record)
        return group_local_results(results)

    def all_records(self) -> list[dict[str, object]]:
        return [record for session in self.sessions.values() for record in session.records]
