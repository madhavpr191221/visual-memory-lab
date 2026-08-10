"""Trajectory generation and artifact persistence."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from visual_memory_lab.environment import (
    AGENT_VIEW_SIZE,
    MAP_HEIGHT,
    MAP_ID,
    MAP_WIDTH,
    TILE_SIZE,
    InspectionGridEnv,
    make_environment,
)
from visual_memory_lab.observations import (
    SCHEMA_VERSION,
    ObservationRecord,
    visible_scene_objects,
)
from visual_memory_lab.route import build_scripted_actions

ROUTE_ID = "scripted-inspection-v1"


@dataclass(frozen=True)
class GenerationConfig:
    output: Path
    episodes: int = 10
    base_seed: int = 42
    max_steps: int = 100

    def validate(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


@dataclass(frozen=True)
class GenerationSummary:
    output: Path
    episode_count: int
    observation_count: int


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        path,
        format="PNG",
        compress_level=9,
        optimize=False,
    )


def _record_observation(
    *,
    env: InspectionGridEnv,
    observation: dict[str, object],
    run_root: Path,
    episode_id: str,
    episode_seed: int,
    step: int,
    action: str | None,
) -> dict[str, object]:
    relative_path = Path("episodes") / episode_id / "frames" / f"{step:04d}.png"
    _save_png(run_root / relative_path, np.asarray(observation["image"]))

    record = ObservationRecord(
        observation_id=f"{episode_id}:{step:04d}",
        episode_id=episode_id,
        step=step,
        sim_time=step,
        environment_seed=episode_seed,
        image_path=relative_path.as_posix(),
        agent_position=tuple(int(value) for value in env.agent_pos),
        agent_direction=int(env.agent_dir),
        action=action,
        visible_objects=visible_scene_objects(env),
    )
    return record.to_dict()


def _generate_episode(
    *,
    run_root: Path,
    episode_index: int,
    episode_seed: int,
    max_steps: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    episode_id = f"episode-{episode_index:03d}"
    episode_root = run_root / "episodes" / episode_id
    frames_root = episode_root / "frames"
    frames_root.mkdir(parents=True)

    wrapped_env = make_environment(max_steps=max_steps)
    try:
        observation, _ = wrapped_env.reset(seed=episode_seed)
        env = wrapped_env.unwrapped
        actions = build_scripted_actions(env)
        if len(actions) >= max_steps:
            raise ValueError(
                f"max_steps={max_steps} is too small for the {len(actions)}-action route"
            )

        overview = env.get_frame(
            highlight=False,
            tile_size=TILE_SIZE,
            agent_pov=False,
        )
        overview_relative = Path("episodes") / episode_id / "overview.png"
        _save_png(run_root / overview_relative, overview)

        records = [
            _record_observation(
                env=env,
                observation=observation,
                run_root=run_root,
                episode_id=episode_id,
                episode_seed=episode_seed,
                step=0,
                action=None,
            )
        ]

        for step, action in enumerate(actions, start=1):
            observation, _, terminated, truncated, _ = wrapped_env.step(action)
            if terminated or truncated:
                ending = "terminated" if terminated else "truncated"
                raise RuntimeError(
                    f"{episode_id} unexpectedly {ending} at simulator step {step}"
                )
            records.append(
                _record_observation(
                    env=env,
                    observation=observation,
                    run_root=run_root,
                    episode_id=episode_id,
                    episode_seed=episode_seed,
                    step=step,
                    action=action.name,
                )
            )

        scene = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "environment_seed": episode_seed,
            "map_id": MAP_ID,
            "route_id": ROUTE_ID,
            "max_steps": max_steps,
            "action_count": len(actions),
            "frame_count": len(records),
            "stop_reason": "route_complete",
            "overview_path": overview_relative.as_posix(),
            "scene_objects": [obj.to_dict() for obj in env.scene_objects],
        }
        _write_json(episode_root / "scene.json", scene)
        return scene, records
    finally:
        wrapped_env.close()


def _generate_into(config: GenerationConfig, run_root: Path) -> GenerationSummary:
    all_records: list[dict[str, object]] = []
    episode_entries: list[dict[str, object]] = []

    for episode_index in range(config.episodes):
        episode_seed = config.base_seed + episode_index
        scene, records = _generate_episode(
            run_root=run_root,
            episode_index=episode_index,
            episode_seed=episode_seed,
            max_steps=config.max_steps,
        )
        all_records.extend(records)
        episode_entries.append(
            {
                "episode_id": scene["episode_id"],
                "environment_seed": episode_seed,
                "scene_path": (
                    Path("episodes") / str(scene["episode_id"]) / "scene.json"
                ).as_posix(),
                "overview_path": scene["overview_path"],
                "observation_count": len(records),
            }
        )

    observations_path = run_root / "observations.jsonl"
    observations_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in all_records
        ),
        encoding="utf-8",
    )

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "map_id": MAP_ID,
        "route_id": ROUTE_ID,
        "base_seed": config.base_seed,
        "episode_count": config.episodes,
        "max_steps": config.max_steps,
        "observation_count": len(all_records),
        "agent_view": {
            "grid_size": AGENT_VIEW_SIZE,
            "tile_size": TILE_SIZE,
            "image_width": AGENT_VIEW_SIZE * TILE_SIZE,
            "image_height": AGENT_VIEW_SIZE * TILE_SIZE,
        },
        "overview": {
            "image_width": MAP_WIDTH * TILE_SIZE,
            "image_height": MAP_HEIGHT * TILE_SIZE,
        },
        "episodes": episode_entries,
    }
    _write_json(run_root / "run.json", run_manifest)

    return GenerationSummary(
        output=config.output,
        episode_count=config.episodes,
        observation_count=len(all_records),
    )


def generate_trajectories(config: GenerationConfig) -> GenerationSummary:
    """Generate a complete run without overwriting existing artifacts."""

    config.validate()
    output = config.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output path is not empty: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        summary = _generate_into(config, temporary)
        if output.exists():
            output.rmdir()
        temporary.replace(output)
        return GenerationSummary(
            output=output,
            episode_count=summary.episode_count,
            observation_count=summary.observation_count,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
