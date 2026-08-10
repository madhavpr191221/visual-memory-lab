"""Serializable observation contract for generated visual memories."""

from __future__ import annotations

from dataclasses import dataclass

from visual_memory_lab.environment import InspectionGridEnv, SceneObject

SCHEMA_VERSION = "1.0"
DIRECTION_NAMES = ("east", "south", "west", "north")


def visible_scene_objects(env: InspectionGridEnv) -> tuple[SceneObject, ...]:
    """Return non-structural scene objects visible from the current agent pose."""

    return tuple(
        sorted(
            (
                obj
                for obj in env.scene_objects
                if env.agent_sees(*obj.position)
            ),
            key=lambda obj: obj.object_id,
        )
    )


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    episode_id: str
    step: int
    sim_time: int
    environment_seed: int
    image_path: str
    agent_position: tuple[int, int]
    agent_direction: int
    action: str | None
    visible_objects: tuple[SceneObject, ...]

    @property
    def agent_direction_name(self) -> str:
        return DIRECTION_NAMES[self.agent_direction]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_id": self.observation_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "sim_time": self.sim_time,
            "environment_seed": self.environment_seed,
            "image_path": self.image_path,
            "agent_position": list(self.agent_position),
            "agent_direction": self.agent_direction,
            "agent_direction_name": self.agent_direction_name,
            "action": self.action,
            "visible_objects": [obj.to_dict() for obj in self.visible_objects],
        }
