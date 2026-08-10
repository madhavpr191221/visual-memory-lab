"""Controlled MiniGrid environment used by the visual-memory experiments."""

from __future__ import annotations

from dataclasses import dataclass

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper

MAP_ID = "inspection-grid-v1"
MAP_WIDTH = 15
MAP_HEIGHT = 9
AGENT_START = (2, 4)
AGENT_START_DIRECTION = 0
AGENT_VIEW_SIZE = 7
TILE_SIZE = 8

LEFT_OBJECT_SLOTS = ((2, 2), (4, 2), (2, 6), (4, 6))
RIGHT_OBJECT_SLOTS = ((10, 2), (12, 2), (10, 6), (12, 6))


@dataclass(frozen=True)
class SceneObject:
    """Ground-truth identity and state for an object in one episode."""

    object_id: str
    type: str
    color: str
    state: str
    position: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "type": self.type,
            "color": self.color,
            "state": self.state,
            "position": list(self.position),
        }


class InspectionGridEnv(MiniGridEnv):
    """Two mirrored rooms with seeded object placement and a fixed start pose."""

    def __init__(
        self,
        *,
        max_steps: int = 100,
        render_mode: str | None = "rgb_array",
    ) -> None:
        mission_space = MissionSpace(mission_func=self._mission)
        self.scene_objects: tuple[SceneObject, ...] = ()
        super().__init__(
            mission_space=mission_space,
            width=MAP_WIDTH,
            height=MAP_HEIGHT,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=AGENT_VIEW_SIZE,
            render_mode=render_mode,
            tile_size=TILE_SIZE,
            highlight=False,
        )

    @staticmethod
    def _mission() -> str:
        return "inspect both rooms"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Three wall columns separate the mirrored rooms. Their openings at y=4
        # form a narrow corridor and make room boundaries true occluders.
        for x in (6, 7, 8):
            for y in range(1, height - 1):
                self.grid.set(x, y, None if y == 4 else Wall())

        self.agent_pos = AGENT_START
        self.agent_dir = AGENT_START_DIRECTION
        self.mission = self._mission()

        left_slots = list(LEFT_OBJECT_SLOTS)
        right_slots = list(RIGHT_OBJECT_SLOTS)
        self.np_random.shuffle(left_slots)
        self.np_random.shuffle(right_slots)

        blue_room = int(self.np_random.integers(0, 2))
        placements: list[tuple[str, WorldObj, tuple[int, int], str]] = [
            ("red-ball-a", Ball("red"), left_slots[0], "stationary"),
            ("red-ball-b", Ball("red"), right_slots[0], "stationary"),
        ]
        if blue_room == 0:
            placements.append(
                ("blue-box", Box("blue"), left_slots[1], "closed")
            )
        else:
            placements.append(
                ("blue-box", Box("blue"), right_slots[1], "closed")
            )

        scene_objects: list[SceneObject] = []
        for object_id, world_object, position, state in placements:
            self.grid.set(*position, world_object)
            scene_objects.append(
                SceneObject(
                    object_id=object_id,
                    type=world_object.type,
                    color=world_object.color,
                    state=state,
                    position=position,
                )
            )
        self.scene_objects = tuple(scene_objects)

    def scene_object_at(self, position: tuple[int, int]) -> SceneObject | None:
        return next(
            (obj for obj in self.scene_objects if obj.position == position),
            None,
        )


def make_environment(*, max_steps: int = 100) -> RGBImgPartialObsWrapper:
    """Create the Phase 1 environment with egocentric RGB observations."""

    env = InspectionGridEnv(max_steps=max_steps, render_mode="rgb_array")
    return RGBImgPartialObsWrapper(env, tile_size=TILE_SIZE)
