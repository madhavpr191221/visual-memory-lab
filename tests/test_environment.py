"""Tests for the controlled Phase 1 environment and scripted route."""

from __future__ import annotations

from collections import defaultdict

from visual_memory_lab.environment import (
    AGENT_START,
    LEFT_OBJECT_SLOTS,
    MAP_HEIGHT,
    MAP_WIDTH,
    RIGHT_OBJECT_SLOTS,
    InspectionGridEnv,
    make_environment,
)
from visual_memory_lab.route import build_scripted_actions


def _scene_positions(seed: int) -> dict[str, tuple[int, int]]:
    env = InspectionGridEnv()
    try:
        env.reset(seed=seed)
        return {obj.object_id: obj.position for obj in env.scene_objects}
    finally:
        env.close()


def test_seeded_scene_is_reproducible_and_varies_between_seeds() -> None:
    assert _scene_positions(42) == _scene_positions(42)
    assert _scene_positions(42) != _scene_positions(43)


def test_map_geometry_and_object_contract() -> None:
    env = InspectionGridEnv()
    try:
        env.reset(seed=42)
        assert env.width == MAP_WIDTH
        assert env.height == MAP_HEIGHT
        assert tuple(env.agent_pos) == AGENT_START

        for x in (6, 7, 8):
            assert env.grid.get(x, 4) is None
            assert env.grid.get(x, 3).type == "wall"

        objects = {obj.object_id: obj for obj in env.scene_objects}
        assert set(objects) == {"red-ball-a", "red-ball-b", "blue-box"}
        assert objects["red-ball-a"].position in LEFT_OBJECT_SLOTS
        assert objects["red-ball-b"].position in RIGHT_OBJECT_SLOTS
        assert len({obj.position for obj in objects.values()}) == 3
    finally:
        env.close()


def test_scripted_route_completes_and_observes_scene() -> None:
    wrapped_env = make_environment(max_steps=100)
    try:
        wrapped_env.reset(seed=42)
        env = wrapped_env.unwrapped
        actions = build_scripted_actions(env)
        seen_from: dict[str, set[tuple[int, int]]] = defaultdict(set)

        def record_visibility() -> None:
            for obj in env.scene_objects:
                if env.agent_sees(*obj.position):
                    seen_from[obj.object_id].add(tuple(env.agent_pos))

        record_visibility()
        for action in actions:
            _, _, terminated, truncated, _ = wrapped_env.step(action)
            assert not terminated
            assert not truncated
            record_visibility()

        assert len(actions) < 100
        assert set(seen_from) == {obj.object_id for obj in env.scene_objects}
        assert len(seen_from["red-ball-a"]) >= 2
        assert len(seen_from["red-ball-b"]) >= 2
    finally:
        wrapped_env.close()
