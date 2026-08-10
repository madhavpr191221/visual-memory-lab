"""Deterministic inspection route for the controlled environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from minigrid.core.actions import Actions

from visual_memory_lab.environment import InspectionGridEnv


@dataclass(frozen=True)
class PoseTarget:
    position: tuple[int, int]
    direction: int


INSPECTION_TARGETS = (
    PoseTarget((3, 4), 3),
    PoseTarget((5, 4), 3),
    PoseTarget((5, 4), 1),
    PoseTarget((3, 4), 1),
    PoseTarget((7, 4), 0),
    PoseTarget((10, 4), 3),
    PoseTarget((12, 4), 3),
    PoseTarget((12, 4), 1),
    PoseTarget((10, 4), 1),
    PoseTarget((7, 4), 2),
    PoseTarget((5, 4), 2),
)

_DIRECTION_TO_VECTOR = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


def _turn_actions(current: int, target: int) -> list[Actions]:
    difference = (target - current) % 4
    if difference == 0:
        return []
    if difference == 1:
        return [Actions.right]
    if difference == 3:
        return [Actions.left]
    return [Actions.right, Actions.right]


def _shortest_path(
    env: InspectionGridEnv,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    queue = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    for_position = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        position = queue.popleft()
        if position == goal:
            break
        for dx, dy in for_position:
            candidate = (position[0] + dx, position[1] + dy)
            if candidate in previous:
                continue
            if not (0 <= candidate[0] < env.width and 0 <= candidate[1] < env.height):
                continue
            cell = env.grid.get(*candidate)
            if cell is not None and not cell.can_overlap():
                continue
            previous[candidate] = position
            queue.append(candidate)

    if goal not in previous:
        raise ValueError(f"No route from {start} to {goal}")

    path = [goal]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def build_scripted_actions(env: InspectionGridEnv) -> list[Actions]:
    """Build the deterministic action sequence from the current seeded scene."""

    actions: list[Actions] = []
    position = tuple(int(value) for value in env.agent_pos)
    direction = int(env.agent_dir)

    for target in INSPECTION_TARGETS:
        path = _shortest_path(env, position, target.position)
        for next_position in path[1:]:
            delta = (
                next_position[0] - position[0],
                next_position[1] - position[1],
            )
            next_direction = next(
                key for key, vector in _DIRECTION_TO_VECTOR.items() if vector == delta
            )
            turns = _turn_actions(direction, next_direction)
            actions.extend(turns)
            actions.append(Actions.forward)
            direction = next_direction
            position = next_position

        turns = _turn_actions(direction, target.direction)
        actions.extend(turns)
        direction = target.direction

    return actions
