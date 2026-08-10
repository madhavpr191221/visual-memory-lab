"""Dependency smoke tests for headless MiniGrid rendering."""

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments
import numpy as np


def test_minigrid_can_render_rgb_array_headlessly() -> None:
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=42)
        frame = env.render()
    finally:
        env.close()

    assert "image" in observation
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
