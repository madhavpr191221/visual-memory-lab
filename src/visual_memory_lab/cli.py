"""Command-line interface for Visual Memory Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from visual_memory_lab import __version__
from visual_memory_lab.trajectory import GenerationConfig, generate_trajectories


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-memory-lab",
        description="Generate and inspect simulator-based visual memories.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="generate reproducible simulator trajectories",
    )
    generate.add_argument("--episodes", type=_positive_integer, default=10)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--max-steps", type=_positive_integer, default=100)
    generate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        try:
            summary = generate_trajectories(
                GenerationConfig(
                    output=args.output,
                    episodes=args.episodes,
                    base_seed=args.seed,
                    max_steps=args.max_steps,
                )
            )
        except (FileExistsError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Generated {summary.observation_count} observations across "
            f"{summary.episode_count} episodes in {summary.output}"
        )
    return 0
