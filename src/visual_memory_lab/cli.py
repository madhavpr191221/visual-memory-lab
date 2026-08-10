"""Command-line interface for Visual Memory Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from visual_memory_lab import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-memory-lab",
        description="Generate and inspect simulator-based visual memories.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
