"""Command-line interface for Visual Memory Lab."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from visual_memory_lab import __version__
from visual_memory_lab.memory import MemoryIndex, build_index, ensure_matching_encoder
from visual_memory_lab.trajectory import GenerationConfig, generate_trajectories


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-memory-lab",
        description="Build and evaluate simulator or real-image visual memories.",
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

    index = subparsers.add_parser(
        "index",
        help="build a frozen CLIP index over a generated trajectory",
    )
    index.add_argument("--input", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--device", default="auto")
    index.add_argument("--batch-size", type=_positive_integer, default=64)

    query = subparsers.add_parser(
        "query",
        help="retrieve observations from a visual-memory index",
    )
    query.add_argument("--index", type=Path, required=True)
    query_input = query.add_mutually_exclusive_group(required=True)
    query_input.add_argument("--text")
    query_input.add_argument("--image", type=Path)
    query_input.add_argument("--observation-id")
    query.add_argument("--top-k", type=_positive_integer, default=5)
    query.add_argument("--episode-id")
    query.add_argument("--include-self", action="store_true")
    query.add_argument("--device", default="auto")
    query.add_argument("--json", action="store_true")

    prepare = subparsers.add_parser(
        "prepare-7-scenes",
        help="validate 7-Scenes Office and write official train/test manifests",
    )
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    zones = subparsers.add_parser(
        "label-zones",
        help="curate cached VLM-assisted place zones from a training manifest",
    )
    zones.add_argument("--input", type=Path, required=True)
    zones.add_argument("--output", type=Path, required=True)
    zones.add_argument("--cache-dir", type=Path, default=Path("outputs/phase3/vlm-cache"))
    zones.add_argument("--model", default="gpt-5.6-terra")

    evaluate = subparsers.add_parser(
        "evaluate-real-memory",
        help="evaluate held-out pose retrieval and frozen semantic place zones",
    )
    evaluate.add_argument("--memory-index", type=Path, required=True)
    evaluate.add_argument("--query-index", type=Path, required=True)
    evaluate.add_argument("--zones", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--seed", type=int, default=42)

    serve = subparsers.add_parser(
        "serve-ui",
        help="serve the local React office-memory explorer",
    )
    serve.add_argument(
        "--memory-index", type=Path, default=Path("outputs/phase3/train-index")
    )
    serve.add_argument(
        "--query-index", type=Path, default=Path("outputs/phase3/test-index")
    )
    serve.add_argument(
        "--zones", type=Path, default=Path("artifacts/phase3/office-zones.json")
    )
    serve.add_argument(
        "--evaluation", type=Path, default=Path("outputs/phase3/evaluation")
    )
    serve.add_argument("--web-dist", type=Path, default=Path("web/dist"))
    serve.add_argument("--analysis-cache", type=Path, default=Path("outputs/phase4/vlm-cache"))
    serve.add_argument("--analysis-model", default="gpt-5.6-terra")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_integer, default=8000)
    serve.add_argument("--verify-source", action="store_true")
    return parser


def _query_payload(args: argparse.Namespace) -> dict[str, object]:
    index = MemoryIndex.load(args.index)
    query_description: dict[str, object]
    exclude_id: str | None = None

    if args.observation_id is not None:
        vector = index.observation_embedding(args.observation_id)
        query_description = {
            "kind": "observation",
            "observation_id": args.observation_id,
        }
        if not args.include_self:
            exclude_id = args.observation_id
    else:
        if args.include_self:
            raise ValueError("--include-self requires --observation-id")
        from visual_memory_lab.encoder import ClipEncoder

        encoder = ClipEncoder(device=args.device)
        ensure_matching_encoder(index, encoder)
        if args.text is not None:
            if not args.text.strip():
                raise ValueError("text query must not be empty")
            vector = encoder.encode_texts([args.text])[0]
            query_description = {"kind": "text", "text": args.text}
        else:
            if not args.image.is_file():
                raise ValueError(f"query image does not exist: {args.image}")
            vector = encoder.encode_images([args.image.resolve()])[0]
            query_description = {
                "kind": "image",
                "image_path": str(args.image.resolve()),
            }

    results = index.search(
        vector,
        top_k=args.top_k,
        episode_id=args.episode_id,
        exclude_observation_id=exclude_id,
    )
    return {
        "query": query_description,
        "episode_filter": args.episode_id,
        "index": {
            "path": str(index.root),
            "model_id": index.model_id,
            "model_revision": index.model_revision,
        },
        "results": [result.to_dict() for result in results],
    }


def _print_query(payload: dict[str, object]) -> None:
    results = payload["results"]
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, dict)
        observation = result["observation"]
        assert isinstance(observation, dict)
        if "agent_position" in observation:
            visible_objects = observation.get("visible_objects", [])
            visible_ids = ", ".join(
                str(item["object_id"])
                for item in visible_objects
                if isinstance(item, dict) and "object_id" in item
            ) or "none"
            context = (
                f'pose={observation["agent_position"]} '
                f'direction={observation["agent_direction_name"]} visible={visible_ids}'
            )
        else:
            pose = observation.get("camera_pose", {})
            translation = pose.get("translation_m") if isinstance(pose, dict) else None
            context = (
                f'sequence={observation.get("sequence_id", observation.get("episode_id"))} '
                f'frame={observation.get("step")} translation_m={translation}'
            )
        print(
            f'{result["rank"]}. {observation["observation_id"]} '
            f'score={float(result["score"]):.4f} {context}'
        )
        print(f'   image: {result["image_path"]}')
        if result["nearby_actions"]:
            print(f'   nearby actions: {result["nearby_actions"]}')


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
    elif args.command == "index":
        try:
            resolved_output = args.output.resolve()
            if resolved_output.exists() and (
                not resolved_output.is_dir() or any(resolved_output.iterdir())
            ):
                raise FileExistsError(f"output path is not empty: {resolved_output}")
            from visual_memory_lab.encoder import ClipEncoder

            encoder = ClipEncoder(device=args.device)
            summary = build_index(
                source=args.input,
                output=args.output,
                encoder=encoder,
                batch_size=args.batch_size,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Indexed {summary.observation_count} observations as "
            f"{summary.embedding_dim}-dimensional CLIP embeddings in "
            f"{summary.output}"
        )
    elif args.command == "query":
        try:
            payload = _query_payload(args)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _print_query(payload)
    elif args.command == "prepare-7-scenes":
        from visual_memory_lab.seven_scenes import prepare_office_dataset

        try:
            summary = prepare_office_dataset(dataset_root=args.input, output=args.output)
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Prepared {summary.train_count} train and {summary.test_count} test "
            f"RGB observations in {summary.output}; depth was not used"
        )
    elif args.command == "label-zones":
        from visual_memory_lab.zone_labeling import label_zones

        try:
            artifact = label_zones(
                source=args.input,
                output=args.output,
                cache_dir=args.cache_dir,
                model=args.model,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Curated {len(artifact['zones'])} place zones in {args.output.resolve()}"
        )
    elif args.command == "evaluate-real-memory":
        from visual_memory_lab.encoder import ClipEncoder
        from visual_memory_lab.evaluation import write_evaluation

        try:
            memory = MemoryIndex.load(args.memory_index)
            queries = MemoryIndex.load(args.query_index)
            encoder = ClipEncoder(device=args.device)
            metrics = write_evaluation(
                memory=memory,
                queries=queries,
                zones_path=args.zones,
                encoder=encoder,
                output=args.output,
                seed=args.seed,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Evaluated {metrics['pose']['query_count']} image queries and "
            f"{metrics['text_zones']['prompt_count']} text queries in {args.output.resolve()}"
        )
    elif args.command == "serve-ui":
        import uvicorn

        from visual_memory_lab.api import AppConfig, create_app

        app = create_app(
            AppConfig(
                memory_index=args.memory_index,
                query_index=args.query_index,
                zones=args.zones,
                evaluation=args.evaluation,
                web_dist=args.web_dist,
                device=args.device,
                verify_source=args.verify_source,
                analysis_model=args.analysis_model,
                analysis_cache=args.analysis_cache,
            )
        )
        uvicorn.run(app, host=args.host, port=args.port)
    return 0
