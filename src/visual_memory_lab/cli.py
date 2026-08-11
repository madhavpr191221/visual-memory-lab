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


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
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

    traversal = subparsers.add_parser(
        "evaluate-traversal-memory",
        help="evaluate pose-comparable retrieval from designated reference traversals",
    )
    traversal.add_argument("--memory-index", type=Path, required=True)
    traversal.add_argument("--query-index", type=Path, required=True)
    traversal.add_argument("--output", type=Path, required=True)
    traversal.add_argument("--seed", type=_non_negative_integer, default=42)

    prepare_eth = subparsers.add_parser(
        "prepare-eth-office",
        help="audit ETH Office bags and meshes and create a browsable RGB gallery",
    )
    prepare_eth.add_argument("--input", type=Path, required=True)
    prepare_eth.add_argument("--output", type=Path, required=True)
    prepare_eth.add_argument("--rgb-samples", type=_positive_integer, default=24)
    prepare_eth.add_argument("--vlm-samples", type=_positive_integer, default=8)

    evaluate_change = subparsers.add_parser(
        "evaluate-eth-change",
        help="produce deterministic change candidates from aligned ETH Office meshes",
    )
    evaluate_change.add_argument("--manifest", type=Path, required=True)
    evaluate_change.add_argument("--output", type=Path, required=True)
    evaluate_change.add_argument("--voxel-size", type=_positive_float, default=0.02)
    evaluate_change.add_argument(
        "--distance-thresholds", type=_positive_float, nargs="+", default=[0.02, 0.05, 0.10]
    )
    evaluate_change.add_argument("--primary-threshold", type=_positive_float, default=0.05)
    evaluate_change.add_argument("--min-cluster-voxels", type=_positive_integer, default=20)

    review_change = subparsers.add_parser(
        "review-eth-change",
        help="create a cached VLM pseudo-reference for ETH Office change candidates",
    )
    review_change.add_argument("--baseline", type=Path, required=True)
    review_change.add_argument("--audit", type=Path, required=True)
    review_change.add_argument("--output", type=Path, required=True)
    review_change.add_argument("--cache-dir", type=Path, default=Path("outputs/phase6a/vlm-cache"))
    review_change.add_argument("--model", default="gpt-5.6-terra")

    localize_objects = subparsers.add_parser(
        "localize-eth-objects",
        help="detect and segment movable objects in dense ETH Office keyframes",
    )
    localize_objects.add_argument("--input", type=Path, required=True)
    localize_objects.add_argument("--output", type=Path, required=True)
    localize_objects.add_argument(
        "--keyframes-per-observation", type=_positive_integer, default=96
    )
    localize_objects.add_argument("--device", default="auto")
    localize_objects.add_argument(
        "--detector-model", default="IDEA-Research/grounding-dino-tiny"
    )
    localize_objects.add_argument(
        "--segmenter-model", default="facebook/sam2.1-hiera-small"
    )
    localize_objects.add_argument("--box-threshold", type=_positive_float, default=0.25)
    localize_objects.add_argument("--text-threshold", type=_positive_float, default=0.20)
    localize_objects.add_argument("--nms-iou", type=_positive_float, default=0.50)
    localize_objects.add_argument("--max-detections", type=_positive_integer, default=20)

    audit_objects = subparsers.add_parser(
        "audit-eth-object-localization",
        help="create a cached VLM pseudo-audit of Phase 6B1 predictions",
    )
    audit_objects.add_argument("--localization", type=Path, required=True)
    audit_objects.add_argument("--output", type=Path, required=True)
    audit_objects.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/phase6b1/vlm-cache")
    )
    audit_objects.add_argument(
        "--frames-per-observation", type=_positive_integer, default=12
    )
    audit_objects.add_argument("--model", default="gpt-5.6-terra")

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
    serve.add_argument("--change-audit", type=Path, default=Path("outputs/phase6a/office-audit"))
    serve.add_argument("--change-baseline", type=Path, default=Path("outputs/phase6a/change-baseline"))
    serve.add_argument("--change-review", type=Path, default=Path("outputs/phase6a/vlm-review"))
    serve.add_argument(
        "--object-localization",
        type=Path,
        default=Path("outputs/phase6b1/object-localization"),
    )
    serve.add_argument(
        "--object-audit", type=Path, default=Path("outputs/phase6b1/vlm-audit")
    )
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
                change_audit=args.change_audit,
                change_baseline=args.change_baseline,
                change_review=args.change_review,
                object_localization=args.object_localization,
                object_audit=args.object_audit,
            )
        )
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "evaluate-traversal-memory":
        from visual_memory_lab.traversal_evaluation import write_traversal_evaluation

        try:
            memory = MemoryIndex.load(args.memory_index)
            queries = MemoryIndex.load(args.query_index)
            metrics = write_traversal_evaluation(
                memory=memory,
                queries=queries,
                output=args.output,
                seed=args.seed,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Evaluated {metrics['query_target_count']} query-target combinations "
            f"across {metrics['pair_count']} traversal pairs in {args.output.resolve()}"
        )
    elif args.command == "prepare-eth-office":
        from visual_memory_lab.eth_office import prepare_eth_office

        try:
            manifest = prepare_eth_office(
                dataset_root=args.input,
                output=args.output,
                rgb_samples=args.rgb_samples,
                vlm_samples=args.vlm_samples,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Prepared {len(manifest['observations'])} ETH Office observations with "
            f"{manifest['rgb_samples_per_observation']} RGB samples each in {args.output.resolve()}"
        )
    elif args.command == "evaluate-eth-change":
        from visual_memory_lab.change_detection import evaluate_eth_change

        try:
            run = evaluate_eth_change(
                manifest_path=args.manifest,
                output=args.output,
                voxel_size=args.voxel_size,
                distance_thresholds=tuple(args.distance_thresholds),
                primary_threshold=args.primary_threshold,
                min_cluster_voxels=args.min_cluster_voxels,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Compared {run['pair_count']} ETH Office pairs and wrote "
            f"{run['candidate_count']} geometric candidates to {args.output.resolve()}"
        )
    elif args.command == "review-eth-change":
        from visual_memory_lab.change_review import review_eth_changes

        try:
            summary = review_eth_changes(
                baseline=args.baseline,
                audit=args.audit,
                output=args.output,
                cache_dir=args.cache_dir,
                model=args.model,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Reviewed {summary['reviewed_candidate_count']} geometric candidates and accepted "
            f"{summary['accepted_pseudo_reference_count']} into the VLM pseudo-reference"
        )
    elif args.command == "localize-eth-objects":
        from visual_memory_lab.object_localization import localize_eth_objects

        try:
            summary = localize_eth_objects(
                dataset_root=args.input,
                output=args.output,
                keyframes_per_observation=args.keyframes_per_observation,
                device=args.device,
                detector_model=args.detector_model,
                segmenter_model=args.segmenter_model,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Localized {summary.detection_count} object predictions across "
            f"{summary.frame_count} ETH Office keyframes on {summary.device} in {summary.output}"
        )
    elif args.command == "audit-eth-object-localization":
        from visual_memory_lab.object_audit import audit_eth_object_localization

        try:
            summary = audit_eth_object_localization(
                localization=args.localization,
                output=args.output,
                cache_dir=args.cache_dir,
                frames_per_observation=args.frames_per_observation,
                model=args.model,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Pseudo-audited {summary['reviewed_detection_count']} detections across "
            f"{summary['frame_count']} fixed ETH Office frames"
        )
    return 0
