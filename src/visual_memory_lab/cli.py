"""Command-line interface for Visual Memory Lab."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from visual_memory_lab import __version__
from visual_memory_lab.memory import MemoryIndex, build_index, ensure_matching_encoder


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
        description="Build and evaluate real-world visual memories.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser(
        "index",
        help="build a frozen CLIP index over an image-memory artifact",
    )
    index.add_argument("--input", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--device", default="auto")
    index.add_argument("--batch-size", type=_positive_integer, default=64)

    query = subparsers.add_parser(
        "query",
        help="retrieve observations from a real-world visual-memory index",
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

    charades = subparsers.add_parser(
        "prepare-charades",
        help="write a deterministic Charades train/test subset manifest",
    )
    charades.add_argument("--input", type=Path, required=True)
    charades.add_argument("--output", type=Path, required=True)
    charades.add_argument("--train-limit", type=_positive_integer, default=300)
    charades.add_argument("--test-limit", type=_positive_integer, default=100)
    charades.add_argument("--seed", type=_non_negative_integer, default=42)

    windows = subparsers.add_parser(
        "build-charades-windows",
        help="turn a Charades manifest into timestamped temporal windows",
    )
    windows.add_argument("--manifest", type=Path, required=True)
    windows.add_argument("--output", type=Path, required=True)
    windows.add_argument("--window-seconds", type=_positive_float, default=4.0)
    windows.add_argument("--stride-seconds", type=_positive_float, default=2.0)

    learned = subparsers.add_parser(
        "build-charades-frames",
        help="write deterministic frame timestamps for learned Charades windows",
    )
    learned.add_argument("--manifest", type=Path, required=True)
    learned.add_argument("--output", type=Path, required=True)
    learned.add_argument("--frames-per-window", type=_positive_integer, default=16)

    cache = subparsers.add_parser(
        "build-charades-video-cache",
        help="decode Charades windows and cache frozen CLIP frame/text embeddings",
    )
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--batch-size", type=_positive_integer, default=16)
    cache.add_argument("--max-videos", type=_positive_integer)
    cache.add_argument("--workers", type=_positive_integer, default=4)
    cache.add_argument("--resume", action="store_true")

    train_video = subparsers.add_parser(
        "train-charades-video",
        help="train the temporal head on cached Charades video embeddings",
    )
    train_video.add_argument("--cache", type=Path, required=True)
    train_video.add_argument("--output", type=Path, required=True)
    train_video.add_argument("--device", default="auto")
    train_video.add_argument("--epochs", type=_positive_integer, default=3)
    train_video.add_argument("--batch-size", type=_positive_integer, default=32)
    train_video.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    train_video.add_argument("--finetune-vision-blocks", type=_non_negative_integer, default=0)
    train_video.add_argument("--action-weight", type=_positive_float, default=1.0)
    train_video.add_argument("--boundary-weight", type=_positive_float, default=2.0)
    train_video.add_argument("--split", choices=("train", "test"), default="train")

    index_video = subparsers.add_parser(
        "index-charades-video",
        help="build an exact learned temporal index from a cache and head checkpoint",
    )
    index_video.add_argument("--cache", type=Path, required=True)
    index_video.add_argument("--checkpoint", type=Path)
    index_video.add_argument("--output", type=Path, required=True)
    index_video.add_argument("--device", default="auto")
    index_video.add_argument("--split", choices=("train", "test", "all"), default="train")

    evaluate_video = subparsers.add_parser(
        "evaluate-charades-video",
        help="evaluate learned Charades retrieval against official action intervals",
    )
    evaluate_video.add_argument("--index", type=Path, required=True)
    evaluate_video.add_argument("--test-manifest", type=Path, required=True)
    evaluate_video.add_argument("--output", type=Path, required=True)
    evaluate_video.add_argument("--device", default="auto")

    multimodal = subparsers.add_parser(
        "audit-multimodal-manifest",
        help="validate a multimodal JSONL manifest and report available sensors",
    )
    multimodal.add_argument("--input", type=Path, required=True)
    multimodal.add_argument("--output", type=Path, required=True)

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

    benchmark = subparsers.add_parser(
        "evaluate-technician-benchmark",
        help="evaluate the authored office technician-style question set",
    )
    benchmark.add_argument(
        "--questions", type=Path, default=Path("data/phase7/technician_questions.jsonl")
    )
    benchmark.add_argument(
        "--memory-index", type=Path, default=Path("outputs/phase3/train-index")
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument(
        "--eth-localization", type=Path, default=Path("outputs/phase6b1/object-localization")
    )
    benchmark.add_argument(
        "--eth-rgbd", type=Path, default=Path("outputs/phase612/rgbd-evidence")
    )
    benchmark.add_argument(
        "--eth-associations", type=Path, default=Path("outputs/phase613/associations")
    )

    prepare_eth = subparsers.add_parser(
        "prepare-eth-office",
        help="audit ETH Office bags and meshes and create a browsable RGB gallery",
    )
    prepare_eth.add_argument("--input", type=Path, required=True)
    prepare_eth.add_argument("--output", type=Path, required=True)
    prepare_eth.add_argument("--rgb-samples", type=_positive_integer, default=24)
    prepare_eth.add_argument("--vlm-samples", type=_positive_integer, default=8)

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

    rgbd = subparsers.add_parser(
        "build-eth-rgbd-evidence",
        help="link frozen ETH Office masks to recorded RGB-coloured point clouds",
    )
    rgbd.add_argument("--input", type=Path, required=True)
    rgbd.add_argument("--localization", type=Path, required=True)
    rgbd.add_argument("--output", type=Path, required=True)

    associate = subparsers.add_parser(
        "associate-eth-objects",
        help="rank cautious same-object candidates across ETH Office visits",
    )
    associate.add_argument("--localization", type=Path, required=True)
    associate.add_argument("--rgbd-evidence", type=Path, required=True)
    associate.add_argument("--output", type=Path, required=True)
    associate.add_argument("--device", default="auto")
    associate.add_argument("--top-per-group", type=_positive_integer, default=200)

    audit_associate = subparsers.add_parser(
        "audit-eth-object-associations",
        help="VLM pseudo-audit the highest-ranked cross-visit candidates",
    )
    audit_associate.add_argument("--associations", type=Path, required=True)
    audit_associate.add_argument("--localization", type=Path, required=True)
    audit_associate.add_argument("--output", type=Path, required=True)
    audit_associate.add_argument("--cache-dir", type=Path, default=Path("outputs/phase613/vlm-cache"))
    audit_associate.add_argument("--model", default="gpt-5.6-terra")
    audit_associate.add_argument("--limit", type=_positive_integer, default=200)

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
    serve.add_argument(
        "--object-localization",
        type=Path,
        default=Path("outputs/phase6b1/object-localization"),
    )
    serve.add_argument(
        "--object-audit", type=Path, default=Path("outputs/phase6b1/vlm-audit")
    )
    serve.add_argument(
        "--rgbd-evidence", type=Path, default=Path("outputs/phase612/rgbd-evidence")
    )
    serve.add_argument(
        "--associations", type=Path, default=Path("outputs/phase613/associations")
    )
    serve.add_argument(
        "--association-audit", type=Path, default=Path("outputs/phase613/vlm-audit")
    )
    serve.add_argument(
        "--technician-questions",
        type=Path,
        default=Path("data/phase7/technician_questions.jsonl"),
    )
    serve.add_argument(
        "--technician-output",
        type=Path,
        default=Path("outputs/phase7/technician-benchmark"),
    )
    serve.add_argument(
        "--charades-windows",
        type=Path,
        default=Path("outputs/charades/learned/windows/windows.jsonl"),
    )
    serve.add_argument(
        "--charades-learned-index",
        type=Path,
        default=Path("outputs/charades/learned/full/index"),
    )
    serve.add_argument("--inspection-db", type=Path, default=Path("outputs/phase8/inspections.sqlite3"))
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
    if args.command == "index":
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
    elif args.command == "prepare-charades":
        from visual_memory_lab.charades import prepare_charades_dataset

        try:
            summary = prepare_charades_dataset(
                dataset_root=args.input,
                output=args.output,
                train_limit=args.train_limit,
                test_limit=args.test_limit,
                seed=args.seed,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Prepared {summary['train_count']} train and {summary['test_count']} "
            f"test Charades videos in {args.output.resolve()}"
        )
    elif args.command == "build-charades-windows":
        from visual_memory_lab.charades import build_temporal_windows

        try:
            summary = build_temporal_windows(
                manifest=args.manifest,
                output=args.output,
                window_s=args.window_seconds,
                stride_s=args.stride_seconds,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Built {summary['window_count']} Charades temporal windows in "
            f"{args.output.resolve()}"
        )
    elif args.command == "build-charades-frames":
        from visual_memory_lab.learned_video import build_frame_manifest

        try:
            summary = build_frame_manifest(
                windows_manifest=args.manifest,
                output=args.output,
                frames_per_window=args.frames_per_window,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(f"Wrote {summary['window_count']} frame records with {summary['frames_per_window']} frames per window in {args.output.resolve()}")
    elif args.command == "build-charades-video-cache":
        from visual_memory_lab.learned_video import build_embedding_cache

        try:
            summary = build_embedding_cache(
                args.manifest,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
                max_videos=args.max_videos,
                workers=args.workers,
                resume=args.resume,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Cached CLIP embeddings for {summary['window_count']} windows on {summary['device']} in {args.output.resolve()}")
    elif args.command == "train-charades-video":
        from visual_memory_lab.learned_video import train_temporal_from_cache

        if args.finetune_vision_blocks:
            parser.error("vision fine-tuning requires raw-window training and is reserved for the next training subphase; use --finetune-vision-blocks 0")
        try:
            summary = train_temporal_from_cache(
                args.cache,
                args.output,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                split=args.split,
                action_weight=args.action_weight,
                boundary_weight=args.boundary_weight,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Trained the temporal head for {summary['epochs']} epochs on {summary['device']} in {args.output.resolve()}")
    elif args.command == "index-charades-video":
        from visual_memory_lab.learned_video import build_index_from_cache

        try:
            summary = build_index_from_cache(
                args.cache,
                args.checkpoint,
                args.output,
                device=args.device,
                split=args.split,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Indexed {summary['count']} learned video windows in {args.output.resolve()}")
    elif args.command == "evaluate-charades-video":
        from visual_memory_lab.learned_video import evaluate_index

        try:
            metrics = evaluate_index(
                args.index,
                args.test_manifest,
                args.output,
                device=args.device,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Evaluated {metrics['query_count']} video queries; "
            f"Recall@1={metrics['recall_at_k']['1']:.4f}, "
            f"Recall@5={metrics['recall_at_k']['5']:.4f}, "
            f"Recall@10={metrics['recall_at_k']['10']:.4f} in {args.output.resolve()}"
        )
    elif args.command == "audit-multimodal-manifest":
        from visual_memory_lab.multimodal import audit_records, load_records

        try:
            records = load_records(args.input)
            summary = audit_records(records)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Audited {summary['record_count']} recordings; "
            f"{summary['all_modalities_count']} contain RGB, audio, depth, and pose "
            f"in {args.output.resolve()}"
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
                object_localization=args.object_localization,
                object_audit=args.object_audit,
                rgbd_evidence=args.rgbd_evidence,
                associations=args.associations,
                association_audit=args.association_audit,
                technician_questions=args.technician_questions,
                technician_output=args.technician_output,
                inspection_db=args.inspection_db,
                charades_windows=args.charades_windows,
                charades_learned_index=args.charades_learned_index,
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
    elif args.command == "evaluate-technician-benchmark":
        from visual_memory_lab.technician_benchmark import write_benchmark
        from visual_memory_lab.encoder import ClipEncoder

        try:
            memory = MemoryIndex.load(args.memory_index)
            encoder = ClipEncoder(device=args.device)
            ensure_matching_encoder(memory, encoder)
            zone_payload = json.loads(Path("artifacts/phase3/office-zones.json").read_text(encoding="utf-8"))
            assignments = zone_payload.get("assignments", {})

            def search(source_observation_id: str) -> list[dict[str, object]]:
                vector = memory.observation_embedding(source_observation_id)
                payloads = []
                for result in memory.search(vector, top_k=10, exclude_observation_id=source_observation_id):
                    observation = result.observation
                    observation_id = str(observation["observation_id"])
                    payloads.append({
                        "rank": result.rank,
                        "observation_id": observation_id,
                        "zone_slug": assignments.get(observation_id),
                        "visit_id": observation.get("sequence_id", observation.get("episode_id")),
                    })
                return payloads

            artifacts = {
                label
                for label, path in {
                    "localization": args.eth_localization / "detections.jsonl",
                    "rgbd": args.eth_rgbd / "evidence.jsonl",
                    "association": args.eth_associations / "associations.jsonl",
                }.items()
                if path.is_file()
            }
            artifact_records: dict[str, list[dict[str, object]]] = {}
            localization_path = args.eth_localization / "detections.jsonl"
            if localization_path.is_file():
                artifact_records["localization"] = [
                    {**json.loads(line), "artifact_id": json.loads(line).get("detection_id")}
                    for line in localization_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            rgbd_path = args.eth_rgbd / "evidence.jsonl"
            if rgbd_path.is_file():
                artifact_records["rgbd"] = [
                    {**json.loads(line), "artifact_id": json.loads(line).get("detection_id")}
                    for line in rgbd_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            association_path = args.eth_associations / "associations.jsonl"
            if association_path.is_file():
                association_records = []
                for line in association_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    for key in ("earlier_detection_id", "later_detection_id"):
                        detection_id = str(record.get(key, ""))
                        if detection_id:
                            record_copy = dict(record)
                            record_copy["frame_id"] = ":".join(detection_id.split(":")[:-1])
                            record_copy["object_class"] = record.get("object_class")
                            association_records.append(record_copy)
                artifact_records["association"] = association_records
            payload = write_benchmark(
                questions_path=args.questions,
                output=args.output,
                search=search,
                available_artifacts=artifacts,
                artifact_records=artifact_records,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Evaluated {payload['question_count']} technician questions; "
            f"evidence recall={payload['evidence_recall']} in {args.output.resolve()}"
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
    elif args.command == "build-eth-rgbd-evidence":
        from visual_memory_lab.rgbd_evidence import build_rgbd_evidence

        try:
            summary = build_rgbd_evidence(
                dataset_root=args.input,
                localization=args.localization,
                output=args.output,
            )
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Built {summary['evidence_count']} RGB-D evidence records; "
            f"{summary['nonempty_evidence_count']} contain point-cloud evidence "
            f"in {args.output.resolve()}"
        )
    elif args.command == "associate-eth-objects":
        from visual_memory_lab.object_association import associate_eth_objects

        try:
            summary = associate_eth_objects(
                localization=args.localization,
                rgbd_evidence=args.rgbd_evidence,
                output=args.output,
                device=args.device,
                top_per_group=args.top_per_group,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Ranked {summary['pair_count']} cross-visit candidates from "
            f"{summary['detection_count']} detections on {summary['device']}"
        )
    elif args.command == "audit-eth-object-associations":
        from visual_memory_lab.association_audit import audit_associations

        try:
            summary = audit_associations(
                associations=args.associations,
                localization=args.localization,
                output=args.output,
                cache_dir=args.cache_dir,
                model=args.model,
                limit=args.limit,
            )
        except (FileExistsError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Pseudo-audited {summary['reviewed_count']} association candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
