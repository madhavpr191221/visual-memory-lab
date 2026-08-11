"""Deterministic 3D change candidates from aligned ETH Office meshes."""

from __future__ import annotations

import json
import html
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import matplotlib
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class VoxelCloud:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    keys: np.ndarray
    source_counts: np.ndarray


def load_voxel_cloud(path: Path, *, voxel_size: float) -> VoxelCloud:
    if voxel_size <= 0:
        raise ValueError("voxel size must be positive")
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue", "normal_x", "normal_y", "normal_z"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"mesh is missing properties: {', '.join(missing)}")
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    colors = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(np.float64)
    normals = np.column_stack([vertex[name] for name in ("normal_x", "normal_y", "normal_z")]).astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError(f"mesh contains non-finite coordinates: {path}")
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    divisor = counts.astype(np.float64)

    def aggregate(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.bincount(inverse, weights=values[:, axis], minlength=len(counts)) / divisor for axis in range(3)]
        )

    aggregate_points = aggregate(points)
    aggregate_colors = aggregate(colors) / 255.0
    aggregate_normals = aggregate(normals)
    lengths = np.linalg.norm(aggregate_normals, axis=1, keepdims=True)
    aggregate_normals = np.divide(
        aggregate_normals,
        lengths,
        out=np.zeros_like(aggregate_normals),
        where=lengths > 1e-12,
    )
    return VoxelCloud(aggregate_points, aggregate_colors, aggregate_normals, unique_keys, counts)


def bidirectional_residuals(
    earlier: VoxelCloud, current: VoxelCloud
) -> dict[str, np.ndarray]:
    earlier_tree = cKDTree(earlier.points)
    current_tree = cKDTree(current.points)
    current_distances, current_matches = earlier_tree.query(current.points, workers=-1)
    earlier_distances, earlier_matches = current_tree.query(earlier.points, workers=-1)
    current_plane = np.abs(
        np.sum(
            (current.points - earlier.points[current_matches]) * earlier.normals[current_matches],
            axis=1,
        )
    )
    earlier_plane = np.abs(
        np.sum(
            (earlier.points - current.points[earlier_matches]) * current.normals[earlier_matches],
            axis=1,
        )
    )
    return {
        "current_distances": current_distances,
        "current_plane": current_plane,
        "earlier_distances": earlier_distances,
        "earlier_plane": earlier_plane,
    }


_NEIGHBORS = [offset for offset in product((-1, 0, 1), repeat=3) if offset != (0, 0, 0)]


def connected_components(keys: np.ndarray, *, min_voxels: int) -> list[np.ndarray]:
    if min_voxels < 1:
        raise ValueError("minimum cluster voxels must be positive")
    lookup = {tuple(key): index for index, key in enumerate(keys.tolist())}
    visited: set[int] = set()
    components: list[np.ndarray] = []
    for start in range(len(keys)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            key = keys[index]
            for offset in _NEIGHBORS:
                neighbor = lookup.get((int(key[0] + offset[0]), int(key[1] + offset[1]), int(key[2] + offset[2])))
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= min_voxels:
            components.append(np.asarray(component, dtype=np.int64))
    return components


def _candidate_records(
    *,
    cloud: VoxelCloud,
    distances: np.ndarray,
    thresholds: list[float],
    primary_threshold: float,
    min_cluster_voxels: int,
    pair_id: str,
    direction: str,
) -> tuple[list[dict[str, object]], np.ndarray]:
    changed = distances > primary_threshold
    changed_indices = np.flatnonzero(changed)
    components = connected_components(cloud.keys[changed], min_voxels=min_cluster_voxels)
    expanded = [changed_indices[component] for component in components]
    expanded.sort(
        key=lambda indices: (
            -len(indices),
            *np.mean(cloud.points[indices], axis=0).tolist(),
        )
    )
    labels = np.full(len(cloud.points), -1, dtype=np.int32)
    records: list[dict[str, object]] = []
    for cluster_index, indices in enumerate(expanded):
        labels[indices] = cluster_index
        points = cloud.points[indices]
        records.append(
            {
                "candidate_id": f"eth-office:{pair_id}:{direction}:cluster-{cluster_index:03d}",
                "pair_id": pair_id,
                "direction": direction,
                "cluster_index": cluster_index,
                "centroid_m": points.mean(axis=0).tolist(),
                "bounds_m": {"minimum": points.min(axis=0).tolist(), "maximum": points.max(axis=0).tolist()},
                "voxel_count": int(len(indices)),
                "source_point_count": int(cloud.source_counts[indices].sum()),
                "mean_distance_m": float(distances[indices].mean()),
                "maximum_distance_m": float(distances[indices].max()),
                "threshold_persistence": {
                    f"{threshold:.3f}": float(np.mean(distances[indices] > threshold))
                    for threshold in thresholds
                },
            }
        )
    return records, labels


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def _write_candidate_ply(path: Path, cloud: VoxelCloud, labels: np.ndarray) -> None:
    selected = np.flatnonzero(labels >= 0)
    vertex = np.empty(
        len(selected),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("cluster_index", "i4")],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertex[name] = cloud.points[selected, axis]
    palette = plt.get_cmap("tab20")((labels[selected] % 20) / 19.0)[:, :3]
    rgb = np.rint(palette * 255).astype(np.uint8)
    for axis, name in enumerate(("red", "green", "blue")):
        vertex[name] = rgb[:, axis]
    vertex["cluster_index"] = labels[selected]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _projection(
    *,
    base: VoxelCloud,
    changed: VoxelCloud,
    labels: np.ndarray,
    title: str,
    output: Path,
    review_limit: int = 6,
) -> None:
    projections = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    base_step = max(1, len(base.points) // 35_000)
    selected = np.flatnonzero(labels >= 0)
    for axis, (a, b, xlabel, ylabel) in zip(axes, projections, strict=True):
        axis.scatter(base.points[::base_step, a], base.points[::base_step, b], s=0.2, c="#c8c8c8", alpha=0.22)
        if len(selected):
            review_selected = selected[labels[selected] < review_limit]
            context_selected = selected[labels[selected] >= review_limit]
            if len(context_selected):
                axis.scatter(
                    changed.points[context_selected, a],
                    changed.points[context_selected, b],
                    s=0.7,
                    c="#d9a6a0",
                    alpha=0.18,
                )
            if len(review_selected):
                colors = plt.get_cmap("tab10")((labels[review_selected] % 10) / 9.0)
                axis.scatter(
                    changed.points[review_selected, a],
                    changed.points[review_selected, b],
                    s=3.0,
                    c=colors,
                    alpha=0.95,
                )
            for cluster_index in range(min(review_limit, int(labels.max()) + 1)):
                center = changed.points[labels == cluster_index].mean(axis=0)
                axis.text(center[a], center[b], str(cluster_index), fontsize=8, weight="bold")
        axis.set_xlabel(f"{xlabel} (m)")
        axis.set_ylabel(f"{ylabel} (m)")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle(title)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _write_report(pair_records: list[dict[str, object]], run: dict[str, object], output: Path) -> None:
    cards: list[str] = []
    for pair in pair_records:
        pair_id = html.escape(str(pair["pair_id"]))
        evidence = pair["evidence"]
        assert isinstance(evidence, dict)
        current = Path(str(evidence["current_only_png"])).relative_to(output).as_posix()
        earlier = Path(str(evidence["earlier_only_png"])).relative_to(output).as_posix()
        cards.append(
            f'<section><h2>{pair_id}</h2><p>{pair["current_only_candidate_count"]} current-only and '
            f'{pair["earlier_only_candidate_count"]} earlier-only clusters. Numbers mark the six largest review candidates.</p>'
            f'<div class="pair"><figure><a href="{current}"><img src="{current}"></a><figcaption>Current-only geometry</figcaption></figure>'
            f'<figure><a href="{earlier}"><img src="{earlier}"></a><figcaption>Earlier-only geometry</figcaption></figure></div></section>'
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ETH Office geometric change baseline</title><style>
body{{font:16px/1.5 system-ui;margin:0 auto;max-width:1500px;padding:32px;background:#f5f2e9;color:#20231f}}
h1,h2{{font-family:Georgia,serif}} section{{margin:36px 0}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
figure{{margin:0;background:white;border:1px solid #d8d2c4;border-radius:10px;overflow:hidden}} img{{display:block;width:100%}}
figcaption{{padding:10px 14px}} @media(max-width:850px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><h1>ETH Office geometric change baseline</h1>
<p>{run["candidate_count"]} geometric candidates across {run["pair_count"]} observation pairs. These are candidates, not verified physical changes.</p>
{"".join(cards)}</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def evaluate_eth_change(
    *,
    manifest_path: Path,
    output: Path,
    voxel_size: float = 0.02,
    distance_thresholds: tuple[float, ...] = (0.02, 0.05, 0.10),
    primary_threshold: float = 0.05,
    min_cluster_voxels: int = 20,
) -> dict[str, object]:
    thresholds = sorted(set(float(value) for value in distance_thresholds))
    if any(value <= 0 for value in thresholds):
        raise ValueError("distance thresholds must be positive")
    if primary_threshold not in thresholds:
        raise ValueError("primary threshold must be included in distance thresholds")
    resolved_output = output.resolve()
    if resolved_output.exists() and (not resolved_output.is_dir() or any(resolved_output.iterdir())):
        raise FileExistsError(f"output path is not empty: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = manifest.get("observations")
    if not isinstance(observations, list) or len(observations) != 4:
        raise ValueError("ETH manifest must contain four observations")
    clouds: list[VoxelCloud] = []
    for observation in observations:
        mesh = observation.get("mesh") if isinstance(observation, dict) else None
        if not isinstance(mesh, dict) or not isinstance(mesh.get("path"), str):
            raise ValueError("observation is missing a mesh path")
        clouds.append(load_voxel_cloud(Path(mesh["path"]), voxel_size=voxel_size))

    pair_records: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for earlier_index, current_index in combinations(range(4), 2):
        pair_id = f"{earlier_index}-to-{current_index}"
        earlier = clouds[earlier_index]
        current = clouds[current_index]
        residuals = bidirectional_residuals(earlier, current)
        current_candidates, current_labels = _candidate_records(
            cloud=current,
            distances=residuals["current_distances"],
            thresholds=thresholds,
            primary_threshold=primary_threshold,
            min_cluster_voxels=min_cluster_voxels,
            pair_id=pair_id,
            direction="current-only",
        )
        earlier_candidates, earlier_labels = _candidate_records(
            cloud=earlier,
            distances=residuals["earlier_distances"],
            thresholds=thresholds,
            primary_threshold=primary_threshold,
            min_cluster_voxels=min_cluster_voxels,
            pair_id=pair_id,
            direction="earlier-only",
        )
        candidates.extend(current_candidates)
        candidates.extend(earlier_candidates)
        pair_dir = resolved_output / pair_id
        pair_dir.mkdir()
        _write_candidate_ply(pair_dir / "current-only.ply", current, current_labels)
        _write_candidate_ply(pair_dir / "earlier-only.ply", earlier, earlier_labels)
        _projection(base=earlier, changed=current, labels=current_labels, title=f"{pair_id}: geometry only in current observation", output=pair_dir / "current-only.png")
        _projection(base=current, changed=earlier, labels=earlier_labels, title=f"{pair_id}: geometry only in earlier observation", output=pair_dir / "earlier-only.png")
        pair_records.append(
            {
                "pair_id": pair_id,
                "earlier_observation": earlier_index,
                "current_observation": current_index,
                "consecutive": current_index == earlier_index + 1,
                "current_only_candidate_count": len(current_candidates),
                "earlier_only_candidate_count": len(earlier_candidates),
                "point_to_point": {
                    "current_to_earlier": _distance_summary(residuals["current_distances"]),
                    "earlier_to_current": _distance_summary(residuals["earlier_distances"]),
                },
                "point_to_plane": {
                    "current_to_earlier": _distance_summary(residuals["current_plane"]),
                    "earlier_to_current": _distance_summary(residuals["earlier_plane"]),
                },
                "changed_fraction": {
                    f"{threshold:.3f}": {
                        "current_only": float(np.mean(residuals["current_distances"] > threshold)),
                        "earlier_only": float(np.mean(residuals["earlier_distances"] > threshold)),
                    }
                    for threshold in thresholds
                },
                "evidence": {
                    "current_only_png": str((pair_dir / "current-only.png").resolve()),
                    "earlier_only_png": str((pair_dir / "earlier-only.png").resolve()),
                    "current_only_ply": str((pair_dir / "current-only.ply").resolve()),
                    "earlier_only_ply": str((pair_dir / "earlier-only.ply").resolve()),
                },
            }
        )

    run = {
        "schema_version": 1,
        "manifest_path": str(manifest_path.resolve()),
        "voxel_size_m": voxel_size,
        "distance_thresholds_m": thresholds,
        "primary_threshold_m": primary_threshold,
        "min_cluster_voxels": min_cluster_voxels,
        "pair_count": len(pair_records),
        "candidate_count": len(candidates),
        "claim_boundary": "Geometric candidates are not verified physical changes.",
    }
    (resolved_output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (resolved_output / "pairs.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in pair_records), encoding="utf-8")
    (resolved_output / "candidates.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in candidates), encoding="utf-8")
    _write_report(pair_records, run, resolved_output)
    return run
