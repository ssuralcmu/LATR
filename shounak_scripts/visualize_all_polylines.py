#!/usr/bin/env python3
"""Visualize matched workzone/lane/ego polyline JSON files in Open3D.

This script pairs files across three folders by a shared suffix key, renders one
Open3D scene per matched key using distinct colors per type, and saves one image
per scene.

Supported JSON patterns (inferred from existing shounak_scripts):
- Workzones: polyline.sampled_xyz, or polyline.coeffs + polyline.z_range (cubic_x_of_z)
- Lanes: pred_laneLines_polyline3d[*].polyline3d, or cubic coeff fallback
- Ego poses: polyline.coeffs + polyline.x_range (cubic_y_of_x)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # Allows --help to work without runtime deps installed.
    np = None


def _sample_cubic(coeffs: Iterable[float], t_range: Iterable[float], n: int) -> np.ndarray:
    if np is None:
        raise ImportError("numpy is required to sample cubic polylines.")
    t0, t1 = [float(v) for v in t_range]
    ts = np.linspace(t0, t1, max(2, n), dtype=np.float64)
    vals = np.polyval(np.asarray(list(coeffs), dtype=np.float64), ts)
    return np.stack([ts, vals], axis=1)


def _safe_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_workzone(payload: Dict, samples: int) -> List[np.ndarray]:
    if np is None:
        raise ImportError("numpy is required to parse workzone polylines.")
    out: List[np.ndarray] = []
    poly = payload.get("polyline", {}) if isinstance(payload, dict) else {}

    sampled = poly.get("sampled_xyz")
    if isinstance(sampled, list) and sampled:
        arr = np.asarray(sampled, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 2:
            out.append(arr)
            return out

    coeffs = poly.get("coeffs")
    z_range = poly.get("z_range")
    if isinstance(coeffs, list) and len(coeffs) == 4 and isinstance(z_range, list) and len(z_range) == 2:
        # cubic_x_of_z: x = f(z); use y=0 when not explicitly available.
        zv_xv = _sample_cubic(coeffs, z_range, samples)
        z = zv_xv[:, 0]
        x = zv_xv[:, 1]
        y = np.zeros_like(z)
        out.append(np.stack([x, y, z], axis=1))
    return out


def _extract_lanes(payload: Dict, samples: int) -> List[np.ndarray]:
    if np is None:
        raise ImportError("numpy is required to parse lane polylines.")
    out: List[np.ndarray] = []
    lanes = payload.get("pred_laneLines_polyline3d", []) if isinstance(payload, dict) else []
    if not isinstance(lanes, list):
        return out

    for lane in lanes:
        if not isinstance(lane, dict):
            continue

        polyline3d = lane.get("polyline3d")
        if isinstance(polyline3d, list) and polyline3d:
            arr = np.asarray(polyline3d, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 2:
                out.append(arr)
                continue

        coeffs = lane.get("poly_coeff_y_of_x")
        x_range = lane.get("x_range")
        if isinstance(coeffs, list) and len(coeffs) == 4 and isinstance(x_range, list) and len(x_range) == 2:
            xy = _sample_cubic(coeffs, x_range, samples)
            x = xy[:, 0]
            y = xy[:, 1]
            z = np.zeros_like(x)
            out.append(np.stack([x, y, z], axis=1))

    return out


def _extract_ego(payload: Dict, samples: int) -> List[np.ndarray]:
    if np is None:
        raise ImportError("numpy is required to parse ego polylines.")
    out: List[np.ndarray] = []
    poly = payload.get("polyline", {}) if isinstance(payload, dict) else {}

    sampled = poly.get("sampled_xyz")
    if isinstance(sampled, list) and sampled:
        arr = np.asarray(sampled, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 2:
            out.append(arr)
            return out

    coeffs = poly.get("coeffs")
    x_range = poly.get("x_range")
    if isinstance(coeffs, list) and len(coeffs) == 4 and isinstance(x_range, list) and len(x_range) == 2:
        # cubic_y_of_x: y = f(x); use z=0 for ego path in vehicle plane.
        xy = _sample_cubic(coeffs, x_range, samples)
        x = xy[:, 0]
        y = xy[:, 1]
        z = np.zeros_like(x)
        out.append(np.stack([x, y, z], axis=1))

    return out


def _file_key(path: Path) -> str:
    """Return matching key from filename suffix.

    Example:
      abc_000123.json -> 000123
      000123.json     -> 000123
    """
    stem = path.stem
    return stem.rsplit("_", 1)[-1]


def _collect_json_by_suffix(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(folder.glob("*.json")):
        out[_file_key(p)] = p
    return out


def _polys_from_file(json_path: Optional[Path], extractor, samples: int) -> List[np.ndarray]:
    if json_path is None:
        return []
    payload = _safe_json(json_path)
    return extractor(payload, samples=samples)


def _match_triplets(workzone_dir: Path, lane_dir: Path, ego_dir: Path) -> List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    workzone = _collect_json_by_suffix(workzone_dir)
    lane = _collect_json_by_suffix(lane_dir)
    ego = _collect_json_by_suffix(ego_dir)
    keys = sorted(set(workzone.keys()) | set(lane.keys()) | set(ego.keys()))
    return [(k, workzone.get(k), lane.get(k), ego.get(k)) for k in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matched workzone/lane/ego polyline JSON files and save one image per match."
    )
    parser.add_argument("--workzone-dir", type=Path, required=True, help="Folder with workzone polyline JSON files.")
    parser.add_argument("--lane-dir", type=Path, required=True, help="Folder with lane polyline JSON files.")
    parser.add_argument("--ego-dir", type=Path, required=True, help="Folder with ego-pose polyline JSON files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save rendered images.")
    parser.add_argument("--max-files", type=int, default=50, help="Render first N matched suffix keys (default: 50).")
    parser.add_argument("--output-prefix", type=str, default="poly3d", help="Output image filename prefix.")
    parser.add_argument("--samples", type=int, default=60, help="Samples for coeff-only polylines (default: 60).")
    parser.add_argument("--width", type=int, default=1920, help="Render width (default: 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Render height (default: 1080).")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size (default: 3.0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if np is None:
        raise ImportError("numpy is required. Please install numpy before running this script.")
    import open3d as o3d

    for d in [args.workzone_dir, args.lane_dir, args.ego_dir]:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

    if args.max_files < 1:
        raise ValueError("--max-files must be >= 1")

    matches = _match_triplets(args.workzone_dir, args.lane_dir, args.ego_dir)
    if not matches:
        raise RuntimeError("No JSON files found to match across folders.")

    matches = matches[: args.max_files]

    # RGB colors: workzone=red, lane=green, ego=blue
    colors = {
        "workzone": np.array([1.0, 0.2, 0.2], dtype=np.float64),
        "lane": np.array([0.2, 0.9, 0.2], dtype=np.float64),
        "ego": np.array([0.2, 0.4, 1.0], dtype=np.float64),
    }

    def add_lines(geometries: List, polys: List[np.ndarray], color: np.ndarray) -> None:
        for pts in polys:
            if len(pts) < 2:
                continue
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
            geometries.append(ls)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0

    for key, wz_path, lane_path, ego_path in matches:
        workzone_polys = _polys_from_file(wz_path, _extract_workzone, args.samples)
        lane_polys = _polys_from_file(lane_path, _extract_lanes, args.samples)
        ego_polys = _polys_from_file(ego_path, _extract_ego, args.samples)

        if not (workzone_polys or lane_polys or ego_polys):
            continue

        geometries = []
        add_lines(geometries, workzone_polys, colors["workzone"])
        add_lines(geometries, lane_polys, colors["lane"])
        add_lines(geometries, ego_polys, colors["ego"])
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Workzone/Lane/Ego 3D Polylines [{key}]",
            width=args.width,
            height=args.height,
            visible=True,
        )
        for g in geometries:
            vis.add_geometry(g)

        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray([0.0, 0.0, 0.0])
        render_opt.point_size = float(args.point_size)

        vis.poll_events()
        vis.update_renderer()

        ctr = vis.get_view_control()
        ctr.set_front([0.0, -1.0, -0.3])
        ctr.set_lookat([0.0, 0.0, 15.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.45)

        vis.poll_events()
        vis.update_renderer()

        out_img = args.output_dir / f"{args.output_prefix}_{key}.png"
        ok = vis.capture_screen_image(str(out_img), do_render=True)
        vis.destroy_window()

        if not ok:
            raise RuntimeError(f"Failed to save image to {out_img}")

        rendered += 1
        print(
            f"Saved image: {out_img} | key={key} | "
            f"workzone={len(workzone_polys)}, lanes={len(lane_polys)}, ego={len(ego_polys)}"
        )

    if rendered == 0:
        raise RuntimeError("No matched files produced valid polylines.")

    print(
        f"Rendered {rendered} image(s) from first {len(matches)} matched key(s). "
        f"colors: workzone={colors['workzone'].tolist()}, lane={colors['lane'].tolist()}, ego={colors['ego'].tolist()}"
    )


if __name__ == "__main__":
    main()
