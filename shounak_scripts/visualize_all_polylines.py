#!/usr/bin/env python3
"""Visualize matched workzone/lane/ego polyline JSON files as static 3D images.

This script pairs files across three folders by a shared suffix key, renders one
3D scene per matched key using distinct colors per type, and saves one image per
scene.

Coordinate harmonization (default behavior):
- Workzones are read from KITTI camera coordinates (x-right, y-down, z-forward)
  and transformed to FRU (x-forward, y-right, z-up).
- Lanes are read from LATR ground coordinates (x-right, y-forward, z-up)
  and transformed to FRU.
- Ego trajectories are assumed FRU by default (or auto-detected from payload
  `output_frame` when available).

Additionally, for each rendered key, this script can project the 3D polylines
onto the corresponding image using KITTI calibration matrix P1 and save overlays
in `viz_polylines_image_proj`.

Supported JSON patterns:
- Workzones: polyline.sampled_xyz, or polyline.coeffs + polyline.z_range (cubic_x_of_z)
- Lanes: pred_laneLines_polyline3d[*].polyline3d, or cubic coeff fallback
- Ego poses: polyline.coeffs + polyline.x_range (cubic_y_of_x)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:  # Allows --help to work without runtime deps installed.
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Allows --help to work without runtime deps installed.
    plt = None


Extractor = Callable[[Dict, int], List["np.ndarray"]]


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


def _to_fru(points: np.ndarray, src_frame: str) -> np.ndarray:
    """Convert Nx3 points from src_frame into FRU (x-forward, y-right, z-up)."""
    if points.size == 0:
        return points

    if src_frame == "fru":
        return points
    if src_frame == "flu":
        # x-forward, y-left, z-up -> x-forward, y-right, z-up
        return np.stack([points[:, 0], -points[:, 1], points[:, 2]], axis=1)
    if src_frame == "rfu":
        # x-right, y-forward, z-up -> x-forward, y-right, z-up
        return np.stack([points[:, 1], points[:, 0], points[:, 2]], axis=1)
    if src_frame == "camera_kitti":
        # KITTI camera: x-right, y-down, z-forward -> FRU
        return np.stack([points[:, 2], points[:, 0], -points[:, 1]], axis=1)
    raise ValueError(f"Unsupported source frame: {src_frame}")


def _fru_to_camera_kitti(points: np.ndarray) -> np.ndarray:
    """Convert FRU (x-forward, y-right, z-up) to KITTI camera (x-right, y-down, z-forward)."""
    if points.size == 0:
        return points
    return np.stack([points[:, 1], -points[:, 2], points[:, 0]], axis=1)


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


def _match_triplets(workzone_dir: Path, lane_dir: Path, ego_dir: Path) -> List[Tuple[str, Optional[Path], Optional[Path], Optional[Path]]]:
    workzone = _collect_json_by_suffix(workzone_dir)
    lane = _collect_json_by_suffix(lane_dir)
    ego = _collect_json_by_suffix(ego_dir)
    keys = sorted(set(workzone.keys()) | set(lane.keys()) | set(ego.keys()))
    return [(k, workzone.get(k), lane.get(k), ego.get(k)) for k in keys]


def _parse_kitti_calib_p1(calib_path: Path) -> np.ndarray:
    try:
        lines = calib_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Failed to read calib file: {calib_path}") from exc

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() != "P1":
            continue
        vals = np.asarray([float(v) for v in value.strip().split()], dtype=np.float64)
        if vals.size != 12:
            raise ValueError(f"P1 must contain 12 values in {calib_path}")
        return vals.reshape(3, 4)

    raise ValueError(f"P1 not found in calib file: {calib_path}")


def _find_file_for_key(folder: Path, key: str, suffixes: List[str]) -> Optional[Path]:
    for suffix in suffixes:
        p = folder / f"{key}{suffix}"
        if p.exists():
            return p

    for suffix in suffixes:
        matches = sorted(folder.glob(f"*{key}*{suffix}"))
        if matches:
            return matches[0]

    return None


def _project_camera_points(p1: np.ndarray, points_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz1 = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1), dtype=np.float64)], axis=1)
    uvw = (p1 @ xyz1.T).T
    valid = uvw[:, 2] > 1e-6

    uv = np.full((points_camera.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = uvw[valid, 0] / uvw[valid, 2]
    uv[valid, 1] = uvw[valid, 1] / uvw[valid, 2]
    return uv, valid


def _draw_projected_polyline(image: np.ndarray, uv: np.ndarray, valid_mask: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int) -> None:
    h, w = image.shape[:2]
    segment: List[Tuple[int, int]] = []

    def flush_segment() -> None:
        if len(segment) >= 2:
            pts = np.asarray(segment, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], isClosed=False, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)

    for idx in range(len(uv)):
        if not valid_mask[idx] or np.isnan(uv[idx, 0]) or np.isnan(uv[idx, 1]):
            flush_segment()
            segment = []
            continue

        x, y = int(round(uv[idx, 0])), int(round(uv[idx, 1]))
        if 0 <= x < w and 0 <= y < h:
            segment.append((x, y))
        else:
            flush_segment()
            segment = []

    flush_segment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matched workzone/lane/ego polyline JSON files and save one image per match."
    )
    parser.add_argument("--workzone-dir", type=Path, required=True, help="Folder with workzone polyline JSON files.")
    parser.add_argument("--lane-dir", type=Path, required=True, help="Folder with lane polyline JSON files.")
    parser.add_argument("--ego-dir", type=Path, required=True, help="Folder with ego-pose polyline JSON files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save rendered 3D images.")
    parser.add_argument("--max-files", type=int, default=200, help="Render first N matched suffix keys (default: 50).")
    parser.add_argument("--output-prefix", type=str, default="poly3d", help="Output image filename prefix.")
    parser.add_argument("--samples", type=int, default=60, help="Samples for coeff-only polylines (default: 60).")
    parser.add_argument("--width", type=int, default=1920, help="Render width (default: 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Render height (default: 1080).")
    parser.add_argument("--line-width", type=float, default=2.0, help="Polyline width in output image (default: 2.0).")
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Deprecated alias for --line-width, kept for backward compatibility.",
    )
    parser.add_argument(
        "--workzone-frame",
        choices=["camera_kitti", "fru", "flu", "rfu"],
        default="camera_kitti",
        help="Frame of workzone polylines before conversion to FRU (default: camera_kitti).",
    )
    parser.add_argument(
        "--lane-frame",
        choices=["rfu", "fru", "flu", "camera_kitti"],
        default="rfu",
        help="Frame of lane polylines before conversion to FRU (default: rfu).",
    )
    parser.add_argument(
        "--ego-frame",
        choices=["auto", "fru", "flu", "rfu", "camera_kitti"],
        default="auto",
        help="Frame of ego polylines before conversion to FRU (default: auto from payload.output_frame).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("../dataset/WorkZone3D/image_1"),
        help="Directory with source images for 3D-to-2D projection overlays.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("../dataset/WorkZone3D/calib"),
        help="Directory with KITTI calibration txt files. P1 is used for projection.",
    )
    parser.add_argument(
        "--image-proj-output-dir",
        type=Path,
        default=Path("viz_polylines_image_proj"),
        help="Directory to save projected image overlays (default: viz_polylines_image_proj).",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image extensions used to find source images.",
    )
    return parser.parse_args()


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    ranges = maxs - mins
    radius = max(float(np.max(ranges)) / 2.0, 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _add_lines(ax, polys: List[np.ndarray], color: np.ndarray, line_width: float, label: str) -> None:
    for idx, pts in enumerate(polys):
        if len(pts) < 2:
            continue
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            linewidth=line_width,
            label=label if idx == 0 else None,
        )


def _ego_frame_for_payload(payload: Dict, cli_frame: str) -> str:
    if cli_frame != "auto":
        return cli_frame
    frame = str(payload.get("output_frame", "fru")).lower()
    if frame in {"fru", "flu", "rfu", "camera_kitti"}:
        return frame
    return "fru"


def _load_polys_with_frame(
    json_path: Optional[Path],
    extractor: Extractor,
    samples: int,
    src_frame: str,
    ego_auto: bool = False,
) -> List[np.ndarray]:
    if json_path is None:
        return []
    payload = _safe_json(json_path)
    polys = extractor(payload, samples=samples)
    if not polys:
        return []
    if ego_auto:
        src_frame = _ego_frame_for_payload(payload, src_frame)
    return [_to_fru(pts, src_frame) for pts in polys]


def main() -> None:
    args = parse_args()

    if np is None:
        raise ImportError("numpy is required. Please install numpy before running this script.")
    if plt is None:
        raise ImportError("matplotlib is required. Please install matplotlib before running this script.")
    if cv2 is None:
        raise ImportError("opencv-python is required for image projection overlays.")

    for d in [args.workzone_dir, args.lane_dir, args.ego_dir]:
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

    if args.max_files < 1:
        raise ValueError("--max-files must be >= 1")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be >= 1")

    line_width = float(args.point_size) if args.point_size is not None else float(args.line_width)
    if line_width <= 0.0:
        raise ValueError("--line-width/--point-size must be > 0")

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_proj_output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0

    for key, wz_path, lane_path, ego_path in matches:
        workzone_polys = _load_polys_with_frame(
            wz_path,
            _extract_workzone,
            args.samples,
            src_frame=args.workzone_frame,
            ego_auto=False,
        )
        lane_polys = _load_polys_with_frame(
            lane_path,
            _extract_lanes,
            args.samples,
            src_frame=args.lane_frame,
            ego_auto=False,
        )
        ego_polys = _load_polys_with_frame(
            ego_path,
            _extract_ego,
            args.samples,
            src_frame=args.ego_frame,
            ego_auto=True,
        )

        if not (workzone_polys or lane_polys or ego_polys):
            continue

        fig = plt.figure(figsize=(args.width / 100.0, args.height / 100.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        _add_lines(ax, workzone_polys, colors["workzone"], line_width, label="workzone")
        _add_lines(ax, lane_polys, colors["lane"], line_width, label="lane")
        _add_lines(ax, ego_polys, colors["ego"], line_width, label="ego")

        all_points = np.concatenate(workzone_polys + lane_polys + ego_polys, axis=0)
        _set_equal_3d_axes(ax, all_points)

        ax.view_init(elev=25, azim=-90)
        ax.grid(False)
        ax.set_xlabel("X_forward", color="white")
        ax.set_ylabel("Y_right", color="white")
        ax.set_zlabel("Z_up", color="white")
        ax.tick_params(colors="white")
        ax.set_title(f"Workzone/Lane/Ego 3D Polylines [{key}] (FRU)", color="white")

        legend = ax.legend(facecolor="black", edgecolor="white")
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("white")

        out_img = args.output_dir / f"{args.output_prefix}_{key}.png"
        fig.savefig(out_img, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        image_path = _find_file_for_key(args.image_dir, key, args.image_exts)
        calib_path = _find_file_for_key(args.calib_dir, key, [".txt"])
        if image_path is not None and calib_path is not None:
            image = cv2.imread(str(image_path))
            if image is not None:
                try:
                    p1 = _parse_kitti_calib_p1(calib_path)
                except ValueError:
                    p1 = None

                if p1 is not None:
                    thickness = max(1, int(round(line_width)))
                    bgr = {
                        "workzone": tuple(int(255 * c) for c in colors["workzone"][::-1]),
                        "lane": tuple(int(255 * c) for c in colors["lane"][::-1]),
                        "ego": tuple(int(255 * c) for c in colors["ego"][::-1]),
                    }

                    for poly in workzone_polys:
                        cam_pts = _fru_to_camera_kitti(poly)
                        uv, valid = _project_camera_points(p1, cam_pts)
                        _draw_projected_polyline(image, uv, valid, bgr["workzone"], thickness)

                    for poly in lane_polys:
                        cam_pts = _fru_to_camera_kitti(poly)
                        uv, valid = _project_camera_points(p1, cam_pts)
                        _draw_projected_polyline(image, uv, valid, bgr["lane"], thickness)

                    for poly in ego_polys:
                        cam_pts = _fru_to_camera_kitti(poly)
                        uv, valid = _project_camera_points(p1, cam_pts)
                        _draw_projected_polyline(image, uv, valid, bgr["ego"], thickness)

                    out_proj = args.image_proj_output_dir / f"{args.output_prefix}_{key}.png"
                    cv2.imwrite(str(out_proj), image)

        rendered += 1
        print(
            f"Saved 3D image: {out_img} | key={key} | "
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
