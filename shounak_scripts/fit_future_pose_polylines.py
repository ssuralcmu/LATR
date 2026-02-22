#!/usr/bin/env python3
"""Fit cubic polylines from generated future ego-pose JSON files.

For each input future-pose JSON:
- Collect (x, y) from future_poses relative_position_xyz.
- Keep points up to max Euclidean distance from origin (default 25m).
- Fit cubic y(x) using provided smoothing/interpolation logic.
- Write one output JSON per input file.

If no usable points are available, writes {}.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from scipy.signal import savgol_filter


# ---------------------------
# Fit cubic y(x)
# ---------------------------
def fit_poly_vehicle_smooth(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 1:
        return None, None

    pts = pts[np.argsort(pts[:, 0])]

    s = np.zeros(len(pts), dtype=np.float64)
    s[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    if s[-1] < 1e-6:
        return None, None

    s_uniform = np.linspace(0, s[-1], len(pts))
    x_interp = np.interp(s_uniform, s, pts[:, 0])
    y_interp = np.interp(s_uniform, s, pts[:, 1])

    if len(y_interp) > 9 and savgol_filter is not None:
        y_interp = savgol_filter(y_interp, 9, 3)

    if len(x_interp) < 4:
        return None, None
    coeffs = np.polyfit(x_interp, y_interp, 3)
    x_range = [float(np.min(x_interp)), float(np.max(x_interp))]
    return coeffs.tolist(), x_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit cubic polylines from future ego-pose JSONs.")
    parser.add_argument("--future-json-dir", required=True, help="Directory containing generated future-pose JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save fitted polyline JSON files.")
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=25.0,
        help="Use future points with sqrt(x^2+y^2) <= this distance (meters).",
    )
    parser.add_argument("--input-glob", default="*.json")
    return parser.parse_args()


def _collect_points_within_distance(payload: Dict, max_distance_m: float) -> np.ndarray:
    points: List[List[float]] = [[0.0, 0.0]]
    for item in payload.get("future_poses", []):
        pos = item.get("relative_position_xyz", {})
        if not all(k in pos for k in ("x", "y")):
            continue
        x, y = float(pos["x"]), float(pos["y"])
        if np.hypot(x, y) <= max_distance_m:
            points.append([x, y])
    return np.asarray(points, dtype=np.float64)


def process_file(input_path: Path, output_path: Path, max_distance_m: float) -> None:
    try:
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        output_path.write_text("{}\n", encoding="utf-8")
        return

    if not payload or not payload.get("future_poses"):
        output_path.write_text("{}\n", encoding="utf-8")
        return

    points_xy = _collect_points_within_distance(payload, max_distance_m)
    coeffs, x_range = fit_poly_vehicle_smooth(points_xy)
    if coeffs is None:
        output_path.write_text("{}\n", encoding="utf-8")
        return

    out = {
        "source_file": payload.get("source_file", input_path.name),
        "source_timestamp": payload.get("source_timestamp"),
        "output_frame": payload.get("output_frame", "unknown"),
        "max_distance_m": float(max_distance_m),
        "num_points_used": int(points_xy.shape[0]),
        "polyline": {
            "type": "cubic_y_of_x",
            "coeffs": coeffs,
            "x_range": x_range,
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.future_json_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.input_glob))
    if not files:
        print("No input files found. Nothing to do.")
        return

    for p in tqdm(files):
        process_file(p, out_dir / p.name, args.max_distance_m)

    print(f"Processed {len(files)} files. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
