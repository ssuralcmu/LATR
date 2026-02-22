#!/usr/bin/env python3
"""Convert predicted 3D lane points JSON files into smoothed 3D polylines.

Expected input file format is the one produced by shounak_scripts/infer_custom.py,
with lane points in `pred_laneLines`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
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


    if len(y_interp) > 9:
        y_interp = savgol_filter(y_interp, 9, 3)


    coeffs = np.polyfit(x_interp, y_interp, 3)
    x_range = [float(np.min(x_interp)), float(np.max(x_interp))]
    return coeffs.tolist(), x_range


def lane_points_to_polyline3d(lane_points: List[List[float]], samples: int) -> Dict[str, Any]:
    lane_arr = np.asarray(lane_points, dtype=np.float64)
    if lane_arr.ndim != 2 or lane_arr.shape[1] != 3 or len(lane_arr) == 0:
        return {
            "polyline3d": [],
            "poly_coeff_y_of_x": None,
            "x_range": None,
            "valid": False,
            "reason": "lane points must be non-empty Nx3",
        }

    coeffs, x_range = fit_poly_vehicle_smooth(lane_arr[:, :2])
    if coeffs is None or x_range is None:
        return {
            "polyline3d": lane_arr.tolist(),
            "poly_coeff_y_of_x": None,
            "x_range": None,
            "valid": False,
            "reason": "insufficient geometric variation to fit cubic",
        }

    x_sorted_idx = np.argsort(lane_arr[:, 0])
    x_sorted = lane_arr[x_sorted_idx, 0]
    z_sorted = lane_arr[x_sorted_idx, 2]

    # Avoid np.interp issues with duplicate x by collapsing to unique x.
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    z_unique = z_sorted[unique_idx]

    x_min, x_max = x_range
    x_samples = np.linspace(x_min, x_max, samples, dtype=np.float64)
    y_samples = np.polyval(np.asarray(coeffs, dtype=np.float64), x_samples)

    if len(x_unique) == 1:
        z_samples = np.full_like(x_samples, fill_value=float(z_unique[0]))
    else:
        z_samples = np.interp(x_samples, x_unique, z_unique)

    polyline3d = np.stack([x_samples, y_samples, z_samples], axis=1).tolist()
    return {
        "polyline3d": polyline3d,
        "poly_coeff_y_of_x": coeffs,
        "x_range": x_range,
        "valid": True,
        "reason": None,
    }


def convert_file(input_path: Path, output_path: Path, samples: int) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lanes = data.get("pred_laneLines", [])
    probs = data.get("pred_laneLines_prob", [])

    output_lanes = []
    for lane_idx, lane in enumerate(lanes):
        lane_poly = lane_points_to_polyline3d(lane, samples=samples)
        lane_poly["lane_index"] = lane_idx
        lane_poly["probability"] = probs[lane_idx] if lane_idx < len(probs) else None
        output_lanes.append(lane_poly)

    out_data = {
        "image": data.get("image"),
        "source_prediction_file": str(input_path),
        "calibration": data.get("calibration"),
        "pred_laneLines_polyline3d": output_lanes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert predicted 3D lane points JSON files to smoothed 3D polylines."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing prediction JSON files (e.g. workzone3d_image1_lanes_once/pred).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where converted polyline JSON files will be saved.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of points per output polyline (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.samples < 2:
        raise ValueError("--samples must be >= 2")

    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.input_dir}")

    for input_file in tqdm(json_files):
        output_file = args.output_dir / input_file.name
        convert_file(input_file, output_file, samples=args.samples)

    print(f"Converted {len(json_files)} files to: {args.output_dir}")


if __name__ == "__main__":
    main()
