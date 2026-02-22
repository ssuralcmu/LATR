#!/usr/bin/env python3
"""Fit 3D polylines from KITTI label_2 work-zone object annotations.

For each KITTI txt file:
- Parse object rows and extract 3D locations (x, y, z).
- Keep rows matching requested class names (default: Channelizer).
- If at least 2 objects are available, fit a smoothed cubic curve in X(Z)
  using fit_poly_vehicle_smooth on (z, x) points.
- Sample points on the fitted curve and emit one JSON per input file.
- If not enough points or fitting fails, write {}.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit 3D polylines from KITTI label_2 object txt files.")
    parser.add_argument("--input-dir", required=True, help="Directory with KITTI label_2 txt files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write output JSON files.")
    parser.add_argument("--input-glob", default="*.txt", help="Glob for input files (default: *.txt).")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["Channelizer"],
        help="Object class names to include (case-sensitive).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of sampled points to emit on each fitted 3D polyline.",
    )
    return parser.parse_args()


def _parse_kitti_object_line(line: str) -> Optional[Tuple[str, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    try:
        obj_type = parts[0]
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
    except ValueError:
        return None
    return obj_type, x, y, z


def _collect_points(path: Path, class_names: List[str]) -> np.ndarray:
    points: List[List[float]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return np.zeros((0, 3), dtype=np.float64)

    class_set = set(class_names)
    for line in lines:
        parsed = _parse_kitti_object_line(line)
        if parsed is None:
            continue
        obj_type, x, y, z = parsed
        if obj_type in class_set:
            points.append([x, y, z])

    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def process_file(input_path: Path, output_path: Path, class_names: List[str], num_samples: int) -> None:
    points_xyz = _collect_points(input_path, class_names)
    if points_xyz.shape[0] < 2:
        output_path.write_text("{}\n", encoding="utf-8")
        return

    # Fit X as a function of Z by reusing fit_poly_vehicle_smooth(points_xy).
    points_zx = np.column_stack([points_xyz[:, 2], points_xyz[:, 0]])
    coeffs, z_range = fit_poly_vehicle_smooth(points_zx)
    if coeffs is None or z_range is None:
        output_path.write_text("{}\n", encoding="utf-8")
        return

    z_samples = np.linspace(z_range[0], z_range[1], max(2, num_samples))
    x_samples = np.polyval(coeffs, z_samples)
    y_const = float(np.mean(points_xyz[:, 1]))

    sampled_xyz = [[float(x), y_const, float(z)] for z, x in zip(z_samples, x_samples)]

    out = {
        "source_file": input_path.name,
        "num_objects_used": int(points_xyz.shape[0]),
        "classes_used": class_names,
        "polyline": {
            "type": "cubic_x_of_z",
            "coeffs": coeffs,
            "z_range": [float(z_range[0]), float(z_range[1])],
            "sampled_xyz": sampled_xyz,
            "y_strategy": "mean_of_objects",
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.input_glob))
    if not files:
        print("No input files found. Nothing to do.")
        return

    for p in tqdm(files):
        process_file(p, out_dir / f"{p.stem}.json", args.class_names, args.num_samples)

    print(f"Processed {len(files)} files. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
