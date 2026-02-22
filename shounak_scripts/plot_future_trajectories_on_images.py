#!/usr/bin/env python3
"""Overlay future ego trajectories on images using KITTI-format calibration.

Inputs:
- Generated future-pose JSONs (from generate_future_ego_poses.py)
- image_1 folder with source images
- calib folder with KITTI calibration txt files

Output:
- image_1_with_traj folder with overlaid trajectories.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot future trajectories on images using KITTI calib.")
    parser.add_argument("--future-json-dir", required=True, help="Directory of generated future-pose JSON files.")
    parser.add_argument("--image-dir", default="image_1", help="Directory with source images.")
    parser.add_argument("--calib-dir", default="calib", help="Directory with KITTI calibration txt files.")
    parser.add_argument("--output-dir", default="image_1_with_traj", help="Directory to save overlaid images.")
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image extensions to search for matching stem.",
    )
    return parser.parse_args()


def _parse_kitti_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            nums = np.array([float(v) for v in value.strip().split()], dtype=np.float64)
            if key.startswith("P") and nums.size == 12:
                data[key] = nums.reshape(3, 4)
            elif key in {"R0_rect", "R_rect"} and nums.size == 9:
                data["R0_rect"] = nums.reshape(3, 3)
            elif key.startswith("Tr_velo_to_cam") and nums.size == 12:
                # Supports Tr_velo_to_cam, Tr_velo_to_cam_0/1/2, ...
                data[key] = nums.reshape(3, 4)
                if key == "Tr_velo_to_cam":
                    data["Tr_velo_to_cam_default"] = data[key]
    return data


def _find_existing_by_stem(folder: Path, stem: str, exts: Optional[List[str]] = None) -> Optional[Path]:
    if exts is None:
        candidate = folder / f"{stem}.txt"
        return candidate if candidate.exists() else None

    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _project_ego_points_to_image(
    points_ego: np.ndarray,
    calib: Dict[str, np.ndarray],
) -> np.ndarray:
    """Project Nx3 ego points to Nx2 image points.

    Assumes ego frame ~= Velodyne frame for projection with KITTI Tr_velo_to_cam.
    """
    p_key = "P2" if "P2" in calib else ("P0" if "P0" in calib else ("P1" if "P1" in calib else ("P3" if "P3" in calib else None)))
    if p_key is None:
        raise ValueError("Missing projection matrix (P2/P0/P1/P3) in calib")
    p = calib[p_key]

    # Prefer camera-specific extrinsic matching selected projection matrix (e.g., P2 -> Tr_velo_to_cam_2).
    tr = calib.get(f"Tr_velo_to_cam_{p_key[1:]}")
    if tr is None:
        tr = calib.get("Tr_velo_to_cam") or calib.get("Tr_velo_to_cam_default")
    r0 = calib.get("R0_rect")

    xyz1 = np.concatenate([points_ego, np.ones((points_ego.shape[0], 1), dtype=np.float64)], axis=1)  # Nx4

    if tr is not None:
        cam = (tr @ xyz1.T).T  # Nx3
    else:
        cam = points_ego.copy()

    if r0 is not None:
        cam = (r0 @ cam.T).T

    cam1 = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)  # Nx4
    uvw = (p @ cam1.T).T

    valid = uvw[:, 2] > 1e-6
    uv = np.full((points_ego.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = uvw[valid, 0] / uvw[valid, 2]
    uv[valid, 1] = uvw[valid, 1] / uvw[valid, 2]
    return uv


def _collect_trajectory_points(future_payload: Dict) -> np.ndarray:
    future_poses = future_payload.get("future_poses", [])
    pts = [[0.0, 0.0, 0.0]]
    for item in future_poses:
        pos = item.get("relative_position_xyz", {})
        if not all(k in pos for k in ("x", "y", "z")):
            continue
        pts.append([float(pos["x"]), float(pos["y"]), float(pos["z"])])
    return np.asarray(pts, dtype=np.float64)


def _draw_trajectory(image: np.ndarray, uv: np.ndarray, point_radius: int, line_thickness: int) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]

    valid_pts: List[Tuple[int, int]] = []
    for x, y in uv:
        if np.isnan(x) or np.isnan(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            valid_pts.append((xi, yi))

    if not valid_pts:
        return canvas

    # Color ramp from red (near/current) to green (far/future)
    n = len(valid_pts)
    for i in range(1, n):
        t = i / max(1, n - 1)
        color = (0, int(255 * t), int(255 * (1 - t)))  # BGR
        cv2.line(canvas, valid_pts[i - 1], valid_pts[i], color, line_thickness, cv2.LINE_AA)

    for i, pt in enumerate(valid_pts):
        t = i / max(1, n - 1)
        color = (0, int(255 * t), int(255 * (1 - t)))
        cv2.circle(canvas, pt, point_radius, color, -1, cv2.LINE_AA)

    return canvas


def main() -> None:
    args = parse_args()

    future_json_dir = Path(args.future_json_dir)
    image_dir = Path(args.image_dir)
    calib_dir = Path(args.calib_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    future_files = sorted(future_json_dir.glob("*.json"))
    if not future_files:
        print("No future-pose JSON files found. Nothing to do.")
        return

    for json_path in tqdm(future_files, desc="Overlay trajectories"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Empty or no future data: skip rendering.
        if not payload or not payload.get("future_poses"):
            continue

        source_name = payload.get("source_file", json_path.name)
        stem = Path(source_name).stem

        img_path = _find_existing_by_stem(image_dir, stem, args.image_exts)
        calib_path = _find_existing_by_stem(calib_dir, stem)
        if img_path is None or calib_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        try:
            calib = _parse_kitti_calib(calib_path)
            pts_ego = _collect_trajectory_points(payload)
            if pts_ego.shape[0] < 2:
                continue
            uv = _project_ego_points_to_image(pts_ego, calib)
            rendered = _draw_trajectory(image, uv, args.point_radius, args.line_thickness)
        except (ValueError, KeyError, TypeError):
            continue

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), rendered)

    print(f"Done. Overlaid images saved to: {output_dir}")


if __name__ == "__main__":
    main()
