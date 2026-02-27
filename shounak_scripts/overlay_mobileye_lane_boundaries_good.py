#!/usr/bin/env python3
"""
Overlay Mobileye lane boundaries from the logger CSV/TXT on a camera image.

The lane model used is:
    lateral_right_m(x) = d0 + c1*x + b2*x^2 + a3*x^3
where x is forward distance in meters.

Coordinate convention used by this script:
  vehicle frame: x forward, y right, z down
  KITTI Velodyne frame: x forward, y left, z up
  camera optical frame (OpenCV): x right, y down, z forward
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraPoseVeh:
    x_m: float
    y_m: float
    z_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass
class LaneBoundary:
    boundary_index: int
    lane_type: int
    a3: float
    b2: float
    c1: float
    d0: float


# KITTI-style calibration values provided by user.
# Parsed from row-major text, same way KITTI calibration files are typically read.
_KITTI_P1_STR = (
    "2329.753910 0.000000 516.122235 0.000000 "
    "0.000000 2343.238660 418.265621 0.000000 "
    "0.000000 0.000000 1.000000 0.000000"
)
_KITTI_R0_RECT_STR = (
    "1.000000 0.000000 0.000000 "
    "0.000000 1.000000 0.000000 "
    "0.000000 0.000000 1.000000"
)
_KITTI_TR_VELO_TO_CAM_1_STR = (
    "-0.027723 -0.999459 0.017672 0.449620 "
    "-0.028772 -0.016873 -0.999444 1.307979 "
    "0.999201 -0.028216 -0.028289 -1.994105"
)


def parse_kitti_matrix(values: str, rows: int, cols: int) -> np.ndarray:
    arr = np.fromstring(values, sep=" ", dtype=np.float64)
    expected = rows * cols
    if arr.size != expected:
        raise ValueError(
            f"KITTI matrix parse error: expected {expected} values, got {arr.size}"
        )
    return arr.reshape(rows, cols)


KITTI_P1 = parse_kitti_matrix(_KITTI_P1_STR, 3, 4)
KITTI_R0_RECT = parse_kitti_matrix(_KITTI_R0_RECT_STR, 3, 3)
KITTI_TR_VELO_TO_CAM_1 = parse_kitti_matrix(_KITTI_TR_VELO_TO_CAM_1_STR, 3, 4)

# Lane filtering constants.
# 1) Remove a recurring spurious lane near the vehicle center that is almost straight.
_SPURIOUS_CENTER_MAX_ABS_D0_M = 0.30
_SPURIOUS_CENTER_MAX_ABS_C1 = 0.02
_SPURIOUS_CENTER_MAX_ABS_B2 = 0.002
_SPURIOUS_CENTER_MAX_ABS_A3 = 0.0001

# 2) If two lanes are nearly parallel and within 0.5 m for most of sampled range,
# keep only the one closer to vehicle center.
_PARALLEL_CLOSE_DIST_M = 0.50
_PARALLEL_MAX_HEADING_DIFF_RAD = math.radians(4.0)
_PARALLEL_REQUIRED_FRACTION = 0.70
_PARALLEL_EVAL_RANGE_M = 40.0
_PARALLEL_EVAL_STEP_M = 1.0

# Only keep lane marker types accepted by the current mapping stack.
_ALLOWED_LANE_TYPES = {0, 1, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Mobileye lane boundaries on an image."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Input image path (single-image mode)")
    input_group.add_argument(
        "--image-dir",
        help="Input image directory (batch mode: process all images in folder)",
    )
    parser.add_argument(
        "--lane-log",
        required=True,
        help="Lane logger CSV/TXT path (e.g., /tmp/...mobileye_lane_vehicle_state_log.csv)",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Image timestamp for single-image mode. Supported: epoch seconds, "
            "YYYYMMDDTHHMMSS(.ffffff), or YYYY-mm-ddTHH:MM:SS(.ffffff). "
            "If omitted, tries to parse from image filename."
        ),
    )
    parser.add_argument(
        "--max-time-diff-s",
        type=float,
        default=0.25,
        help="Max allowed |lane_ts - image_ts| before warning (default: 0.25 s)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=50.0,
        help="Only overlay boundaries with lane_confidence strictly greater than this value (default: 50)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (single-image mode; default: <input>_lane_overlay.<ext>)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (batch mode; default: <image-dir>/viz)",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=0.5,
        help="Sampling step along lane curve in meters (default: 0.5)",
    )
    parser.add_argument(
        "--max-range-m",
        type=float,
        default=40.0,
        help="Max forward range to draw each boundary (default: 70 m)",
    )
    parser.add_argument(
        "--max-lateral-abs-m",
        type=float,
        default=25.0,
        help="Drop sampled points with |lateral| above this value (default: 25 m)",
    )
    parser.add_argument(
        "--forward-offset-m",
        type=float,
        default=-1.9,
        help="Additional forward offset (vehicle x, meters) applied to all lane points before projection (default: 0.0)",
    )
    parser.add_argument(
        "--z-offset-m",
        type=float,
        default=-0.90,
        help="Additional z offset (vehicle z-down, meters) applied to all lane points before projection (default: 0.0)",
    )
    parser.add_argument(
        "--y-offset-m",
        type=float,
        default=-0.3,
        help="Additional lateral offset (vehicle y-right, meters) applied to all lane points before projection (default: 0.0)",
    )

    # Ground plane height in vehicle z-down frame (CT6 Ground z in config/calibration-ct6.rb)
    parser.add_argument("--ground-z-m", type=float, default=0.7366)

    return parser.parse_args()


def parse_float(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def parse_timestamp_to_epoch_seconds(value: str) -> Optional[float]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None

    # epoch seconds first
    try:
        return float(s)
    except ValueError:
        pass

    formats = (
        "%Y%m%dT%H%M%S.%f",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(s, fmt)
            # ptime/to_iso_string values are treated as UTC for matching log epoch timestamps
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.timestamp()
        except ValueError:
            continue
    return None


def try_parse_timestamp_from_image_name(image_path: str) -> Optional[float]:
    name = os.path.basename(image_path)
    front_cam_match = re.search(
        r"Front_Camera_(\d{8}T\d{6}(?:\.\d+)?)\.(?:jpg|jpeg|png|bmp|tif|tiff)$",
        name,
        flags=re.IGNORECASE,
    )
    if front_cam_match:
        ts = parse_timestamp_to_epoch_seconds(front_cam_match.group(1))
        if ts is not None:
            return ts

    base = os.path.splitext(name)[0]
    # Supports names like 20241029T185634.123456 and ..._long_20241029T185634
    matches = re.findall(r"\d{8}T\d{6}(?:\.\d+)?", base)
    for m in reversed(matches):
        ts = parse_timestamp_to_epoch_seconds(m)
        if ts is not None:
            return ts
    return None


def list_images_in_dir(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for entry in sorted(os.listdir(image_dir)):
        p = os.path.join(image_dir, entry)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in exts:
            out.append(p)
    return out


def choose_delimiter(sample: str) -> str:
    comma = sample.count(",")
    tab = sample.count("\t")
    return "," if comma >= tab else "\t"


def load_lane_rows(lane_log_path: str) -> List[Dict[str, str]]:
    with open(lane_log_path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = choose_delimiter(sample)
        reader = csv.DictReader(f, delimiter=delim)
        rows: List[Dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            lane_ts_raw = (row.get("lane_timestamp_s") or "").strip()
            if not lane_ts_raw or lane_ts_raw == "lane_timestamp_s":
                continue
            lane_ts = parse_float(lane_ts_raw)
            if not math.isfinite(lane_ts):
                continue
            row["_lane_ts_key"] = lane_ts_raw
            row["_lane_ts"] = lane_ts
            rows.append(row)
    return rows


def find_closest_timestamp_rows(
    rows: Sequence[Dict[str, str]], target_ts: float
) -> Tuple[List[Dict[str, str]], float, float]:
    closest_row = min(rows, key=lambda r: abs(r["_lane_ts"] - target_ts))
    closest_ts = float(closest_row["_lane_ts"])
    key = str(closest_row["_lane_ts_key"])
    group = [r for r in rows if str(r["_lane_ts_key"]) == key]
    abs_dt = abs(closest_ts - target_ts)
    return group, closest_ts, abs_dt


def rows_to_boundaries(
    rows: Sequence[Dict[str, str]], min_confidence: float
) -> Tuple[List[LaneBoundary], int, int]:
    out: List[LaneBoundary] = []
    skipped_low_conf = 0
    skipped_lane_type = 0
    for row in rows:
        conf = parse_float(row.get("lane_confidence"))
        if not math.isfinite(conf) or conf <= min_confidence:
            skipped_low_conf += 1
            continue

        lane_type = int(parse_float(row.get("boundary_lane_type")))
        if lane_type not in _ALLOWED_LANE_TYPES:
            skipped_lane_type += 1
            continue

        a3 = parse_float(row.get("a3_curvature_derivative"))
        b2 = parse_float(row.get("b2_curvature"))
        c1 = parse_float(row.get("c1_heading_angle"))
        d0 = parse_float(row.get("d0_position"))
        if not all(math.isfinite(v) for v in (a3, b2, c1, d0)):
            continue
        boundary_index = int(parse_float(row.get("boundary_index")))
        out.append(
            LaneBoundary(
                boundary_index=boundary_index,
                lane_type=lane_type,
                a3=a3,
                b2=b2,
                c1=c1,
                d0=d0,
            )
        )
    return out, skipped_low_conf, skipped_lane_type


def evaluate_lane_lateral(lane: LaneBoundary, x_samples_m: np.ndarray) -> np.ndarray:
    return (
        lane.d0
        + lane.c1 * x_samples_m
        + lane.b2 * (x_samples_m ** 2)
        + lane.a3 * (x_samples_m ** 3)
    )


def evaluate_lane_heading_rad(lane: LaneBoundary, x_samples_m: np.ndarray) -> np.ndarray:
    dydx = lane.c1 + 2.0 * lane.b2 * x_samples_m + 3.0 * lane.a3 * (x_samples_m ** 2)
    return np.arctan(dydx)


def is_spurious_center_straight_boundary(lane: LaneBoundary) -> bool:
    return (
        abs(lane.d0) <= _SPURIOUS_CENTER_MAX_ABS_D0_M
        and abs(lane.c1) <= _SPURIOUS_CENTER_MAX_ABS_C1
        and abs(lane.b2) <= _SPURIOUS_CENTER_MAX_ABS_B2
        and abs(lane.a3) <= _SPURIOUS_CENTER_MAX_ABS_A3
    )


def lanes_are_parallel_and_close_mostly(
    lane_a: LaneBoundary, lane_b: LaneBoundary, x_samples_m: np.ndarray
) -> bool:
    y_a = evaluate_lane_lateral(lane_a, x_samples_m)
    y_b = evaluate_lane_lateral(lane_b, x_samples_m)
    h_a = evaluate_lane_heading_rad(lane_a, x_samples_m)
    h_b = evaluate_lane_heading_rad(lane_b, x_samples_m)

    close_fraction = float(np.mean(np.abs(y_a - y_b) <= _PARALLEL_CLOSE_DIST_M))
    parallel_fraction = float(np.mean(np.abs(h_a - h_b) <= _PARALLEL_MAX_HEADING_DIFF_RAD))

    return (
        close_fraction >= _PARALLEL_REQUIRED_FRACTION
        and parallel_fraction >= _PARALLEL_REQUIRED_FRACTION
    )


def lane_center_proximity_score(lane: LaneBoundary, x_samples_m: np.ndarray) -> float:
    # Smaller means laterally closer to the vehicle centerline over the sampled range.
    y = evaluate_lane_lateral(lane, x_samples_m)
    return float(np.mean(np.abs(y)))


def suppress_parallel_close_duplicates(
    boundaries: Sequence[LaneBoundary], max_range_m: float
) -> Tuple[List[LaneBoundary], int]:
    if len(boundaries) <= 1:
        return list(boundaries), 0

    eval_end = min(max_range_m, _PARALLEL_EVAL_RANGE_M)
    if eval_end <= 0.0:
        x_samples_m = np.array([0.0], dtype=np.float64)
    else:
        x_samples_m = np.arange(
            0.0, eval_end + 0.5 * _PARALLEL_EVAL_STEP_M, _PARALLEL_EVAL_STEP_M, dtype=np.float64
        )
        if x_samples_m.size == 0:
            x_samples_m = np.array([0.0], dtype=np.float64)

    ordered = sorted(
        boundaries,
        key=lambda lane: (
            lane_center_proximity_score(lane, x_samples_m),
            lane.boundary_index,
        ),
    )

    kept: List[LaneBoundary] = []
    removed = 0
    for candidate in ordered:
        duplicate_of_kept = False
        for anchor in kept:
            if lanes_are_parallel_and_close_mostly(candidate, anchor, x_samples_m):
                duplicate_of_kept = True
                removed += 1
                break
        if not duplicate_of_kept:
            kept.append(candidate)

    return kept, removed


def filter_lane_boundaries(
    boundaries: Sequence[LaneBoundary], max_range_m: float
) -> Tuple[List[LaneBoundary], int, int]:
    no_center_straight = [
        lane for lane in boundaries if not is_spurious_center_straight_boundary(lane)
    ]
    removed_center_straight = len(boundaries) - len(no_center_straight)

    deduped, removed_parallel_close = suppress_parallel_close_duplicates(
        no_center_straight, max_range_m=max_range_m
    )

    return deduped, removed_center_straight, removed_parallel_close


def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    # Camera-body orientation in vehicle frame (x forward, y right, z down)
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def sample_lane_points_vehicle(
    lane: LaneBoundary,
    x_start_m: float,
    x_end_m: float,
    step_m: float,
    ground_z_m: float,
    max_lateral_abs_m: float,
    forward_offset_m: float,
    z_offset_m: float,
    y_offset_m: float,
) -> np.ndarray:
    x = np.arange(x_start_m, x_end_m + 0.5 * step_m, step_m, dtype=np.float64)
    y = lane.d0 + lane.c1 * x + lane.b2 * (x ** 2) + lane.a3 * (x ** 3)
    valid = np.isfinite(y) & (np.abs(y) <= max_lateral_abs_m)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.stack(
        [
            forward_offset_m + x,  # forward
            y_offset_m + y,  # right
            np.full_like(x, ground_z_m + z_offset_m),  # down
        ],
        axis=1,
    )
    return pts


def project_vehicle_to_image_kitti(
    pts_vehicle: np.ndarray, min_depth_m: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    if pts_vehicle.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)

    # Convert vehicle frame (x fwd, y right, z down) -> KITTI velodyne (x fwd, y left, z up).
    pts_velo = np.empty_like(pts_vehicle)
    pts_velo[:, 0] = pts_vehicle[:, 0]
    pts_velo[:, 1] = -pts_vehicle[:, 1]
    pts_velo[:, 2] = -pts_vehicle[:, 2]

    pts_velo_h = np.hstack([pts_velo, np.ones((pts_velo.shape[0], 1), dtype=np.float64)])

    # KITTI projection chain:
    # x_img ~ P1 * [R0_rect * (Tr_velo_to_cam_1 * x_velo_h); 1]
    pts_cam = (KITTI_TR_VELO_TO_CAM_1 @ pts_velo_h.T).T  # Nx3
    pts_cam_rect = (KITTI_R0_RECT @ pts_cam.T).T         # Nx3

    depth = pts_cam_rect[:, 2]
    valid = np.isfinite(depth) & (depth > min_depth_m)

    pts_cam_rect_h = np.hstack(
        [pts_cam_rect, np.ones((pts_cam_rect.shape[0], 1), dtype=np.float64)]
    )
    uvw = (KITTI_P1 @ pts_cam_rect_h.T).T

    uv = np.empty((pts_vehicle.shape[0], 2), dtype=np.float64)
    uv[:] = np.nan
    denom = uvw[:, 2]
    good = valid & np.isfinite(denom) & (denom > min_depth_m)
    uv[good, 0] = uvw[good, 0] / denom[good]
    uv[good, 1] = uvw[good, 1] / denom[good]
    return uv, valid


def split_contiguous(indices: np.ndarray) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, splits)


def draw_projected_polyline(
    image: np.ndarray, uv: np.ndarray, valid: np.ndarray, color: Tuple[int, int, int], thickness: int
) -> Optional[Tuple[int, int]]:
    h, w = image.shape[:2]
    in_view = (
        valid
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= -200)
        & (uv[:, 0] < w + 200)
        & (uv[:, 1] >= -200)
        & (uv[:, 1] < h + 200)
    )
    idx = np.where(in_view)[0]
    if idx.size < 2:
        return None

    first_label_point: Optional[Tuple[int, int]] = None
    for seg_idx in split_contiguous(idx):
        if seg_idx.size < 2:
            continue
        pts = uv[seg_idx]
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts_i], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if first_label_point is None:
            first_label_point = (int(pts_i[0, 0, 0]), int(pts_i[0, 0, 1]))
    return first_label_point


def default_output_path(image_path: str) -> str:
    base, ext = os.path.splitext(image_path)
    if not ext:
        ext = ".png"
    return f"{base}_lane_overlay{ext}"


def render_overlay(
    image_path: str,
    rows: Sequence[Dict[str, str]],
    out_path: str,
    args: argparse.Namespace,
    timestamp_override: Optional[str] = None,
) -> bool:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: failed to read image: {image_path}", file=sys.stderr)
        return False

    image_ts = (
        parse_timestamp_to_epoch_seconds(timestamp_override)
        if timestamp_override
        else None
    )
    if image_ts is None:
        image_ts = try_parse_timestamp_from_image_name(image_path)
    if image_ts is None:
        print(
            f"ERROR: could not determine timestamp from filename: {image_path}",
            file=sys.stderr,
        )
        return False

    closest_rows, lane_ts, abs_dt = find_closest_timestamp_rows(rows, image_ts)
    if abs_dt > args.max_time_diff_s:
        print(
            f"WARNING: closest lane timestamp differs by {abs_dt:.6f} s "
            f"(threshold {args.max_time_diff_s:.6f} s) for image={os.path.basename(image_path)}",
            file=sys.stderr,
        )

    boundaries, skipped_low_conf, skipped_lane_type = rows_to_boundaries(
        closest_rows, min_confidence=args.min_confidence
    )
    boundaries_in_group = len(boundaries)

    boundaries, filtered_center_straight, filtered_parallel_close = filter_lane_boundaries(
        boundaries, max_range_m=args.max_range_m
    )

    if not boundaries:
        print(
            f"ERROR: no valid lane boundary coefficients after filtering "
            f"(min_confidence>{args.min_confidence:.3f}, "
            f"allowed_lane_types={sorted(_ALLOWED_LANE_TYPES)}, "
            f"center_straight_removed={filtered_center_straight}, "
            f"parallel_close_removed={filtered_parallel_close}) "
            f"for image={os.path.basename(image_path)}",
            file=sys.stderr,
        )
        return False

    # BGR colors
    palette = [
        (0, 255, 255),
        (0, 200, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (0, 128, 255),
        (255, 128, 0),
    ]

    drawn = 0
    for lane in sorted(boundaries, key=lambda x: x.boundary_index):
        pts_vehicle = sample_lane_points_vehicle(
            lane=lane,
            x_start_m=0.0,
            x_end_m=args.max_range_m,
            step_m=args.step_m,
            ground_z_m=args.ground_z_m,
            max_lateral_abs_m=args.max_lateral_abs_m,
            forward_offset_m=args.forward_offset_m,
            z_offset_m=args.z_offset_m,
            y_offset_m=args.y_offset_m,
        )
        uv, valid = project_vehicle_to_image_kitti(pts_vehicle)
        color = palette[lane.boundary_index % len(palette)]
        label_point = draw_projected_polyline(image, uv, valid, color=color, thickness=2)
        if label_point is not None:
            txt = f"id={lane.boundary_index} type={lane.lane_type}"
            cv2.putText(
                image,
                txt,
                (label_point[0] + 5, label_point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            drawn += 1

    ok = cv2.imwrite(out_path, image)
    if not ok:
        print(f"ERROR: failed to write output image: {out_path}", file=sys.stderr)
        return False

    print(
        f"image={os.path.basename(image_path)} "
        f"image_timestamp_s={image_ts:.9f} "
        f"matched_lane_timestamp_s={lane_ts:.9f} "
        f"abs_time_diff_s={abs_dt:.9f} "
        f"min_confidence={args.min_confidence:.3f} "
        f"skipped_low_conf={skipped_low_conf} "
        f"skipped_lane_type={skipped_lane_type} "
        f"boundaries_in_group={boundaries_in_group} "
        f"filtered_center_straight={filtered_center_straight} "
        f"filtered_parallel_close={filtered_parallel_close} "
        f"boundaries_after_filter={len(boundaries)} "
        f"boundaries_drawn={drawn} "
        f"output={out_path}"
    )
    return True


def main() -> int:
    args = parse_args()

    rows = load_lane_rows(args.lane_log)
    if not rows:
        print(f"ERROR: no valid lane rows found in {args.lane_log}", file=sys.stderr)
        return 1

    if args.image is not None:
        out_path = args.output if args.output else default_output_path(args.image)
        ok = render_overlay(
            image_path=args.image,
            rows=rows,
            out_path=out_path,
            args=args,
            timestamp_override=args.timestamp,
        )
        if ok:
            print("projection_model=KITTI(P1,R0_rect,Tr_velo_to_cam_1)")
            return 0
        return 1

    if args.timestamp is not None:
        print(
            "WARNING: --timestamp is ignored in --image-dir mode; using Front_Camera_<timestamp> from each filename.",
            file=sys.stderr,
        )

    image_dir = str(args.image_dir)
    if not os.path.isdir(image_dir):
        print(f"ERROR: image directory does not exist: {image_dir}", file=sys.stderr)
        return 1

    image_paths = list_images_in_dir(image_dir)
    if not image_paths:
        print(f"ERROR: no images found in directory: {image_dir}", file=sys.stderr)
        return 1

    out_dir = args.output_dir if args.output_dir else os.path.join(image_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    ok_count = 0
    for image_path in image_paths:
        out_path = os.path.join(out_dir, os.path.basename(image_path))
        ok = render_overlay(
            image_path=image_path,
            rows=rows,
            out_path=out_path,
            args=args,
            timestamp_override=None,
        )
        if ok:
            ok_count += 1

    print("projection_model=KITTI(P1,R0_rect,Tr_velo_to_cam_1)")
    print(f"batch_total={len(image_paths)}")
    print(f"batch_success={ok_count}")
    print(f"batch_failed={len(image_paths) - ok_count}")
    print(f"output_dir={out_dir}")
    return 0 if ok_count == len(image_paths) else 2


if __name__ == "__main__":
    sys.exit(main())
