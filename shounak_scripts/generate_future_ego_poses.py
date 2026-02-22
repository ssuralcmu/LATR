#!/usr/bin/env python3
"""Generate per-frame future ego poses relative to current ego frame."""

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

@dataclass
class FrameState:
    path: Path
    timestamp: datetime
    pose: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate future relative ego poses for vehicle-state JSONs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-seconds", type=float, default=10.0)
    parser.add_argument(
        "--timestamp-source",
        choices=["vehicle_timestamp", "ref_timestamp_prefix"],
        default="vehicle_timestamp",
    )
    parser.add_argument("--input-glob", default="*.json")
    parser.add_argument("--rotation-order", default="zyx", choices=["xyz", "zyx"])
    parser.add_argument(
        "--output-frame",
        default="fru",
        choices=["flu", "fru"],
        help="Output relative deltas in FLU (front-left-up) or FRU (front-right-up).",
    )
    return parser.parse_args()


def parse_timestamp(payload: Dict, source: str) -> datetime:
    if source == "vehicle_timestamp":
        return datetime.fromisoformat(payload["closest_vehicle_state"]["vehicle_timestamp"])
    prefix = payload["ref_timestamp_prefix"]
    return datetime.strptime(prefix[1:], "%Y%m%dT%H%M%S.%f")


def mat3_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat3_transpose(a: List[List[float]]) -> List[List[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mat3_vec_mul(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[i] - b[i] for i in range(3)]


def rotation_matrix(axis: str, angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == "y":
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    if axis == "z":
        return [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    raise ValueError(axis)


def pose_to_rt(pose: Dict[str, float], order: str) -> Tuple[List[List[float]], List[float]]:
    angles = [float(pose["rot1"]), float(pose["rot2"]), float(pose["rot3"])]
    r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for ax, ang in zip(order, angles):
        r = mat3_mul(r, rotation_matrix(ax, ang))
    t = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return r, t


def relative_rt(r_cur, t_cur, r_fut, t_fut):
    r_cur_t = mat3_transpose(r_cur)
    rel_r = mat3_mul(r_cur_t, r_fut)
    rel_t = mat3_vec_mul(r_cur_t, vec_sub(t_fut, t_cur))
    return rel_r, rel_t




def convert_flu_to_frame(rel_r: List[List[float]], rel_t: List[float], output_frame: str) -> Tuple[List[List[float]], List[float]]:
    if output_frame == "flu":
        return rel_r, rel_t

    # FLU -> FRU conversion: flip Y axis.
    d = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    rel_r_fru = mat3_mul(mat3_mul(d, rel_r), d)
    rel_t_fru = [rel_t[0], -rel_t[1], rel_t[2]]
    return rel_r_fru, rel_t_fru

def rotation_matrix_to_euler_xyz(r: List[List[float]]) -> Tuple[float, float, float]:
    sy = math.sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0])
    singular = sy < 1e-8
    if not singular:
        x = math.atan2(r[2][1], r[2][2])
        y = math.atan2(-r[2][0], sy)
        z = math.atan2(r[1][0], r[0][0])
    else:
        x = math.atan2(-r[1][2], r[1][1])
        y = math.atan2(-r[2][0], sy)
        z = 0.0
    return x, y, z


def load_frame_states(input_dir: Path, input_glob: str, timestamp_source: str) -> List[FrameState]:
    states = []
    for path in sorted(input_dir.glob(input_glob)):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            states.append(
                FrameState(
                    path=path,
                    timestamp=parse_timestamp(payload, timestamp_source),
                    pose=payload["closest_vehicle_state"]["pose"],
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Skip malformed files or files missing timestamp/vehicle-state fields.
            continue
    return sorted(states, key=lambda s: s.timestamp)


def build_output_for_frame(
    idx: int,
    states: List[FrameState],
    horizon_seconds: float,
    rotation_order: str,
    output_frame: str,
) -> Optional[Dict]:
    cur = states[idx]
    r_cur, t_cur = pose_to_rt(cur.pose, rotation_order)
    futures = []
    for j in range(idx + 1, len(states)):
        fut = states[j]
        dt = (fut.timestamp - cur.timestamp).total_seconds()
        if dt <= 0:
            continue
        if dt > horizon_seconds:
            break
        r_fut, t_fut = pose_to_rt(fut.pose, rotation_order)
        rel_r, rel_t = relative_rt(r_cur, t_cur, r_fut, t_fut)
        rel_r, rel_t = convert_flu_to_frame(rel_r, rel_t, output_frame)
        e1, e2, e3 = rotation_matrix_to_euler_xyz(rel_r)
        futures.append(
            {
                "future_file": fut.path.name,
                "future_timestamp": fut.timestamp.isoformat(),
                "delta_t_seconds": dt,
                "relative_position_xyz": {"x": rel_t[0], "y": rel_t[1], "z": rel_t[2]},
                "relative_rotation_matrix": rel_r,
                "relative_rotation_euler_xyz": {"rot1": e1, "rot2": e2, "rot3": e3},
            }
        )
    if not futures:
        return None
    return {
        "source_file": cur.path.name,
        "source_timestamp": cur.timestamp.isoformat(),
        "horizon_seconds": horizon_seconds,
        "future_poses": futures,
        "output_frame": output_frame,
    }


def main() -> None:
    args = parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = load_frame_states(input_dir, args.input_glob, args.timestamp_source)
    if not states:
        print("No input JSON files found. Nothing to do.")
        return
    for i, state in tqdm(enumerate(states)):
        payload = build_output_for_frame(
            i,
            states,
            args.horizon_seconds,
            args.rotation_order,
            args.output_frame,
        )
        with (output_dir / state.path.name).open("w", encoding="utf-8") as f:
            json.dump(payload if payload is not None else {}, f, indent=2)
    print(f"Processed {len(states)} files. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
