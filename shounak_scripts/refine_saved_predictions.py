import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _patch_numpy_compat() -> None:
    alias_map = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for alias, target in alias_map.items():
        if not hasattr(np, alias):
            setattr(np, alias, target)

    if not hasattr(np, "typeDict"):
        np.typeDict = np.sctypeDict


_patch_numpy_compat()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmcv.utils import Config, DictAction  # noqa: E402
from utils.utils import (  # noqa: E402
    homography_crop_resize,
    projection_g2im,
    projection_g2im_extrinsic,
    projective_transformation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine already-generated LATR prediction JSON files from multiple model runs "
            "by zigzag filtering + cross-model 3D voting."
        )
    )
    parser.add_argument("--pred-dirs", nargs="+", required=True, help="Directories containing model prediction JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save refined JSONs and visualizations.")
    parser.add_argument("--image-root", default=None, help="Optional root used to resolve relative image paths from JSON.")
    parser.add_argument("--vote-dist-thresh", type=float, default=2.0)
    parser.add_argument("--zigzag-angle-thresh-deg", type=float, default=45.0)
    parser.add_argument("--zigzag-std-thresh-deg", type=float, default=18.0)
    parser.add_argument("--min-support-models", type=int, default=2)
    parser.add_argument("--skip-vis", action="store_true", help="Only write refined JSON outputs.")

    parser.add_argument("--config", required=True, help="Same model config used by infer_custom.py (for exact projection geometry).")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config settings, e.g. --cfg-options resize_h=720 resize_w=960",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _as_np3x3(matrix: Optional[Sequence]) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape == (3, 4):
        arr = arr[:, :3]
    if arr.shape != (3, 3):
        raise ValueError(f"Intrinsic must be 3x3 or 3x4, got {arr.shape}")
    return arr


def _as_np4x4(matrix: Optional[Sequence]) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Extrinsic must be 4x4, got {arr.shape}")
    return arr


def _lane_direction_angles_deg(lane_arr: np.ndarray) -> np.ndarray:
    diffs = np.diff(lane_arr[:, :2], axis=0)
    seg_norm = np.linalg.norm(diffs, axis=1)
    valid = seg_norm > 1e-4
    if valid.sum() < 2:
        return np.empty((0,), dtype=np.float32)
    diffs = diffs[valid]
    return np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1]))


def is_zigzag_lane(lane: Sequence[Sequence[float]], angle_thresh_deg: float, std_thresh_deg: float) -> bool:
    lane_arr = np.asarray(lane, dtype=np.float32)
    if lane_arr.shape[0] < 4:
        return True
    angles = _lane_direction_angles_deg(lane_arr)
    if angles.size < 2:
        return True
    angle_delta = np.abs(np.diff(angles))
    angle_delta = np.minimum(angle_delta, 360.0 - angle_delta)
    if angle_delta.size == 0:
        return False
    return bool(angle_delta.max() > angle_thresh_deg or float(np.std(angles)) > std_thresh_deg)


def lane_distance_3d(lane_a: Sequence[Sequence[float]], lane_b: Sequence[Sequence[float]]) -> float:
    arr_a = np.asarray(lane_a, dtype=np.float32)
    arr_b = np.asarray(lane_b, dtype=np.float32)
    if arr_a.shape[0] < 2 or arr_b.shape[0] < 2:
        return float("inf")

    y_min = max(float(np.min(arr_a[:, 1])), float(np.min(arr_b[:, 1])))
    y_max = min(float(np.max(arr_a[:, 1])), float(np.max(arr_b[:, 1])))
    if y_max - y_min < 1.0:
        return float("inf")

    sample_y = np.linspace(y_min, y_max, num=20, dtype=np.float32)
    xa = np.interp(sample_y, arr_a[:, 1], arr_a[:, 0])
    za = np.interp(sample_y, arr_a[:, 1], arr_a[:, 2])
    xb = np.interp(sample_y, arr_b[:, 1], arr_b[:, 0])
    zb = np.interp(sample_y, arr_b[:, 1], arr_b[:, 2])

    dist = np.sqrt((xa - xb) ** 2 + (za - zb) ** 2)
    return float(np.mean(dist))


def refine_by_voting(
    model_lanes: Sequence[Sequence[Sequence[Sequence[float]]]],
    model_probs: Sequence[Sequence[float]],
    vote_dist_thresh: float,
    zigzag_angle_thresh_deg: float,
    zigzag_std_thresh_deg: float,
    min_support_models: int,
) -> Tuple[List[List[List[float]]], List[float]]:
    candidates = []
    for model_idx, (lanes, probs) in enumerate(zip(model_lanes, model_probs)):
        for lane, score in zip(lanes, probs):
            if is_zigzag_lane(lane, zigzag_angle_thresh_deg, zigzag_std_thresh_deg):
                continue
            candidates.append({"model": model_idx, "lane": lane, "score": float(score)})

    if not candidates:
        return [], []

    supported = []
    for lane_i in candidates:
        support_models = {lane_i["model"]}
        for lane_j in candidates:
            if lane_j["model"] == lane_i["model"]:
                continue
            if lane_distance_3d(lane_i["lane"], lane_j["lane"]) <= vote_dist_thresh:
                support_models.add(lane_j["model"])
        if len(support_models) >= min_support_models:
            supported.append(lane_i)

    if not supported:
        return [], []

    deduped = []
    for lane in sorted(supported, key=lambda x: x["score"], reverse=True):
        if any(lane_distance_3d(lane["lane"], kept["lane"]) <= vote_dist_thresh * 0.6 for kept in deduped):
            continue
        deduped.append(lane)

    return [d["lane"] for d in deduped], [d["score"] for d in deduped]


def _find_shared_stems(pred_dirs: Sequence[Path]) -> List[str]:
    stem_sets = [{p.stem for p in pred_dir.glob("*.json")} for pred_dir in pred_dirs]
    shared = set.intersection(*stem_sets) if stem_sets else set()
    return sorted(shared)


def _resolve_image_path(image_str: str, image_root: Optional[Path]) -> Path:
    p = Path(image_str)
    if p.is_absolute() and p.exists():
        return p
    if image_root is not None:
        candidate = image_root / p
        if candidate.exists():
            return candidate
    return p




def _load_projection_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    return cfg


def _build_projection_matrix(calibration: Dict, cfg: Config) -> np.ndarray:
    intrinsic = _as_np3x3(calibration.get("intrinsic") or calibration.get("calibration"))
    if intrinsic is None:
        raise ValueError("Calibration must contain intrinsic/calibration matrix for visualization.")

    extrinsic = _as_np4x4(calibration.get("extrinsic"))
    cam_pitch = float(calibration.get("cam_pitch", 0.0))
    cam_height = float(calibration.get("cam_height", 1.55))

    if extrinsic is not None:
        proj = projection_g2im_extrinsic(extrinsic, intrinsic)
    else:
        proj = projection_g2im(cam_pitch, cam_height, intrinsic)

    org_h = int(cfg.org_h)
    org_w = int(cfg.org_w)
    crop_y = int(cfg.crop_y)
    resize_h = int(cfg.resize_h)
    resize_w = int(cfg.resize_w)

    h_crop = homography_crop_resize([org_h, org_w], crop_y, [resize_h, resize_w])
    return np.matmul(h_crop, proj).astype(np.float32)


def _prepare_vis_image(image_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    crop_y = int(cfg.crop_y)
    cropped = image_bgr[crop_y:, :, :] if crop_y > 0 else image_bgr
    return cv2.resize(cropped, (int(cfg.resize_w), int(cfg.resize_h)), interpolation=cv2.INTER_LINEAR)


def _draw_overlay(image_bgr: np.ndarray, lanes_3d: Sequence[Sequence[Sequence[float]]], proj_m: np.ndarray) -> np.ndarray:
    canvas = image_bgr.copy()
    for lane in lanes_3d:
        lane_arr = np.asarray(lane, dtype=np.float32)
        if lane_arr.shape[0] < 2:
            continue
        xs, ys = projective_transformation(proj_m, lane_arr[:, 0], lane_arr[:, 1], lane_arr[:, 2])
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), (0, 255, 255), 2)
    return canvas


def main() -> None:
    args = parse_args()

    pred_dirs = [Path(p) for p in args.pred_dirs]
    for pred_dir in pred_dirs:
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    if not args.skip_vis and cv2 is None:
        raise RuntimeError("OpenCV (cv2) is not installed. Use --skip-vis to write only refined JSON outputs.")

    output_dir = Path(args.output_dir)
    pred_out = output_dir / "pred"
    vis_out = output_dir / "vis"
    pred_out.mkdir(parents=True, exist_ok=True)
    if not args.skip_vis:
        vis_out.mkdir(parents=True, exist_ok=True)

    cfg = _load_projection_cfg(args)
    image_root = Path(args.image_root) if args.image_root else None
    shared_stems = _find_shared_stems(pred_dirs)
    if not shared_stems:
        raise RuntimeError("No common JSON filenames found across --pred-dirs.")

    kept_images = 0
    for stem in tqdm(shared_stems, desc="Refining"):
        records = [_load_json(pred_dir / f"{stem}.json") for pred_dir in pred_dirs]

        model_lanes = [r.get("pred_laneLines", []) for r in records]
        model_probs = [r.get("pred_laneLines_prob", []) for r in records]

        refined_lanes, refined_probs = refine_by_voting(
            model_lanes=model_lanes,
            model_probs=model_probs,
            vote_dist_thresh=args.vote_dist_thresh,
            zigzag_angle_thresh_deg=args.zigzag_angle_thresh_deg,
            zigzag_std_thresh_deg=args.zigzag_std_thresh_deg,
            min_support_models=args.min_support_models,
        )
        if not refined_lanes:
            continue

        base_record = records[0]
        payload = {
            "image": base_record.get("image", ""),
            "pred_laneLines": refined_lanes,
            "pred_laneLines_prob": refined_probs,
            "calibration": base_record.get("calibration"),
            "source_pred_dirs": [str(p) for p in pred_dirs],
        }
        with (pred_out / f"{stem}.json").open("w") as f:
            json.dump(payload, f, indent=2)

        if not args.skip_vis:
            image_path = _resolve_image_path(str(base_record.get("image", "")), image_root)
            image_bgr = cv2.imread(str(image_path)) if image_path else None
            calib = base_record.get("calibration") or {}
            if image_bgr is not None and (calib.get("intrinsic") or calib.get("calibration")):
                vis_img = _prepare_vis_image(image_bgr, cfg)
                proj_m = _build_projection_matrix(calib, cfg)
                overlay = _draw_overlay(vis_img, refined_lanes, proj_m)
                cv2.imwrite(str(vis_out / f"{stem}_refined.png"), overlay)

        kept_images += 1

    print(f"Refinement complete. Kept {kept_images}/{len(shared_stems)} images with annotations.")


if __name__ == "__main__":
    main()
