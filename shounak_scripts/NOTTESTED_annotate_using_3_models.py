import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.utils import Config, DictAction
from torchvision import transforms

# NOTE:
# LATR -> utils.utils imports scipy at import-time. Older scipy releases rely on
# NumPy aliases/attributes removed in NumPy>=2.0 (e.g. np.int, np.typeDict).
# Patch them before importing project modules so this script can still start.
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

    # Old SciPy sometimes expects these dictionaries.
    if not hasattr(np, "typeDict"):
        np.typeDict = np.sctypeDict


_patch_numpy_compat()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.latr import LATR
from utils.utils import (
    homography_crop_resize,
    projection_g2im,
    projection_g2im_extrinsic,
    projective_transformation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LATR inference on custom images and visualize 3D predicted lanes."
    )
    parser.add_argument("--config", required=True, help="Model config file.")
    parser.add_argument("--checkpoint", required=True, help="Pretrained model checkpoint (.pth/.pth.tar).")
    parser.add_argument(
        "--extra-model",
        nargs=2,
        metavar=("CONFIG", "CHECKPOINT"),
        action="append",
        default=[],
        help=(
            "Additional model pair for ensemble voting. "
            "Use multiple times, e.g. --extra-model cfg2.py ckpt2.pth --extra-model cfg3.py ckpt3.pth"
        ),
    )
    parser.add_argument("--input-dir", required=True, help="Directory with custom input images.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations/predictions.")
    parser.add_argument(
        "--calib-dir",
        default=None,
        help=(
            "Optional directory with per-image calibration JSON files using the same stem as each image. "
            "Each calibration file can contain intrinsic/extrinsic, or cam_pitch/cam_height."
        ),
    )
    parser.add_argument(
        "--calib-json",
        default=None,
        help="Optional global calibration JSON file used for all images.",
    )
    parser.add_argument(
        "--img-exts",
        nargs="+",
        default=["jpg", "jpeg", "png", "bmp"],
        help="Image extensions to scan under input-dir.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Confidence threshold for lane queries.")
    parser.add_argument(
        "--vote-dist-thresh",
        type=float,
        default=2.0,
        help="3D distance threshold (meters) to count a lane match across models.",
    )
    parser.add_argument(
        "--zigzag-angle-thresh-deg",
        type=float,
        default=45.0,
        help="Max allowed segment direction change (degrees) inside one lane before filtering as zigzag.",
    )
    parser.add_argument(
        "--zigzag-std-thresh-deg",
        type=float,
        default=18.0,
        help="Max allowed standard deviation of segment directions (degrees) for one lane.",
    )
    parser.add_argument("--max-images", type=int, default=-1, help="Max number of images to run. -1 means all.")
    parser.add_argument("--device", default="cuda:1", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config settings, e.g. --cfg-options resize_h=720 resize_w=960",
    )
    return parser.parse_args()


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)


def _load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _find_calibration(
    image_path: Path,
    calib_dir: Optional[Path],
    global_calib: Optional[Dict],
) -> Dict:
    if calib_dir is not None:
        calib_file = calib_dir / f"{image_path.stem}.json"
        if calib_file.exists():
            return _load_json(calib_file)
    if global_calib is not None:
        return global_calib
    return {}


def _as_np3x3(matrix: Optional[List]) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape == (3, 4):
        arr = arr[:, :3]
    if arr.shape != (3, 3):
        raise ValueError(f"Intrinsic must be 3x3 or 3x4, got {arr.shape}")
    return arr


def _as_np4x4(matrix: Optional[List]) -> Optional[np.ndarray]:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Extrinsic must be 4x4, got {arr.shape}")
    return arr


def build_projection_matrix(cfg: Config, calib: Dict) -> Tuple[np.ndarray, Dict]:
    intrinsic = _as_np3x3(calib.get("intrinsic") or calib.get("calibration"))
    if intrinsic is None:
        intrinsic = np.asarray(cfg.K, dtype=np.float32)

    extrinsic = _as_np4x4(calib.get("extrinsic"))
    cam_pitch = calib.get("cam_pitch", np.pi / 180.0 * float(getattr(cfg, "pitch", 0.0)))
    cam_height = calib.get("cam_height", float(getattr(cfg, "cam_height", 1.55)))

    if extrinsic is not None:
        proj = projection_g2im_extrinsic(extrinsic, intrinsic)
    else:
        proj = projection_g2im(float(cam_pitch), float(cam_height), intrinsic)

    h_crop = homography_crop_resize(
        [int(cfg.org_h), int(cfg.org_w)], int(cfg.crop_y), [int(cfg.resize_h), int(cfg.resize_w)]
    )
    m = np.matmul(h_crop, proj)

    meta = {
        "intrinsic": intrinsic.tolist(),
        "extrinsic": extrinsic.tolist() if extrinsic is not None else None,
        "cam_pitch": float(cam_pitch),
        "cam_height": float(cam_height),
    }
    return m.astype(np.float32), meta


def preprocess_image(image_bgr: np.ndarray, cfg: Config) -> Tuple[torch.Tensor, np.ndarray]:
    cropped = image_bgr[int(cfg.crop_y):, :, :]
    resized = cv2.resize(cropped, (int(cfg.resize_w), int(cfg.resize_h)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=list(cfg.get("mean", [0.485, 0.456, 0.406])),
        std=list(cfg.get("std", [0.229, 0.224, 0.225])),
    )
    tensor = normalize(to_tensor(rgb).float())
    return tensor, resized


def build_extra_dict(cfg: Config, lidar2img: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    h, w = int(cfg.resize_h), int(cfg.resize_w)
    max_lanes = int(cfg.max_lanes)
    num_y_steps = int(cfg.num_y_steps)
    num_category = int(cfg.num_category)
    anchor_dim = 3 * num_y_steps + num_category

    seg_label = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    seg_idx_label = torch.zeros((max_lanes, h, w), dtype=torch.float32, device=device)
    ground_lanes = torch.zeros((max_lanes, anchor_dim), dtype=torch.float32, device=device)
    num_y_steps_dense = len(np.asarray(cfg.anchor_y_steps_dense, dtype=np.float32))
    ground_lanes_dense = torch.zeros((max_lanes, 3 * num_y_steps_dense), dtype=torch.float32, device=device)

    extra_dict = {
        "seg_label": seg_label.unsqueeze(0),
        "seg_idx_label": seg_idx_label.unsqueeze(0),
        "ground_lanes": ground_lanes.unsqueeze(0),
        "ground_lanes_dense": ground_lanes_dense.unsqueeze(0),
        "lidar2img": torch.from_numpy(lidar2img).to(device).unsqueeze(0),
        "pad_shape": torch.tensor([h, w], dtype=torch.float32, device=device).unsqueeze(0),
    }
    return extra_dict


def decode_prediction(output: Dict, cfg: Config, score_thresh: float) -> Tuple[List[List[List[float]]], List[float]]:
    lane_pred = output["all_line_preds"][-1][0].detach().cpu().numpy()
    cls_score = output["all_cls_scores"][-1][0]

    if int(cfg.num_category) > 1:
        cls_idx = torch.argmax(cls_score, dim=-1)
        prob = torch.softmax(cls_score, dim=-1).max(dim=-1).values
        keep = (cls_idx > 0) & (prob > score_thresh)
    else:
        prob = torch.sigmoid(cls_score.squeeze(-1))
        keep = prob > score_thresh

    keep = keep.detach().cpu().numpy()
    probs = prob.detach().cpu().numpy()[keep].tolist()
    lanes_raw = lane_pred[keep]

    lanes_3d = []
    anchor_y = np.asarray(cfg.anchor_y_steps, dtype=np.float32)
    n_steps = int(cfg.num_y_steps)

    for lane in lanes_raw:
        xs = lane[0:n_steps]
        zs = lane[n_steps:2 * n_steps]
        vis = lane[2 * n_steps:3 * n_steps] > 0
        if vis.sum() < 2:
            continue
        lane_pts = np.stack([xs[vis], anchor_y[vis], zs[vis]], axis=-1)
        lanes_3d.append(lane_pts.astype(float).tolist())

    if len(lanes_3d) != len(probs):
        probs = probs[: len(lanes_3d)]
    return lanes_3d, probs

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
        if len(support_models) >= 2:
            supported.append(lane_i)

    if not supported:
        return [], []

    deduped = []
    for lane in sorted(supported, key=lambda x: x["score"], reverse=True):
        if any(lane_distance_3d(lane["lane"], kept["lane"]) <= vote_dist_thresh * 0.6 for kept in deduped):
            continue
        deduped.append(lane)

    return [d["lane"] for d in deduped], [d["score"] for d in deduped]

def draw_lane_overlay(image_bgr: np.ndarray, lanes_3d: List[List[List[float]]], proj_m: np.ndarray) -> np.ndarray:
    canvas = image_bgr.copy()
    for lane in lanes_3d:
        lane_arr = np.asarray(lane, dtype=np.float32)
        xs, ys = projective_transformation(proj_m, lane_arr[:, 0], lane_arr[:, 1], lane_arr[:, 2])
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), color=(0, 255, 255), thickness=2)
    return canvas


def save_visualization(
    image_path: Path,
    out_path: Path,
    overlay_bgr: np.ndarray,
    lanes_3d: List[List[List[float]]],
    probs: List[float],
) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"2D projection - {image_path.name}")
    ax_img.axis("off")

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    for lane, score in zip(lanes_3d, probs):
        arr = np.asarray(lane, dtype=np.float32)
        ax_3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=2, label=f"{score:.2f}")

    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")
    ax_3d.set_title("Predicted 3D lanes")

    if lanes_3d:
        ax_3d.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


def collect_images(input_dir: Path, exts: List[str]) -> List[Path]:
    patterns = [f"*.{ext.lower()}" for ext in exts] + [f"*.{ext.upper()}" for ext in exts]
    images = []
    for pattern in patterns:
        images.extend(sorted(input_dir.glob(pattern)))
    dedup = sorted(set(images))
    return dedup


def main() -> None:
    args = parse_args()


    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    model_specs = [(args.config, args.checkpoint)] + [tuple(x) for x in args.extra_model]
    models = []
    cfgs = []
    for cfg_path, ckpt_path in model_specs:
        cfg = Config.fromfile(cfg_path)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        if not hasattr(cfg, "anchor_y_steps_dense"):
            cfg.anchor_y_steps_dense = np.linspace(3, 103, 200, dtype=np.float32)
        else:
            cfg.anchor_y_steps_dense = np.asarray(cfg.anchor_y_steps_dense, dtype=np.float32)
        model = LATR(cfg).to(device)
        model.eval()
        _load_checkpoint(model, ckpt_path, device)
        cfgs.append(cfg)
        models.append(model)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis"
    pred_dir = output_dir / "pred"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    global_calib = _load_json(Path(args.calib_json)) if args.calib_json else None

    image_paths = collect_images(input_dir, args.img_exts)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir} with extensions {args.img_exts}")

    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths)):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"[WARN] Skip unreadable image: {image_path}")
                continue

            calib = _find_calibration(image_path, calib_dir, global_calib)
            all_model_lanes = []
            all_model_probs = []
            resized_bgr = None
            proj_m = None
            calib_meta = None

            for cfg, model in zip(cfgs, models):
                proj_m_cur, calib_meta_cur = build_projection_matrix(cfg, calib)
                image_tensor, resized_bgr_cur = preprocess_image(image_bgr, cfg)
                lidar2img = np.eye(4, dtype=np.float32)
                lidar2img[:3] = proj_m_cur

                extra_dict = build_extra_dict(cfg, lidar2img, device)
                image_tensor = image_tensor.unsqueeze(0).to(device)

                output = model(image=image_tensor, extra_dict=extra_dict, is_training=False)
                lanes_3d, probs = decode_prediction(output, cfg, args.score_thresh)
                all_model_lanes.append(lanes_3d)
                all_model_probs.append(probs)

                if resized_bgr is None:
                    resized_bgr = resized_bgr_cur
                    proj_m = proj_m_cur
                    calib_meta = calib_meta_cur

            lanes_3d, probs = refine_by_voting(
                model_lanes=all_model_lanes,
                model_probs=all_model_probs,
                vote_dist_thresh=args.vote_dist_thresh,
                zigzag_angle_thresh_deg=args.zigzag_angle_thresh_deg,
                zigzag_std_thresh_deg=args.zigzag_std_thresh_deg,
            )
            if not lanes_3d:
                print(f"[{idx + 1}/{len(image_paths)}] skipped (no refined annotations): {image_path.name}")
                continue

            overlay = draw_lane_overlay(resized_bgr, lanes_3d, proj_m)
            save_visualization(
                image_path=image_path,
                out_path=vis_dir / f"{image_path.stem}_vis.png",
                overlay_bgr=overlay,
                lanes_3d=lanes_3d,
                probs=probs,
            )

            pred_payload = {
                "image": str(image_path),
                "pred_laneLines": lanes_3d,
                "pred_laneLines_prob": probs,
                "calibration": calib_meta,
            }
            with (pred_dir / f"{image_path.stem}.json").open("w") as f:
                json.dump(pred_payload, f, indent=2)

            print(f"[{idx + 1}/{len(image_paths)}] saved: {vis_dir / f'{image_path.stem}_vis.png'}")


if __name__ == "__main__":
    main()
