import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    parser.add_argument("--max-images", type=int, default=-1, help="Max number of images to run. -1 means all.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
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
    ground_lanes_dense = torch.zeros((max_lanes, 3 * 200), dtype=torch.float32, device=device)

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

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    model = LATR(cfg).to(device)
    model.eval()
    _load_checkpoint(model, args.checkpoint, device)

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
        for idx, image_path in enumerate(image_paths):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"[WARN] Skip unreadable image: {image_path}")
                continue

            calib = _find_calibration(image_path, calib_dir, global_calib)
            proj_m, calib_meta = build_projection_matrix(cfg, calib)

            image_tensor, resized_bgr = preprocess_image(image_bgr, cfg)
            lidar2img = np.eye(4, dtype=np.float32)
            lidar2img[:3] = proj_m

            extra_dict = build_extra_dict(cfg, lidar2img, device)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            output = model(image=image_tensor, extra_dict=extra_dict, is_training=False)
            lanes_3d, probs = decode_prediction(output, cfg, args.score_thresh)

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
