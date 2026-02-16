import sys
import os
import torch
import cv2
import numpy as np

# Add LATR repo root to Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)

# Import from LATR
from utils.config import Config
from models import build_model
from utils.transforms import build_augmentations

def preprocess(image_path, input_size):
    """
    Read image, resize & normalize appropriately for LATR.
    Uses repo's augmentations if available.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, tuple(input_size))

    # Use LATR augmentation pipeline for inference
    aug = build_augmentations()  # this applies normalization
    processed = aug(image=img_resized)["image"]

    # convert to torch tensor (BCHW)
    img_tensor = torch.from_numpy(processed).float().permute(2, 0, 1).unsqueeze(0)
    return img_bgr, img_tensor

def visualize(output_img, lanes, out_file="output.jpg"):
    """
    Draw predicted lanes back onto image.
    LATR outputs 3D lanes; here we draw projected points.
    """
    for lane in lanes:
        # each lane might be shape (N,3) [x,y,z]; ignore z for 2D drawing
        pts = lane.cpu().numpy()
        for (x, y, z) in pts:
            cv2.circle(output_img, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imwrite(out_file, output_img)
    print(f"Saved results to {out_file}")

def main():
    # ----- Model setup -----
    cfg = Config("config/openlane_1000.py")  # use your specific config
    model = build_model(cfg.model)
    model.cuda().eval()

    ckpt = torch.load("pretrained_models/openlane_1000.pth.tar")
    model.load_state_dict(ckpt["state_dict"])

    # ----- Load and preprocess image -----
    img_path = "shounak_scripts/image.png"
    orig_img, img_tensor = preprocess(img_path, cfg.input_size)
    img_tensor = img_tensor.cuda()

    # ----- Inference -----
    with torch.no_grad():
        outputs = model(img_tensor)

    print("Raw model outputs:", outputs)

    # ----- Visualization -----
    # LATR outputs often include "lanes" key with 3D coordinates
    if "lanes" in outputs:
        visualize(orig_img, outputs["lanes"], out_file="output_with_lanes.jpg")
    else:
        print("No 'lanes' key in outputs; inspect raw output format.")

if __name__ == "__main__":
    main()

