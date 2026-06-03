#!/usr/bin/env python3
"""
convert_replica.py  —  converts a Replica scene to NeuralRecon demo format

Usage:
    python convert_replica.py --src ../Replica/office0 --dst ../Replica_converted/office0
"""
import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

# ── Replica intrinsics (NICE-SLAM version) ──────────────────────────────────
# Change these if your resolution differs from 680×1200
INTRINSICS = {
    (680, 1200): dict(fx=600.0,   fy=600.0,   cx=599.5, cy=339.5),
    (512, 512):  dict(fx=320.0,   fy=320.0,   cx=256.0, cy=256.0),
    (480, 640):  dict(fx=320.0,   fy=320.0,   cx=319.5, cy=239.5),
}

def make_intrinsic_matrix(fx, fy, cx, cy):
    K = np.eye(4)
    K[0, 0] = fx;  K[1, 1] = fy
    K[0, 2] = cx;  K[1, 2] = cy
    return K

def convert(src: Path, dst: Path):
    # ── output folders ──────────────────────────────────────────────────────
    color_dir = dst / 'color'
    depth_dir = dst / 'depth'
    pose_dir  = dst / 'pose'
    intr_dir  = dst / 'intrinsic'
    for d in [color_dir, depth_dir, pose_dir, intr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── sort frames & depth maps ─────────────────────────────────────────────
    rgb_files   = sorted((src / 'results').glob('frame*.jpg'))
    depth_files = sorted((src / 'results').glob('depth*.png'))

    assert len(rgb_files) == len(depth_files), \
        f"Frame/depth count mismatch: {len(rgb_files)} vs {len(depth_files)}"
    print(f"Found {len(rgb_files)} frames")

    # ── detect resolution & pick intrinsics ──────────────────────────────────
    W, H = Image.open(rgb_files[0]).size
    if (H, W) not in INTRINSICS:
        raise ValueError(
            f"Unknown resolution {H}×{W}. "
            f"Add it to the INTRINSICS dict in this script."
        )
    K = make_intrinsic_matrix(**INTRINSICS[(H, W)])
    np.savetxt(intr_dir / 'intrinsic_color.txt',  K, fmt='%.6f')
    np.savetxt(intr_dir / 'intrinsic_depth.txt',  K, fmt='%.6f')
    print(f"Resolution {H}×{W}  →  fx={INTRINSICS[(H,W)]['fx']}")

    # ── load all poses (1 line = 16 floats = row-major 4×4 c2w) ─────────────
    traj_path = src / 'traj.txt'
    poses_c2w = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            assert len(vals) == 16, f"Expected 16 values, got {len(vals)}"
            poses_c2w.append(np.array(vals).reshape(4, 4))

    assert len(poses_c2w) == len(rgb_files), \
        f"Pose/frame count mismatch: {len(poses_c2w)} vs {len(rgb_files)}"

    # ── copy frames and write poses ──────────────────────────────────────────
    for idx, (rgb_f, dep_f, c2w) in enumerate(zip(rgb_files, depth_files, poses_c2w)):
        name = f'frame-{idx:06d}'

        # RGB
        shutil.copy(rgb_f, color_dir / f'{name}.jpg')

        # Depth (keep as-is; NeuralRecon ignores depth at demo time)
        shutil.copy(dep_f, depth_dir / f'{name}.png')

        # Pose: NeuralRecon expects camera-to-world (c2w) as a 4×4 txt
        np.savetxt(pose_dir / f'{name}.txt', c2w, fmt='%.10f')

    print(f"Done. Written to {dst}")
    print(f"  color/   : {len(rgb_files)} images")
    print(f"  depth/   : {len(depth_files)} images")
    print(f"  pose/    : {len(poses_c2w)} txt files")
    print(f"  intrinsic: intrinsic_color.txt + intrinsic_depth.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='path to Replica scene (e.g. ../Replica/office0)')
    parser.add_argument('--dst', required=True, help='output path for converted scene')
    args = parser.parse_args()
    convert(Path(args.src), Path(args.dst))