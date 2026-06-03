import os
import pickle
import numpy as np
from tqdm import tqdm

# Replica intrinsics for 1200x680 (NICE-SLAM version)
REPLICA_INTRINSICS = {
    (680, 1200): np.array([[600.0,   0.0, 599.5],
                           [  0.0, 600.0, 339.5],
                           [  0.0,   0.0,   1.0]]),
    (480, 640):  np.array([[320.0,   0.0, 319.5],
                           [  0.0, 320.0, 239.5],
                           [  0.0,   0.0,   1.0]]),
}

def process_replica(data_path, window_size=9, min_angle=15, min_distance=0.1):
    """
    Converts a Replica scene to NeuralRecon's fragments.pkl format.
    Replica structure expected:
        data_path/
            results/frame000000.jpg ...
            results/depth000000.png ...
            traj.txt   (1 line per frame, 16 floats, row-major c2w 4x4)
    """
    results_dir = os.path.join(data_path, 'results')

    # ── collect and sort frames ──────────────────────────────────────────────
    all_files = sorted(os.listdir(results_dir))
    rgb_files  = sorted([f for f in all_files if f.startswith('frame') and f.endswith('.jpg')])
    
    if len(rgb_files) == 0:
        raise FileNotFoundError(f"No frame*.jpg found in {results_dir}")
    print(f"Found {len(rgb_files)} frames")

    # ── detect resolution ────────────────────────────────────────────────────
    from PIL import Image
    sample_img = Image.open(os.path.join(results_dir, rgb_files[0]))
    W, H = sample_img.size
    if (H, W) not in REPLICA_INTRINSICS:
        raise ValueError(f"Unknown resolution {H}x{W}. Add it to REPLICA_INTRINSICS.")
    K = REPLICA_INTRINSICS[(H, W)]
    print(f"Resolution {H}x{W}, using fx={K[0,0]}, fy={K[1,1]}, cx={K[0,2]}, cy={K[1,2]}")

    # ── load poses ───────────────────────────────────────────────────────────
    traj_path = os.path.join(data_path, 'traj.txt')
    poses_c2w = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            assert len(vals) == 16, f"Expected 16 values per line, got {len(vals)}"
            poses_c2w.append(np.array(vals).reshape(4, 4))

    assert len(poses_c2w) == len(rgb_files), \
        f"Pose/frame count mismatch: {len(poses_c2w)} poses vs {len(rgb_files)} frames"

    # ── keyframe selection (same logic as process_arkit_data.py) ────────────
    all_ids = []
    ids     = []
    count   = 0
    last_pose = None

    for idx, (rgb_f, pose) in enumerate(tqdm(zip(rgb_files, poses_c2w),
                                              total=len(rgb_files),
                                              desc='Keyframe selection')):
        if count == 0:
            ids.append(idx)
            last_pose = pose
            count += 1
        else:
            angle = np.arccos(np.clip(
                ((np.linalg.inv(pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T)
                 * np.array([0, 0, 1])).sum(), -1.0, 1.0))
            dist = np.linalg.norm(pose[:3, 3] - last_pose[:3, 3])

            if angle > (min_angle / 180) * np.pi or dist > min_distance:
                ids.append(idx)
                last_pose = pose
                count += 1
                if count == window_size:
                    all_ids.append(ids)
                    ids   = []
                    count = 0

    print(f"Total fragments: {len(all_ids)}")
    if len(all_ids) == 0:
        raise RuntimeError("No fragments generated — check min_angle/min_distance thresholds.")

    # ── build fragments list ─────────────────────────────────────────────────
    scene_name = os.path.basename(data_path.rstrip('/'))
    fragments  = []

    for frag_id, frame_ids in enumerate(tqdm(all_ids, desc='Building fragments')):
        frag_poses      = []
        frag_intrinsics = []
        frag_image_ids  = []

        for idx in frame_ids:
            pose = poses_c2w[idx].copy()
            # NeuralRecon / ScanNet convention: shift z up by 1.5m
            # (same adjustment made in process_arkit_data.py)
            pose[2, 3] += 1.5
            frag_poses.append(pose)
            frag_intrinsics.append(K.copy())

            # image id = filename without extension, e.g. 'frame000042'
            frag_image_ids.append(rgb_files[idx].replace('.jpg', ''))

        fragments.append({
            'scene':       scene_name,
            'fragment_id': frag_id,
            'image_ids':   frag_image_ids,
            'extrinsics':  frag_poses,
            'intrinsics':  frag_intrinsics,
        })

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(data_path, 'fragments.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(fragments, f)
    print(f"Saved {len(fragments)} fragments → {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='path to Replica scene')
    parser.add_argument('--window_size', type=int, default=9)
    parser.add_argument('--min_angle',   type=float, default=15.0)
    parser.add_argument('--min_distance',type=float, default=0.1)
    args = parser.parse_args()
    process_replica(args.path, args.window_size, args.min_angle, args.min_distance)