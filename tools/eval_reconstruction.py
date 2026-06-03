import numpy as np
import open3d as o3d
import trimesh
import argparse, json, sys, os

def load_as_o3d_pointcloud(path, n_samples):
    """Load any mesh (including quad meshes) via trimesh, then sample."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    # trimesh handles quads, obj, ply with polygons — Open3D often can't
    mesh = trimesh.load(path, force='mesh', process=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    if len(mesh.faces) == 0:
        raise RuntimeError(f"No faces loaded from: {path}")

    print(f"  {os.path.basename(path)}: {len(mesh.faces)} faces, {len(mesh.vertices)} verts")

    # sample surface points
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd

def evaluate(pred_path, gt_path, n_samples=200000, thresholds=[0.05, 0.1], z_shift=1.5):
    pred_pc = load_as_o3d_pointcloud(pred_path, n_samples)
    gt_pc   = load_as_o3d_pointcloud(gt_path,   n_samples)

    # compensate for the +1.5m z-shift applied during fragment generation
    # NeuralRecon output is shifted up; bring GT up to match
    gt_pts = np.asarray(gt_pc.points)
    gt_pts[:, 2] += z_shift
    gt_pc.points = o3d.utility.Vector3dVector(gt_pts)

    d_pred_to_gt = np.asarray(pred_pc.compute_point_cloud_distance(gt_pc))
    d_gt_to_pred = np.asarray(gt_pc.compute_point_cloud_distance(pred_pc))

    results = {
        'accuracy'     : float(d_pred_to_gt.mean()),
        'completeness' : float(d_gt_to_pred.mean()),
        'chamfer'      : float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2),
    }

    for t in thresholds:
        precision = float((d_pred_to_gt < t).mean())
        recall    = float((d_gt_to_pred < t).mean())
        f = 2 * precision * recall / (precision + recall + 1e-10)
        results[f'precision@{int(t*100)}cm'] = precision
        results[f'recall@{int(t*100)}cm']    = recall
        results[f'fscore@{int(t*100)}cm']    = f

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',    required=True)
    parser.add_argument('--gt',      required=True)
    parser.add_argument('--scene',   default='unknown')
    parser.add_argument('--z_shift', type=float, default=1.5,
                        help='z offset applied during fragment generation (default 1.5)')
    args = parser.parse_args()

    print(f"\nEvaluating: {args.scene}")
    print(f"  pred: {args.pred}")
    print(f"  gt:   {args.gt}")

    try:
        results = evaluate(args.pred, args.gt, z_shift=args.z_shift)
        results['scene'] = args.scene
        print(json.dumps(results, indent=2))
        out = 'results/results_metrics.jsonl'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'a') as f:
            f.write(json.dumps(results) + '\n')
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)