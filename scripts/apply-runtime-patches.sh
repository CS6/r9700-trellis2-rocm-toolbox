#!/usr/bin/env bash
set -euo pipefail

TRELLIS_DIR="${1:-/workspace/TRELLIS.2_rocm}"
VENV_DIR="${2:-/workspace/.venv}"

python - "$TRELLIS_DIR" "$VENV_DIR" <<'PY'
from pathlib import Path
import sys

trellis_dir = Path(sys.argv[1])
venv_dir = Path(sys.argv[2])

conv = trellis_dir / "trellis2/modules/sparse/conv/conv_flex_gemm.py"
if conv.exists():
    text = conv.read_text()
    text = text.replace(
        "hashmap_build_submanifold_conv_neighbour_map_cuda(",
        "hashmap_build_submanifold_conv_neighbour_map(",
    )
    conv.write_text(text)

site_roots = list((venv_dir / "lib").glob("python*/site-packages"))
site_roots += list((venv_dir / "lib64").glob("python*/site-packages"))
if not site_roots:
    raise SystemExit(f"site-packages not found under {venv_dir}")

postprocess = None
for root in site_roots:
    candidate = root / "o_voxel/postprocess.py"
    if candidate.exists():
        postprocess = candidate
        break
if postprocess is None:
    raise SystemExit("o_voxel/postprocess.py not found")

text = postprocess.read_text()
if "_cpu_kdtree_project_to_mesh" not in text:
    marker = "def _log_cumesh(mesh, tag):\n"
    helper = r'''
def _closest_points_on_triangles_np(points, triangles):
    """Vectorized closest point from points [N, 3] to triangles [N, K, 3, 3]."""
    import numpy as np
    p = points[:, None, :]
    a = triangles[:, :, 0, :]
    b = triangles[:, :, 1, :]
    c = triangles[:, :, 2, :]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp = p - c
    d5 = np.sum(ab * cp, axis=-1)
    d6 = np.sum(ac * cp, axis=-1)
    closest = np.empty_like(a)
    mask_a = (d1 <= 0) & (d2 <= 0)
    closest[mask_a] = a[mask_a]
    mask_b = (d3 >= 0) & (d4 <= d3)
    closest[mask_b] = b[mask_b]
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~(mask_a | mask_b)
    v = d1 / np.maximum(d1 - d3, 1e-12)
    closest[mask_ab] = a[mask_ab] + ab[mask_ab] * v[mask_ab, None]
    mask_c = (d6 >= 0) & (d5 <= d6)
    closest[mask_c] = c[mask_c]
    vb = d5 * d2 - d1 * d6
    used = mask_a | mask_b | mask_ab | mask_c
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~used
    w = d2 / np.maximum(d2 - d6, 1e-12)
    closest[mask_ac] = a[mask_ac] + ac[mask_ac] * w[mask_ac, None]
    va = d3 * d6 - d5 * d4
    used = used | mask_ac
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~used
    w = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-12)
    closest[mask_bc] = b[mask_bc] + (c[mask_bc] - b[mask_bc]) * w[mask_bc, None]
    used = used | mask_bc
    mask_face = ~used
    denom = np.maximum(va + vb + vc, 1e-12)
    v = vb / denom
    w = vc / denom
    closest[mask_face] = a[mask_face] + ab[mask_face] * v[mask_face, None] + ac[mask_face] * w[mask_face, None]
    dist2 = np.sum((closest - p) ** 2, axis=-1)
    best = np.argmin(dist2, axis=1)
    return closest[np.arange(points.shape[0]), best], np.sqrt(dist2[np.arange(points.shape[0]), best])


def _cpu_kdtree_project_to_mesh(valid_pos, vertices, faces, k=8):
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    points_np = valid_pos.detach().cpu().numpy().astype(np.float32, copy=False)
    vertices_np = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    tri = vertices_np[faces_np]
    centroids = tri.mean(axis=1)
    tree = cKDTree(centroids)
    _, candidate_ids = tree.query(points_np, k=min(k, len(centroids)))
    if candidate_ids.ndim == 1:
        candidate_ids = candidate_ids[:, None]
    candidate_triangles = tri[candidate_ids]
    projected, distances = _closest_points_on_triangles_np(points_np, candidate_triangles)
    return (
        torch.from_numpy(projected.astype(np.float32, copy=False)).to(device=valid_pos.device),
        float(np.nanmean(distances)),
        float(np.nanmax(distances)),
    )

'''
    if marker not in text:
        raise SystemExit("could not find insertion point for CPU KDTree helper")
    text = text.replace(marker, helper + marker)

old = """    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    _info(f"  {elapsed()}  cumesh: building BVH  vertices={vertices.shape}  faces={faces.shape}")
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
"""
new = """    projection_mode = os.environ.get("OVOXEL_PROJECTION_MODE", "").strip().lower()
    needs_gpu_bvh = remesh or (projection_mode != "cpu_kdtree" and os.environ.get("OVOXEL_SKIP_BVH_CORRECTION", "0") != "1")

    # Build BVH for remeshing or the original GPU projection path.
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose and needs_gpu_bvh:
        print(f"Building BVH for current mesh...", end='', flush=True)
    if needs_gpu_bvh:
        _info(f"  {elapsed()}  cumesh: building BVH  vertices={vertices.shape}  faces={faces.shape}")
        bvh = cumesh.cuBVH(vertices, faces)
    else:
        _info(f"  {elapsed()}  cumesh: skipped GPU BVH build  projection_mode={projection_mode or 'default'}  remesh={remesh}")
        bvh = None
    if use_tqdm:
        pbar.update(1)
    if verbose and needs_gpu_bvh:
        print("Done")
"""
if old in text and new not in text:
    text = text.replace(old, new)

old = """    # Map these positions back to the *original* high-res mesh to get accurate attributes.
    # This corrects geometric errors introduced by simplification/remeshing.
    _info(f"  {elapsed()}  BVH unsigned_distance: querying {valid_pos.shape[0]} points")
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    _info(f"  {elapsed()}  BVH result: face_id range=[{face_id.min().item()},{face_id.max().item()}]  "
          f"faces_available={faces.shape[0]}")
    if face_id.max().item() >= faces.shape[0]:
        _error(f"  {elapsed()}  ⚠ BVH face_id OUT OF BOUNDS — face_id.max={face_id.max().item()}  faces={faces.shape[0]}")
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    _info(f"  {elapsed()}  BVH-corrected valid_pos: "
          f"x=[{valid_pos[:,0].min():.4g},{valid_pos[:,0].max():.4g}]  "
          f"y=[{valid_pos[:,1].min():.4g},{valid_pos[:,1].max():.4g}]  "
          f"z=[{valid_pos[:,2].min():.4g},{valid_pos[:,2].max():.4g}]")
"""
new = """    projection_mode = os.environ.get("OVOXEL_PROJECTION_MODE", "").strip().lower()
    if projection_mode == "cpu_kdtree":
        k = int(os.environ.get("OVOXEL_CPU_KDTREE_K", "8"))
        _info(f"  {elapsed()}  CPU KDTree projection: querying {valid_pos.shape[0]} points  k={k}")
        valid_pos, mean_dist, max_dist = _cpu_kdtree_project_to_mesh(valid_pos, vertices, faces, k=k)
        _info(f"  {elapsed()}  CPU KDTree projection done: mean_dist={mean_dist:.4g}  max_dist={max_dist:.4g}")
        _info(f"  {elapsed()}  CPU-projected valid_pos: "
              f"x=[{valid_pos[:,0].min():.4g},{valid_pos[:,0].max():.4g}]  "
              f"y=[{valid_pos[:,1].min():.4g},{valid_pos[:,1].max():.4g}]  "
              f"z=[{valid_pos[:,2].min():.4g},{valid_pos[:,2].max():.4g}]")
    else:
        # Map these positions back to the *original* high-res mesh to get accurate attributes.
        # This corrects geometric errors introduced by simplification/remeshing.
        _info(f"  {elapsed()}  BVH unsigned_distance: querying {valid_pos.shape[0]} points")
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        _info(f"  {elapsed()}  BVH result: face_id range=[{face_id.min().item()},{face_id.max().item()}]  "
              f"faces_available={faces.shape[0]}")
        if face_id.max().item() >= faces.shape[0]:
            _error(f"  {elapsed()}  ⚠ BVH face_id OUT OF BOUNDS — face_id.max={face_id.max().item()}  faces={faces.shape[0]}")
        orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        _info(f"  {elapsed()}  BVH-corrected valid_pos: "
              f"x=[{valid_pos[:,0].min():.4g},{valid_pos[:,0].max():.4g}]  "
              f"y=[{valid_pos[:,1].min():.4g},{valid_pos[:,1].max():.4g}]  "
              f"z=[{valid_pos[:,2].min():.4g},{valid_pos[:,2].max():.4g}]")
"""
if old in text and new not in text:
    text = text.replace(old, new)

if "skipped GPU BVH build" not in text:
    raise SystemExit("GPU BVH skip patch was not applied")
if "CPU KDTree projection:" not in text:
    raise SystemExit("CPU KDTree projection patch was not applied")

postprocess.write_text(text)
print(f"patched {conv}")
print(f"patched {postprocess}")
PY
