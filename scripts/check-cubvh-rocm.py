import torch
from cumesh import cuBVH

vertices = torch.tensor(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    device="cuda",
    dtype=torch.float32,
)
faces = torch.tensor(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    device="cuda",
    dtype=torch.int32,
)
points = torch.tensor([[0, 0, 0], [0.25, 0.25, 0.25]], device="cuda", dtype=torch.float32)

bvh = cuBVH(vertices, faces)
torch.cuda.synchronize()
dist, face_id, uvw = bvh.unsigned_distance(points, return_uvw=True)
torch.cuda.synchronize()
print("dist", dist.detach().cpu().tolist())
print("face_id", face_id.detach().cpu().tolist())
print("uvw", uvw.detach().cpu().tolist())
