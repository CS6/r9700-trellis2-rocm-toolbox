# ROCm Notes

## GPU BVH Issue

The upstream textured GLB path uses `cumesh.cuBVH.unsigned_distance()` to project UV texels back to the source mesh. On the tested ROCm/gfx1201 environment, that call can put HIP into an illegal state even for a tiny cube mesh.

Symptoms:

```text
torch.AcceleratorError: CUDA error: the operation cannot be performed in the present state
hipErrorIllegalState
```

The workaround in this toolbox is:

```bash
export OVOXEL_PROJECTION_MODE=cpu_kdtree
export OVOXEL_CPU_KDTREE_K=8
```

This bypasses the GPU BVH distance kernel during texture baking and uses a CPU KDTree plus closest-point-on-triangle projection.

## Quality Settings

Recommended high-quality fallback settings:

```text
texture_size=4096
decimation_target=1000000
remesh=False
OVOXEL_PROJECTION_MODE=cpu_kdtree
OVOXEL_CPU_KDTREE_K=8
```

Why `remesh=False`:

- The official `remesh=True` path depends on the same GPU BVH distance kernel.
- Until that native ROCm extension is fixed, `remesh=True` can still fail.

## Performance

4096 texture export can query millions of texels during CPU projection. Expect CPU projection to be the slowest step.
