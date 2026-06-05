# R9700 TRELLIS.2 ROCm Toolbox

Public toolbox for running TRELLIS.2 image-to-3D experiments on AMD Radeon AI PRO R9700 / gfx1201 with ROCm.

This repo is intentionally source-only. It does not include model weights, Hugging Face tokens, private hostnames, private IPs, VM names, or generated GLB files.

## Status

Tested path:

- Base image: `docker.io/kyuz0/amd-r9700-toolboxes:rocm-7.2.3`
- TRELLIS fork: `https://github.com/Cardboard-box-a/TRELLIS.2_rocm`
- Branch: `rocm`
- GPU target: `gfx1201`
- Textured GLB export: works with a local CPU KDTree projection fallback

Known limitation:

- `cumesh.cuBVH.unsigned_distance()` can fail on ROCm with `hipErrorIllegalState`.
- Because `remesh=True` depends on that BVH distance path, the tested high-quality export path uses `remesh=False`.
- The CPU KDTree projection fallback is slower than the original GPU BVH path, but it avoids the ROCm crash and keeps a real projection step.

## Repository Layout

```text
Containerfile
README.md
docs/
  rocm-notes.md
patches/
  conv-flex-gemm-api.patch
  o-voxel-cpu-kdtree-projection.patch
scripts/
  apply-runtime-patches.sh
  check-cubvh-rocm.py
  run-container.sh
  run-textured-export.py
```

## Build

```bash
podman build -t localhost/r9700-trellis2-rocm-toolbox:latest .
```

The build may take a long time because ROCm Flash Attention and native extensions are compiled or installed.

## Model Storage

Do not put model files in this repository or in the container image. Mount a host directory as `/models`.

Suggested layout:

```text
/models/huggingface
/models/cache
```

Download gated models with your own Hugging Face account. Do not commit tokens.

## Run Container

```bash
MODEL_ROOT=$HOME/ai-models \
WORK_ROOT=$PWD/work \
scripts/run-container.sh
```

The script mounts:

- `$MODEL_ROOT` to `/models`
- `$WORK_ROOT` to `/workspace/work`

## Textured Export Example

Inside the container:

```bash
cd /workspace/TRELLIS.2_rocm
source /workspace/.venv/bin/activate

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export ROCM_SAFE_SPCONV=1
export OVOXEL_PROJECTION_MODE=cpu_kdtree
export OVOXEL_CPU_KDTREE_K=8
export HF_HOME=/models/huggingface
export HUGGINGFACE_HUB_CACHE=/models/huggingface/hub
export XDG_CACHE_HOME=/models/cache

python /opt/r9700-trellis2/scripts/run-textured-export.py \
  --input /workspace/TRELLIS.2_rocm/assets/example_image/T.png \
  --output /workspace/work/sample-4096.glb \
  --texture-size 4096 \
  --decimation-target 1000000
```

## Privacy Rules

This is a public repo. Keep these out:

- private IPs and hostnames
- SSH usernames and paths from a personal machine
- Hugging Face tokens
- PVE/VM inventory details
- generated outputs and model weights
