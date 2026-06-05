import argparse
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:128")
os.environ.setdefault("HSA_XNACK", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
os.environ.setdefault("ROCM_SAFE_SPCONV", "1")
os.environ.setdefault("OVOXEL_PROJECTION_MODE", "cpu_kdtree")
os.environ.setdefault("OVOXEL_CPU_KDTREE_K", "8")

import cv2
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1000000)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.90)

    envmap = EnvMap(
        torch.tensor(
            cv2.cvtColor(cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device="cuda",
        )
    )
    del envmap

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    image = Image.open(args.input).convert("RGBA")
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
        use_tqdm=True,
    )
    glb.export(args.output, extension_webp=True)
    print(args.output)


if __name__ == "__main__":
    main()
