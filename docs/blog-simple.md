# R9700 跑 TRELLIS.2 ROCm：先把能跑的路徑整理出來

這份工具箱不是要把 TRELLIS.2 重新包成一個完整產品。比較實際的目標是：在 AMD Radeon AI PRO R9700 這張卡上，先整理出一條可以重現的 image-to-3D 測試流程。

R9700 的硬體規格看起來很適合做本地 AI Lab，但 ROCm、RDNA4、3D 生成模型這幾個東西湊在一起，細節其實不少。官方或社群專案通常會先以 NVIDIA / CUDA 當主要路徑，AMD 這邊常常要自己補一段。

## 為什麼要另外整理一包

我測的是 [TRELLIS.2 的 ROCm fork](https://github.com/Cardboard-box-a/TRELLIS.2_rocm)。模型本身可以跑，問題主要卡在高品質輸出時的貼圖與 mesh 後處理。

原本的 textured GLB export 會走 GPU BVH 路徑，其中有一段 `cumesh.cuBVH.unsigned_distance()`。在 R9700 / gfx1201 的 ROCm 環境下，這段有機會讓 HIP 進入 illegal state。不是模型完全不能跑，而是輸出流程跑到這裡會炸。

所以這個 repo 做了兩件事：

- 固定一個可以重現的 ROCm 容器環境
- 把貼圖投影改成 CPU KDTree fallback，避開目前不穩的 GPU BVH 路徑

這不是最快的做法，但至少可以把 textured GLB 生出來。

## 目前可用的路徑

目前測過比較穩的設定是：

```text
texture_size=4096
decimation_target=1000000
remesh=False
OVOXEL_PROJECTION_MODE=cpu_kdtree
OVOXEL_CPU_KDTREE_K=8
```

`remesh=False` 是刻意的。原本 `remesh=True` 看起來比較漂亮，但它還是會碰到同一條 GPU BVH distance path。只要那段 native ROCm extension 還沒修好，高品質設定就不能只看參數名稱，還要看它底下實際呼叫了什麼。

## 測試紀錄：robot 4096 版

這次測試用 4096 texture 設定，目標是先確認高解析貼圖能不能穩定完成，而不是追求最快速度。

設定如下：

```text
texture_size=4096
decimation_target=1000000
remesh=False
OVOXEL_PROJECTION_MODE=cpu_kdtree
OVOXEL_CPU_KDTREE_K=8
```

輸出結果：

```text
GLB size: 約 41MB
final mesh: 823,375 vertices / 954,302 faces
valid texture pixels: 8,331,054
CPU projection mean distance: 3.255e-05
CPU projection max distance: 0.003561
GLB check: materials=1, textures=2, images=2, baseColorTexture exists
```

這次 robot 4096 版大約花了 6 分 50 秒。

時間拆開看：

| 階段 | 時間 | 備註 |
| --- | ---: | --- |
| 啟動 / 載入 pipeline | 約 1 分鐘多 | 第一次啟動會花時間載入模型與 pipeline |
| 模型生成 / sampling / decode | 約 2 分 50 秒 | 從開始跑圖到 `to_glb` 開始前 |
| GLB 匯出總時間 | 約 2 分 49 秒 | 22:23:22 開始 `to_glb`，22:26:11 完成 |
| 4096 texture baking + CPU projection | 約 2 分 35 秒 | 4096 貼圖解析度下最重的一段 |
| CPU KDTree projection 本身 | 約 2 分 23 秒 | 22:23:36 開始，22:25:59 完成 |

最耗時的是這段：

```text
CPU KDTree projection: querying 8,331,054 points
```

它要處理 833 萬個 texture points。換句話說，4096 貼圖解析度主要就是卡在這裡。CPU KDTree fallback 可以避開 ROCm GPU BVH 的問題，但代價就是 texture baking 會變成 CPU 工作。

## 使用方式

先 build 容器：

```bash
podman build -t localhost/r9700-trellis2-rocm-toolbox:latest .
```

模型不要放進 repo，也不要打進 image。建議把模型目錄掛到 `/models`：

```bash
MODEL_ROOT=$HOME/ai-models \
WORK_ROOT=$PWD/work \
scripts/run-container.sh
```

進容器後跑輸出：

```bash
cd /workspace/TRELLIS.2_rocm
source /workspace/.venv/bin/activate

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

## 這包適合誰

如果你只是想要最省事地跑 3D 生成，NVIDIA 環境目前還是比較少坑。

但如果你手上已經有 R9700，或是想測 AMD AI 生態，這包可以省掉一些重複踩坑的時間。至少不用從「為什麼貼圖 export 會壞」開始查。

這個 repo 是公開的，所以裡面只放 source、patch、腳本和說明。模型權重、Hugging Face token、私有 IP、VM 名稱、個人路徑都不要放進來。

## 後續想補的東西

- 把 Web UI 的流程也整理成可重現版本
- 補一份 build 時間與生成時間紀錄
- 測不同 `texture_size` 對品質與時間的影響
- 等 ROCm / native extension 更新後，再回頭測 `remesh=True`
