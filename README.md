<div align="center">

# 🧰 npu-linux-kit

**Practical applications for the AMD XDNA NPU on Linux** — the things the NPU is *actually* good at,
in one kit. Built on the [open-xdna](https://github.com/Scottcjn/open-xdna) gen-1 (XDNA1 / Phoenix)
bring-up.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)
[![Dual License: Commercial](https://img.shields.io/badge/License-Commercial-green.svg)](COMMERCIAL.md)

</div>

---

## Why

AMD's NPU is underused on Linux: the official LLM stack targets XDNA2, and the NPU's real strengths
(**low-power, conv/vision, batchable compute, offload**) aren't packaged into usable Linux apps. This
kit does that — each module plays to a **measured** NPU strength, and the status of each is stated
honestly (no "coming soon" dressed up as done).

## Modules

| Module | What it does | NPU fit | Status |
|--------|--------------|---------|--------|
| **[camera/](camera/)** | Real-time webcam effects → a **v4l2 virtual camera** (Zoom/OBS/browser) | conv/vision — its native strength | ✅ **working** (edge-stylize) |
| camera/ — background blur | Sharp foreground + blurred background (real MODNet segmentation) | conv/CNN | ✅ **working** (clean CPU blur + real seg; NPU blur path experimental) |
| **[embeddings/](embeddings/)** | Batch sentence/RAG embeddings, offloaded from CPU/iGPU | batchable matmul → amortizes dispatch | 🔬 candidate |
| **[audio/](audio/)** | Wake-word / real-time denoise | the NPU's classic edge job | 🔬 candidate (untested) |
| **[ggml-backend/](ggml-backend/)** | XDNA backend for ggml/llama.cpp | foundational enabler | 🚧 planned (biggest lift) |

## Featured: camera effects (working today)

Live webcam → NPU effect → virtual camera, **measured on a Ryzen 7 8845HS (XDNA1)**:

- Full per-frame conv/colorspace pipeline: **220 FPS @1080p / 451 FPS @720p** (incl. host DMA round-trip).
- In the live daemon the NPU effect adds **~2 ms/frame at ~6.6 W** — end-to-end is **camera-capture-bound**, never NPU-bound. So there's large headroom for heavier effects (segmentation).
- Outputs to a **v4l2loopback** device — any app sees `NPU Camera`.

```bash
# (after the open-xdna NPU stack is up + this kit's requirements installed)
sudo modprobe v4l2loopback video_nr=10 card_label="NPU Camera" exclusive_caps=1
python3 camera/npu_camera_fps.py                                  # FPS feasibility gate
python3 camera/npu_camera_daemon.py --loopback /dev/video10 --stream-frames 0   # live → virtual cam
# then pick "NPU Camera" in Zoom/OBS/your browser
```

## Requirements

A working **XDNA NPU stack** (XRT + amd/xdna-driver + IRON/mlir-aie + Peano) — see
[open-xdna](https://github.com/Scottcjn/open-xdna) for the bring-up. Plus host deps:
`pip install -r requirements.txt` (opencv-python-headless, pyvirtualcam, numpy, ml_dtypes) and
`v4l2loopback` for the virtual camera.

## Honesty / scope

Verified on **XDNA1 (Phoenix)**, Ubuntu 25.10 / kernel 6.17. The **camera module is real and
measured**; the other modules are marked candidate/planned and make **no working claims** until built.
Research/community project, not a production stack.

## License

AGPLv3 + commercial dual-license — see [LICENSE](LICENSE) and [COMMERCIAL.md](COMMERCIAL.md).
© 2026 Scott Boudreaux / Elyan Labs.
