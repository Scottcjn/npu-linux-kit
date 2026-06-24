# camera — NPU webcam effects for Linux

Real-time webcam effects on the AMD XDNA NPU, output as a **v4l2 virtual camera** so any app
(Zoom, OBS, browsers) can use it. This is an open-source take on the "Studio Effects" idea that
Windows ships on the NPU and Linux lacks.

## Status

- ✅ **edge-stylize** (`--effect edge`) — rgba2gray → 3×3 Laplacian → threshold → blend.
- ✅ **blur** (`--effect blur`, `blur_pipeline.py`) — NPU 3×3 box-blur conv, **449 FPS @720p**, verified
  (row variance 16256→255 on a 1px test pattern; `test_blur.py`). v1 is grayscale + a single 3×3 pass (mild blur).
- ✅ **background blur** (`npu_background_blur.py`) — clean, working end-to-end:
  - **Segmentation = real**: MODNet via onnxruntime (`--onnx modnet.onnx`) → accurate person cutout
    (~136 ms/frame CPU). Center-ellipse placeholder only if no model is given.
  - **Blur = clean, on the NPU**: a **custom AIE kernel** (`kernels/blur3x3.cc`, bound by `blur_plane.py`)
    gives a correct, artifact-free unity-gain Gaussian — int32 acc + int16 coeffs + `>>12` + `[0,255]`
    clamp, where the stock `filter2d` saturated bright pixels. Enable with `--npu-blur`.
  - **Composite**: sharp foreground + blurred background, feathered edges.
  - **Perf caveat (honest):** the custom kernel is *scalar*, so the full color multi-pass NPU blur is
    ~2.4 FPS (correct, not real-time). **Real-time needs a vectorized kernel** (fixing the vector-path
    int8/saturation). The default blur path is the clean CPU Gaussian for real-time use.
  - Get MODNet: `curl -L -o modnet.onnx https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx`
  - Still test: `python3 npu_background_blur.py --image scene.png --onnx modnet.onnx [--npu-blur]`

## Measured (Ryzen 7 8845HS, XDNA1, Ubuntu 25.10 / kernel 6.17)

`npu_camera_fps.py` — full per-frame pipeline incl. host DMA round-trip:

| resolution | ms/frame | FPS | headroom @30 |
|---|---|---|---|
| 640×480 | 1.02 | **985** | 32× |
| 1280×720 | 2.21 | **451** | 15× |
| 1920×1080 | 4.54 | **220** | 7× |

`npu_camera_daemon.py` — live `/dev/video0` → NPU effect → `/dev/video10` virtual cam:
the NPU effect adds **~2 ms/frame (~6.6 W)**; end-to-end is **camera-capture-bound** (~15 FPS on a
typical USB webcam), **never** NPU-bound. The NPU is essentially free here.

## Usage

```bash
# 1) create the virtual camera (once per boot)
sudo modprobe v4l2loopback video_nr=10 card_label="NPU Camera (open-xdna)" exclusive_caps=1

# 2) feasibility gate (optional)
python3 npu_camera_fps.py

# 3) run the daemon: live camera -> NPU effect -> virtual cam
python3 npu_camera_daemon.py --loopback /dev/video10 --stream-frames 0   # 0 = run until Ctrl-C

# 4) in Zoom/OBS/your browser, select camera "NPU Camera (open-xdna)"
```

Notes:
- Needs the XDNA NPU stack up (see [open-xdna](https://github.com/Scottcjn/open-xdna)); run under the
  IRON venv. `/dev/accel` access typically needs `sudo` or the `render` group.
- Set `MLIR_AIE_DIR` if your mlir-aie checkout isn't at `~/open-xdna/mlir-aie` (the daemon imports the
  `edge_detect` IRON pipeline from there).
- Without `--loopback`, the daemon runs a capture-vs-NPU **benchmark** and saves a before/after sample.
