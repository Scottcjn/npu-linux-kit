#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna / npu-linux-kit :: background blur — the marquee Studio-Effects feature.
#
#   (1) COLOR blur   — the grayscale blur design is run PER CHANNEL (R/G/B), recombined → color blur.
#   (2) MULTI-PASS   — each channel is blurred N times for strong bokeh (449 FPS/pass → headroom).
#   (3) SEGMENTATION — a person/background mask composites sharp-foreground + NPU-blurred-background.
#       Mask source: MediaPipe Selfie (if installed) → ONNX via opencv-DNN (if a model is given) →
#       a center-ellipse PLACEHOLDER (honest fallback; not real segmentation). A seg model running
#       ON the NPU is the future drop-in — the blur engine is already proven on-NPU.
#
# Honesty (v1): blur defaults to a clean CPU Gaussian; the NPU blur path (--npu-blur) is
# EXPERIMENTAL (fixed-point quality WIP). Segmentation runs on the CPU. The proven NPU effect is
# the edge-stylize in npu_camera_daemon.py; a clean on-NPU blur + on-NPU seg are the next builds.

import os, sys, time, argparse
_NPU_SCALE = 4
import numpy as np, cv2

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aie.iron as iron
from blur_plane import blur_plane                    # custom clean NPU 3x3 blur kernel (blur3x3.cc)


_plane_in = _plane_out = _plane_n = None
def _blur_planes_npu(img, w, h, passes):
    """Blur each color channel (uint8 plane) with the custom blur3x3 NPU kernel, N passes."""
    global _plane_in, _plane_out, _plane_n
    if _plane_n != w * h:
        _plane_in = iron.tensor(np.zeros(w * h, np.uint8), dtype=np.uint8, device="npu")
        _plane_out = iron.zeros(w * h, dtype=np.uint8, device="npu")
        _plane_n = w * h
    out = img.copy()
    for c in range(3):
        ch = np.ascontiguousarray(out[:, :, c])
        for _ in range(passes):
            _plane_in.numpy()[:] = ch.reshape(-1)
            blur_plane(_plane_in, _plane_out, width=w, height=h)
            ch = _plane_out.numpy().reshape(h, w).copy()
        out[:, :, c] = ch
    return out

def npu_color_blur(bgr, W, H, passes=1, scale=4):
    """Clean color blur on the NPU. Real-time path: blur at reduced resolution (scale) then upscale —
    background bokeh is low-frequency, so a downscaled blur is visually ~identical and ~scale^2 faster.
    The blur itself is bit-clean (custom blur3x3 kernel: no rgba2gray round-trip, no speckle)."""
    if scale > 1:
        w2 = max(64, ((W // scale) // 64) * 64)   # width must be a multiple of 64 (AIE granularity)
        h2 = max(2, ((H // scale) // 2) * 2)       # even height
        small = cv2.resize(bgr, (w2, h2), interpolation=cv2.INTER_AREA)
        sb = _blur_planes_npu(small, w2, h2, passes)
        return cv2.resize(sb, (W, H), interpolation=cv2.INTER_LINEAR)
    return _blur_planes_npu(bgr, W, H, passes)


_mask_cache = {"m": None, "i": -1, "every": 3}
def get_mask(bgr, W, H, seg):
    """Person mask (1=foreground). Recomputed every Nth frame and reused between (people move slowly)."""
    _mask_cache["i"] += 1
    if _mask_cache["m"] is not None and seg[0] != "placeholder" and (_mask_cache["i"] % _mask_cache["every"]):
        return _mask_cache["m"]
    kind, obj = seg
    if kind == "mediapipe":
        res = obj.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        m = (res.segmentation_mask > 0.5).astype(np.uint8)
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if kind == "onnx":                               # onnxruntime session (MODNet-style matte)
        inp = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32)
        inp = ((inp / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]   # [-1,1], NCHW RGB
        out = obj.run(None, {obj.get_inputs()[0].name: inp})[0]
        m = (cv2.resize(out[0, 0], (W, H)) > 0.5).astype(np.uint8); _mask_cache["m"] = m; return m
    # placeholder: center ellipse (NOT real segmentation — demo stub)
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (W // 2, H // 2), (W // 5, int(H * 0.45)), 0, 0, 360, 1, -1)
    return m


def load_seg(onnx_path=None):
    try:
        import mediapipe as mp
        return ("mediapipe", mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1))
    except Exception:
        pass
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            return ("onnx", ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]))
        except Exception as e:
            print(f"  [seg] onnxruntime load failed: {e}")
    print("  [seg] no mediapipe/onnx — using CENTER-ELLIPSE PLACEHOLDER (not real segmentation)")
    return ("placeholder", None)


def background_blur(bgr, W, H, in_t, b_t, out_t, seg, passes, npu_blur=False):
    mask = get_mask(bgr, W, H, seg)                          # CPU segmentation
    if npu_blur:
        # EXPERIMENTAL: NPU color multi-pass blur. Known quality issues (fixed-point: the
        # filter2d int16->int8 coeff truncation + multi-pass gain compounding → artifacts /
        # saturation). Needs a purpose-built unity-gain NPU blur kernel. Not production-ready.
        blurred = npu_color_blur(bgr, W, H, passes, _NPU_SCALE)
    else:
        # Default: strong CPU Gaussian (clean bokeh). The NPU's proven role here is the edge
        # effect; a clean NPU blur is WIP (see npu_blur above).
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=8 + 2 * passes)
    m3 = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)[:, :, None]   # feather mask edges
    return (bgr * m3 + blurred * (1 - m3)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("-W", "--width", type=int, default=1280)
    ap.add_argument("-H", "--height", type=int, default=720)
    ap.add_argument("--passes", type=int, default=4, help="blur strength")
    ap.add_argument("--npu-blur", action="store_true", help="blur on the NPU (custom kernel); default = CPU Gaussian")
    ap.add_argument("--npu-scale", type=int, default=4, help="NPU blur downscale factor (4 = validated/known-good; >4 can ERT-timeout, AIE dim-sensitive)")
    ap.add_argument("--onnx", default=None, help="optional segmentation ONNX model for opencv-DNN")
    ap.add_argument("--loopback", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--image", default=None, help="blur a still image instead of the camera (test)")
    opts = ap.parse_args()
    W, H = opts.width, opts.height
    global _NPU_SCALE; _NPU_SCALE = opts.npu_scale
    seg = load_seg(opts.onnx)
    ts = W * H * 4
    in_t = iron.tensor(np.zeros(ts, np.int8), dtype=np.int8, device="npu")
    b_t = iron.zeros(16 * 16, dtype=np.int32, device="npu")
    out_t = iron.zeros(ts, dtype=np.int8, device="npu")

    if opts.image:                                           # still-image test path
        img = cv2.resize(cv2.imread(opts.image), (W, H))
        background_blur(img, W, H, in_t, b_t, out_t, seg, opts.passes, opts.npu_blur)  # warm
        t0 = time.perf_counter(); res = background_blur(img, W, H, in_t, b_t, out_t, seg, opts.passes, opts.npu_blur)
        dt = time.perf_counter() - t0
        cv2.imwrite("./bgblur_before.png", img); cv2.imwrite("./bgblur_after.png", res)
        print(f"  background-blur still: {dt*1e3:.1f} ms ({opts.passes} passes/ch, seg={seg[0]}) -> saved bgblur_*.png")
        return

    cap = cv2.VideoCapture(opts.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, f = cap.read()
    if not ok: print("ERROR: no camera frame"); sys.exit(1)
    f = cv2.resize(f, (W, H)); background_blur(f, W, H, in_t, b_t, out_t, seg, opts.passes, opts.npu_blur)  # warm
    n = 0; t0 = time.perf_counter()
    sink = None
    if opts.loopback:
        import pyvirtualcam
        sink = pyvirtualcam.Camera(width=W, height=H, fps=30, device=opts.loopback,
                                   fmt=pyvirtualcam.PixelFormat.RGB)
    while n < opts.frames or (opts.loopback and opts.frames == 0):
        ok, f = cap.read()
        if not ok: break
        if f.shape[1] != W or f.shape[0] != H: f = cv2.resize(f, (W, H))
        res = background_blur(f, W, H, in_t, b_t, out_t, seg, opts.passes, opts.npu_blur)
        if sink: sink.send(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)); sink.sleep_until_next_frame()
        n += 1
    dt = time.perf_counter() - t0; cap.release()
    if sink: sink.close()
    print(f"  background-blur live: {n} frames, {n/dt:.1f} FPS end-to-end "
          f"({opts.passes} strength, seg={seg[0]}, blur={'NPU-exp' if opts.npu_blur else 'CPU'})")


if __name__ == "__main__":
    main()
