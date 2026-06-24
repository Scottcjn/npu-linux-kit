#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna / npu-linux-kit :: background blur — NPU does the blur, the marquee Studio-Effects feature.
#
#   (1) COLOR blur   — the grayscale blur design is run PER CHANNEL (R/G/B), recombined → color blur.
#   (2) MULTI-PASS   — each channel is blurred N times for strong bokeh (449 FPS/pass → headroom).
#   (3) SEGMENTATION — a person/background mask composites sharp-foreground + NPU-blurred-background.
#       Mask source: MediaPipe Selfie (if installed) → ONNX via opencv-DNN (if a model is given) →
#       a center-ellipse PLACEHOLDER (honest fallback; not real segmentation). A seg model running
#       ON the NPU is the future drop-in — the blur engine is already proven on-NPU.
#
# Honesty: the BLUR runs on the NPU (its strength); SEGMENTATION currently runs on the CPU.

import os, sys, time, argparse
import numpy as np, cv2

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aie.iron as iron
from blur_pipeline import blur                      # proven NPU grayscale box-blur design


def npu_color_blur(bgr, W, H, in_t, b_t, out_t, passes=1):
    """Color blur on the NPU: blur each channel via the grayscale design, N passes, recombine."""
    out = bgr.copy()
    alpha = np.full((H, W), 255, np.uint8)
    for c in range(3):                               # B, G, R
        ch = out[:, :, c]
        for _ in range(passes):
            rgba = np.dstack([ch, ch, ch, alpha]).reshape(-1)   # all channels = this channel
            in_t.numpy()[:] = rgba.view(np.int8)
            blur(in_t, b_t, out_t, width=W, height=H)
            ch = out_t.numpy().view(np.uint8).reshape(H, W, 4)[:, :, 0]   # blurred channel
        out[:, :, c] = ch
    return out


def get_mask(bgr, W, H, seg):
    """Person mask (1=foreground). seg = ('mediapipe', obj) | ('onnx', net) | ('placeholder', None)."""
    kind, obj = seg
    if kind == "mediapipe":
        res = obj.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        m = (res.segmentation_mask > 0.5).astype(np.uint8)
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if kind == "onnx":
        blobimg = cv2.dnn.blobFromImage(bgr, 1/255.0, (256, 256), swapRB=True)
        obj.setInput(blobimg); out = obj.forward()
        m = (out[0, 0] > 0.5).astype(np.uint8)
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
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
        return ("onnx", cv2.dnn.readNetFromONNX(onnx_path))
    print("  [seg] no mediapipe/onnx — using CENTER-ELLIPSE PLACEHOLDER (not real segmentation)")
    return ("placeholder", None)


def background_blur(bgr, W, H, in_t, b_t, out_t, seg, passes):
    mask = get_mask(bgr, W, H, seg)                          # CPU
    blurred = npu_color_blur(bgr, W, H, in_t, b_t, out_t, passes)   # NPU
    m3 = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)[:, :, None]   # feather edges (CPU)
    return (bgr * m3 + blurred * (1 - m3)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("-W", "--width", type=int, default=1280)
    ap.add_argument("-H", "--height", type=int, default=720)
    ap.add_argument("--passes", type=int, default=4, help="blur passes per channel (strength)")
    ap.add_argument("--onnx", default=None, help="optional segmentation ONNX model for opencv-DNN")
    ap.add_argument("--loopback", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--image", default=None, help="blur a still image instead of the camera (test)")
    opts = ap.parse_args()
    W, H = opts.width, opts.height
    seg = load_seg(opts.onnx)
    ts = W * H * 4
    in_t = iron.tensor(np.zeros(ts, np.int8), dtype=np.int8, device="npu")
    b_t = iron.zeros(16 * 16, dtype=np.int32, device="npu")
    out_t = iron.zeros(ts, dtype=np.int8, device="npu")

    if opts.image:                                           # still-image test path
        img = cv2.resize(cv2.imread(opts.image), (W, H))
        background_blur(img, W, H, in_t, b_t, out_t, seg, opts.passes)  # warm
        t0 = time.perf_counter(); res = background_blur(img, W, H, in_t, b_t, out_t, seg, opts.passes)
        dt = time.perf_counter() - t0
        cv2.imwrite("./bgblur_before.png", img); cv2.imwrite("./bgblur_after.png", res)
        print(f"  background-blur still: {dt*1e3:.1f} ms ({opts.passes} passes/ch, seg={seg[0]}) -> saved bgblur_*.png")
        return

    cap = cv2.VideoCapture(opts.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, f = cap.read()
    if not ok: print("ERROR: no camera frame"); sys.exit(1)
    f = cv2.resize(f, (W, H)); background_blur(f, W, H, in_t, b_t, out_t, seg, opts.passes)  # warm
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
        res = background_blur(f, W, H, in_t, b_t, out_t, seg, opts.passes)
        if sink: sink.send(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)); sink.sleep_until_next_frame()
        n += 1
    dt = time.perf_counter() - t0; cap.release()
    if sink: sink.close()
    print(f"  background-blur live: {n} frames, {n/dt:.1f} FPS end-to-end "
          f"({opts.passes} passes/ch, seg={seg[0]}, blur on NPU)")


if __name__ == "__main__":
    main()
