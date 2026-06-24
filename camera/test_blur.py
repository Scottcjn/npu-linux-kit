import sys, time, os
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__) or ".")
import aie.iron as iron
from blur_pipeline import blur

W, H = 1280, 720
ts = W * H * 4
# sharp checkerboard test pattern — blur must smooth it
img = np.zeros((H, W, 4), np.uint8); img[:, :, 3] = 255
mask = (np.add.outer(np.arange(H), np.arange(W))) % 2 == 0
img[mask] = [255, 255, 255, 255]

in_t = iron.tensor(img.reshape(-1).view(np.int8), dtype=np.int8, device="npu")
b_t = iron.zeros(16 * 16, dtype=np.int32, device="npu")
out_t = iron.zeros(ts, dtype=np.int8, device="npu")
blur(in_t, b_t, out_t, width=W, height=H)
blur(in_t, b_t, out_t, width=W, height=H)
t0 = time.perf_counter()
for _ in range(50):
    blur(in_t, b_t, out_t, width=W, height=H)
dt = (time.perf_counter() - t0) / 50
o = out_t.numpy().view(np.uint8).reshape(H, W, 4)

sharp_var = float(img[H // 2, :, 0].astype(np.float32).var())
blur_var = float(o[H // 2, :, 0].astype(np.float32).var())
cv2.imwrite("/home/scott/npu-linux-kit/camera/blur_before.png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
cv2.imwrite("/home/scott/npu-linux-kit/camera/blur_after.png", cv2.cvtColor(o, cv2.COLOR_RGBA2BGR))
smoothed = "SMOOTHED ok" if blur_var < sharp_var * 0.8 else "NOT smoothed?"
print(f"blur @720p: {dt*1e3:.2f} ms/frame -> {1/dt:.0f} FPS")
print(f"row variance sharp={sharp_var:.0f} -> blurred={blur_var:.0f}  ({smoothed})")
