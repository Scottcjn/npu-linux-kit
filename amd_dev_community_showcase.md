TITLE (one line):
npu-linux-kit: practical AMD XDNA NPU apps for Linux — a working NPU webcam-effects daemon (220 FPS pipeline), plus an honest roadmap

CATEGORY: Showcase  (also fits Projects & Demos)

----- BODY -----

**TL;DR.** A kit of practical Linux applications for the AMD XDNA NPU — built on my open-source
gen-1 (XDNA1 / Phoenix) bring-up. The first module is a **working real-time NPU webcam-effects
daemon** that outputs a v4l2 virtual camera (so Zoom/OBS/browsers use it). The other modules
(background-blur, embeddings, audio, a ggml backend) are on a roadmap and **clearly marked as
candidate/planned — no working claims until built.** Repo: https://github.com/Scottcjn/npu-linux-kit
(foundation: https://github.com/Scottcjn/open-xdna)

**Why.** AMD's NPU is underused on Linux — the official LLM stack targets XDNA2, and the NPU's real
strengths (low-power, conv/vision, batchable compute, offload) aren't packaged into usable Linux
apps. This kit does that, one measured module at a time.

**Working today — NPU webcam effects.** Live camera → NPU conv/colorspace effect (edge-stylize:
rgba2gray → 3×3 filter2d → threshold → blend) → **v4l2loopback virtual camera**. Measured on a
Ryzen 7 8845HS (XDNA1), Ubuntu 25.10 / kernel 6.17:

- Full per-frame pipeline incl. host DMA round-trip: **220 FPS @1080p, 451 FPS @720p** (7–32× headroom over 30fps).
- In the live daemon, the NPU effect adds **~2 ms/frame at ~6.6 W** — end-to-end is **camera-capture-bound** (~15 FPS on a USB webcam), *never* NPU-bound.

So the NPU runs the effect essentially for free, leaving the CPU/iGPU idle — exactly the low-power
offload role it's built for. There's huge headroom for heavier effects.

**Roadmap (honestly marked):**
- **Background blur** — the marquee "Studio Effects" feature; needs a segmentation model on the NPU. The headroom says feasible; not built.
- **Embeddings** — batch sentence/RAG embeddings; matmul-heavy + batchable → amortizes the NPU's dispatch overhead. Candidate.
- **Audio** — wake-word / denoise; the NPU's classic edge job, but **untested here** — listed as a candidate, not a claim.
- **ggml backend** — an XDNA backend for ggml/llama.cpp; the foundational enabler. Biggest lift.

**Honesty / scope.** Verified on XDNA1 (Phoenix). The camera module is real and measured; the rest
make no working claims. Research/community project, not a production stack. The kit builds on
open-xdna, where I also filed the bring-up issues/PRs upstream (amd/xdna-driver #1447/#1448/#1449,
plus an mlir-aie discussion on the AIE2 vector ISA).

Feedback very welcome — especially from anyone on Phoenix/Hawk Point hardware who wants NPU camera
effects on Linux. 👉 https://github.com/Scottcjn/npu-linux-kit
