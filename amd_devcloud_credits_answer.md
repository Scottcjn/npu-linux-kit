How do you plan to use your AMD GPUs on the AMD Developer Cloud?

I maintain open-source AMD-stack projects: **open-xdna** (the first-gen XDNA1 / Phoenix NPU brought
up on Linux end-to-end with a 100% open stack — XRT + amd/xdna-driver + IRON/mlir-aie + Peano) and
**npu-linux-kit** (practical NPU apps, including a working webcam-effects daemon with real-time
background blur). From that work I've filed upstream contributions to amd/xdna-driver (issue #1447,
docs PR #1448, code PR #1449) and a Xilinx/mlir-aie discussion. I'd use Instinct GPUs on the
Developer Cloud for the **datacenter→client pipeline** the full AMD stack is built for:

1. **Train & quantize models for NPU deployment.** Fine-tune and INT8-quantize lightweight
   vision/segmentation models (MODNet-class portrait matting, and the conv stages for my NPU camera
   kit) on MI300X, then deploy them to the Ryzen AI NPU via IRON/mlir-aie — the "train on Instinct,
   infer on XDNA" loop. My kit currently runs segmentation on the CPU; this is how I move it onto AMD
   silicon end-to-end.

2. **Settle a live, community-reviewed open question: ROCm vs Vulkan GEMM.** I measured that on a
   Ryzen 8845HS, CPU prompt-prefill beats the Radeon 780M iGPU ~2.6x on llama.cpp's Vulkan/RADV
   backend (while decode favors the iGPU). A community reviewer rightly questioned whether that's a
   hardware result or an immature open Vulkan GEMM path; my on-silicon re-test showed it's robust to
   ubatch/length tuning, which points at the Mesa/RADV Q4_K GEMM kernels rather than the hardware.
   The clean way to settle it is to run the **identical workload on a mature ROCm/hipBLAS GEMM path**
   and profile achieved FLOPS/occupancy with rocprof — isolating software-stack vs hardware. Instinct
   GPUs on the Dev Cloud let me do exactly that, then port my hand-authored matmul/conv/prune kernels
   to ROCm and publish honest heterogeneous-partitioning guidance for Ryzen AI builders.

3. **LLM inference & quantization research.** Benchmark and quantize LLMs (Gemma/Qwen) on MI300X to
   feed the integrated-NPU deployment work, and validate my non-bijunctive prune/collapse experiments
   at scale.

All outputs are open-source, continuing the public track record above (github.com/Scottcjn).
