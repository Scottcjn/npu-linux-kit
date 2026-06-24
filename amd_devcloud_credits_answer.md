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

2. **Validate kernels and my heterogeneous-inference findings on ROCm.** I've measured a clear
   device-specialization pattern on a Ryzen 8845HS (decode→iGPU, prefill→CPU, prune→NPU). The Dev
   Cloud lets me port my hand-authored matmul/conv/prune kernels to ROCm and benchmark how that
   picture changes on datacenter GPUs — useful guidance for anyone building heterogeneous Ryzen AI
   pipelines.

3. **LLM inference & quantization research.** Benchmark and quantize LLMs (Gemma/Qwen) on MI300X to
   feed the integrated-NPU deployment work, and validate my non-bijunctive prune/collapse experiments
   at scale.

All outputs are open-source, continuing the public track record above (github.com/Scottcjn).
