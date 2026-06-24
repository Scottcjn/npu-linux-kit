# embeddings (planned)

Batch sentence/RAG embedding generation on the NPU. **Status: candidate, not built.**
Rationale: embeddings are matmul-heavy and *batchable*, so they amortize the NPU's dispatch
overhead toward its ~64 GFLOP/s peak (measured in open-xdna) and offload the CPU/iGPU at ~6.6 W.
Needs: an embedding model's matmuls expressed via IRON, + a server interface.
