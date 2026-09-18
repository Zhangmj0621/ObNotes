# 概述
本文档详解sglang k3的attn_res fused kernel的实现，具体的代码在python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh中；
# Code walkthrough
由于k3的attn_res采用了复杂的num_valid_blocks机制，因而，该attn_res实则也是一个复杂的persistent kernel，其中，同时把rmsnorm给fuse到了其中；
