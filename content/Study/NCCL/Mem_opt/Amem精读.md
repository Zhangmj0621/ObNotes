# 1. Background

NCCL显存占用巨大，在强化学习场景，共卡模式，需要在rollout和actor间切换，因此在切换时往往需要卸载另一块的显存，此外，对于分离模式的权重同步的通信组，每个step只会使用一次，然而，他仍然会占据大量的显存，影响用户开较大的batchsize，因此，需要一个solution来卸载NCCL占用的显存；

# 2. Solution

Inclusion AI提出了Amem，核心基于CUDA的VMM API，设计了简介的两层解耦方案，实现对NCCL显存的透明卸载和恢复的三重功能保障；

具体而言：Code Dive
![[Pasted image 20260408133810.png]]
![[Pasted image 20260408133853.png]]
# 3. Code Dive

## 3.1. NCCL patch

amem首先提供针对NCCL2.27的小patch，本节详解amem对NCCL具体的改动