本文章概述Deepseek V4文章中和infra相关的部分，如Hybrid attention/CSA, HCA/MegaMoE等。
# 2.3 Hybrid Attention with CSA and HCA
为了应对日益激增的超长上下文场景，attention逐渐成为瓶颈 (传统transformer O(n^2)复杂度)，提出两种新的attention架构，分别是Compressed Sparse Attention (CSA)和Heavily Compressed Attention (HCA)；
* CSA：把m个token的KV压缩成一个entry，随后，应用Deepseek Sparse Attention (DSA)；
* HCA，把m' (远大于m)个token的KV压缩成一个entry，随后，使用dense attention；
