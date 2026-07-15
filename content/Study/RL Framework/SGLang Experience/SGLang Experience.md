本文为精读赵晨阳https://docs.sglang.io/advanced_features/sglang_for_rl.html的笔记，记录在RL framework中集成SGLang的一些痛点，包括rollout、evalutaion、training以及权重更新等等；

# 为什么选择SGLang

chenyang给出了五个选择sglang的理由

- 细粒度的engine的sleep和wake up；
- open-to-use：支持colocation和disaggregation两种模式；
- 支持partial rollout以及特定的rollout控制；
- 确定性推理：确保训练推理的精度对齐
- 精细的sglang router：cache_aware，load_balancing自动选择最优的rollout DP