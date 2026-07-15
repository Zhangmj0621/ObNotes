# Background

随着Model training范式的切换，带来了越来越多的适用于不同特定场景的通信库，如：

- 集合通信：NCCL是de facto集合通信库，实现TP/DP/PP所需的高效AllReduce/AllGather原语；
- MoE通信的Dispatch/Combine：DeepEP/pplx kernel，提供normal/low latency模式，专门支持训练+prefill/decode下的高效Expert通信；Mooncake-EP，配合sglang社区，实现Expert的partial fault tolerance机制；nccl-ep，提供ncclDispatch(…)/ncclCombine(…)接口，对上层隐去normal/LL差异；
- 高效的P2P传输需求：mooncake transfer engine/NIXL支持高效的点对点kvcache传输；TransferEngine支持高效的权重同步p2p更新；

相同：

- 通信都是机内走NVLINK/机间走RDMA，占用相同的带宽；
- 同样都有监控/稳定性的需求；

不同：

- 不同的通信库向上提供接口不同，调用差异大；
- 监控方式异构，统计指标难以聚合；

# Why VenusCL?

虽然如NCCL的defacto CCL尝试将不同的通信pattern融入其内，实现all-in-one的通信库标准，然而，该方案有如下的几种限制，不利于真正的All-in-one：

1. NCCL目前实现了NCCL-ep期望向上暴露统一的dispatch和combine接口，然而，开发能够达到deepEP级别的ep通信库需要较大的effort，且不同框架 (e.g., megatron/sglang)往往需要额外的集成适配开销与大规模测试；
2. 同时，在NCCL通信库之上进一步实现ep，使得框架越来越臃肿，不利于社区开发者们的二次开发与使用；
3. 未来随着模型训练/推理的需求变化，会进一步促进新通信范式的出现，届时异构的通信是否还能集成入NCCL是未知；

而将不同的通信范式，如AllReduce/AllGather等集合通信通信与Dispatch/combine等ep流量等等做统一的管理，又是很有必要的，因为：

1. 统一管理通信库的监控能够真正统一到全网的流量，单个通信库的profiler往往由于无法感知其余外部的通信流量，而无法真正做到异常检测 (如mycroft等工作往往根据集合通信时单个rank的完成时间来判断rank是否处于异常状况，然而，复杂的异构通信需求往往使得通信严重imbalance，单个rank通信慢往往并不是由于异常和故障，而是流量冲突)；
2. 通信库往往在内部会尽可能的做优化来实现高效的流量工程，然后在通信库间流量往往会有冲突 (pd分离场景，kvcache的传输与EP通信流量冲突)，带来many-to-one的incast，通信库的隔离性导致缺乏整网流量感知与调度，例如对于上述case，当流量出现冲突时，可以先短暂pause kvcache的传输，等更latency-sensitive的a2a流量完成后，再resume；
3. 故障域隔离，网口故障经常发生，虽然我们能够在不同的通信库内均实现fault tolerance，然而，通信库的故障恢复始终会带来一定的同步开销，最佳的方式是，在通信库意识到故障已经发生时即刻停止下发，通过对不同通信库的不同backend做统一隔离，我们能够在一个通信库故障发生时，pause其余通信库对应网口的的下发通信操作，将故障域进行压缩；

而做到这样的一个支持不同通信backend的统一通信库，有如下的几点挑战：

- 如何实现高效的接口抽象，不同的通信库差别大，通信范式不同？
- 如何尽可能non-trival的实现统一监控/故障域隔离/流量优先级调度等feature？

为此，我们提出VenusCL，……