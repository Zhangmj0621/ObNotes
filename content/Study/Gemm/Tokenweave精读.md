# Overall
Tokenweave的核心motivation如下几点，其为token-level的fused kernel：
- 普通的把AR拆成RS+AG，虽然RMSNorm可以只算自己那一部分，然后，收益抵不过劣势 (HBM traffic增大)，因而FLUX这种策略无法直接应用；
- 原先的策略将大kernel decompose成小kernel，会影响计算效率；
- 过往工作没能利用NVLS的multimem特性，reduce操作仍然由kernel来完成；
因此Tokenweave核心做了如下的三点优化，首先关注到RMSNorm实际上是fused kernel很好的切入点，将RS+RMSNorm+AG fused成一个kernel，既降低了RMSNorm的重复计算开销，又没有增加HBM traffic；
其中，核心利用NVLS的multimem接口如multimem_ld_reduce_add和multimem_st primitive，具体的fused kernel流程如下：
- 首先计算该block/SM对应需要计算的token数量和对应的vec_hidden_size，调用sync_remote_blocks(...)，确保各个TP rank的o_proj已经计算完成；
- block针对自己对应的token，开始for循环遍历，在循环中，获取该block此次计算RMSNorm对应的token_id，将variance存放在register中，将s_variance (最终结果)放在SMEM中，随后，获取该token此次对应的offset，利用multimem_ld_reduce_add(...)操作，对多个TP rank的同一块symm memory的相同地址进行load和reduce操作，记作temp，将该值加入residual残差，对该值计算第一次方差，放入register中作为下一层的残差输入；
- 随后调用rsqrtf进行标准方差规约，存入SMEM (s_variance)中，随后，计算norm得到对应的value，调用multimem_st将结果写回到所有tp rank的symm memory里，随后调用一次sync_remote_blocks(...)通知下游GEMM上游的对应RMSNorm已经完成；
基于上，Tokenweave成功实现了RMSNorm的fused kernel，为了实现有效的计算和通信的overlap，Tokenweave将batch split成two batch，采用pingpong的方式，利用qkv_proj+attention+o_proj和RMS-fused kernel overlap的方式来进行，其中，采用了wave-aware的方式来避免拆分成多个小kernel以后，计算wave数变多的问题
![[Pasted image 20260810152124.png]]
同时，由于采用了NVLS来实现reduce操作，因而，该RMS fused kernel不需要占用大量的SM而影响attention计算，本paper论证fused RMSNorm kernel只需要2-8个SM就可以实现最佳性能，因此能够降低通信和计算的SM竞争；
![[Pasted image 20260810152131.png]]
# 可能的问题和优化点
- 核心优化采用two-batch overlap的方式，即使采用smart-split保证了wave总数不变，仍然会影响计算kernel效率
- 很难完全配平attention_block和fused-RMSNorm kernel，引入的额外同步仍然会导致bubble；
- 目前采用粗粒度的overlap方式，attention block和fused RMS kernel、fused RMS kernel和MLP block，很难保证attention block/MLP block能够较好的overlap住fused kernel，在DSA/MoE下将更加难以预测；