1. 概述
本文详细分析在sglang中，decode开大EP下，sp_collective对应的fused kernel实现，其中，本文档包含从框架层的完整调用链code walkthrough和具体fused kernel的实现；
2. Code walkthrough
2.1 Framework walkthrough
在kimi-k3中，对于pd分离部署的decode端，假设Decode以大EP模式部署，且attention侧仍然以TP+DP形式切开，则在该并行策略做forward时，仍然会存在TP相关的AR通信，为此，sglang针对kimi-k3，分别提供了SGLANG_K3_SP_COLLECTIVE与SGLANG_K3_SP_ATTN_RES两种Ar fusion策略相关的变量；
其中，在KimiK3DecoderLayer的初始化中中，首先会根据a2a_backend里确认是否使能sp，即，同时存在Attention侧的TP与Expert侧的EP，其中，一旦使能sp，则会默认关闭all_reduce_fusion，这也使得在ep+tp设置下，一旦kernel fusion无法顺利使能， 会直接fallback到NCCL而非传统的ar fusion；开启sp后，会把TP的allreduce拆成ReduceScatter和AllGather，其中ReduceScatter完后，做完对应的prefix_add, attn_residual和rmsnorm后，并不做对应的allgather，此时的token已经天然在不同的tp rank间分开，各自进入对应的mlp层进行forward；
self._sp_moe = (
    (
        _a2a_backend.is_megamoe()
        or _a2a_backend.is_deepep()
        or _a2a_backend.is_mooncake()
        or _a2a_backend.is_ascend_fuseep()
        or _a2a_backend.is_mori()
    )
    and self._is_moe_layer
    and get_parallel().attn_tp_group.world_size > 1
)
此外，如果使能k3_sp_collective.enabled()，即开启环境变量SGLANG_K3_SP_COLLECTIVE且通过一系列参数检查，则会修改o_proj的output，让他直接输出到预设的symmetric memory中，这样就能后续直接走custom allreduce v2的RS与AG通信；
在layer的forward中，核心调用_forward_attn_residual(...)函数，其中，首先会判断是否input_sharded，即上一层的输出是否是sp后的，缺少一次defer allgather，若是，则调用attn_res.forward_sp_all_gather(...)调用一次fuse allgather操作，其中，要求设置SGLANG_K3_SP_ATTN_RES，如果不是，则直接返回None，分别执行attn_res+Norm的fused kernel和Allgather kernel；
    def _forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        attn_res: AttnResidual,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        input_sharded: bool,
        keep_sharded: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        if input_sharded:
            assert self._sp_moe
            input_rows = _sp_local_rows(hidden_states)
            fused_ag = attn_res.forward_sp_all_gather(
                hidden_states,
                prefix_sum,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                rows=input_rows,
                write=self.is_block_write_layer,
            )
            if fused_ag is not None:
                hidden_states, prefix_sum = fused_ag
            else:
                hidden_states, prefix_sum = attn_res.forward(
                    hidden_states,
                    prefix_sum,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    rows=input_rows,
                    write=self.is_block_write_layer,
                )
                # Aggregate/norm and snapshot only this rank's rows, then
                # gather the normalized tensor consumed by attention.
                hidden_states = _sp_all_gather_rows(hidden_states)
随后，正常调用_run_self_attn(...)进行attention+o_proj层的计算，其中由于前面的override，能够保证o_proj的输出之间写入对应的symmetric memory中，随后，进行RS的fused kernel逻辑，首先尝试调用attn_res中的forward_sp_reduce_scatter(...) kernel，其中将RS+Prefix_add+attn_res+Norm fuse到一个kernel中，开启条件同上，需要开启SGLANG_K3_SP_ATTN_RES，若未开启，则自动fallback，分别调用RS+prefix_add和attn_res fused kernel；
[Image]
至此，一层layer的forward结束；
2.2 Kernel walkthrough
本小节详解目前的forward_sp_reduce_scatter(...)和forward_sp_all_gather(...)两个fuse kernel的实现，通过分析，能显而易见目前的fuse kernel为何还未做到极致；
注：补充一个perliminary，Kimi-k3的residual add和传统模型的residual add不太一样，他的residual add不只是会看上一层计算的prefix，而是会回看特定num_valid_blocks的特定几层的prefix来进行计算，具体数学公式如下：
- Traditional: $$
  \mathrm{residual}_n = 
  \mathrm{residual}_{n-1}
  +
  \Delta_n
 $$；
- Kimi-K3: $$\mathrm{prefix}_{n} = \mathrm{prefix}_{n-1} + \Delta_n$$， $$\mathrm{residual}_n = \sum_{i=0}^{N-1} \alpha_i \cdot \mathrm{bank}_i + \alpha_N \cdot \mathrm{prefix}_n$$，其中 $$bank_i$$代表前面的特定num_valid_blocks对应的层的 $$prefix_i$$；
因而，k3的attn_res本质是一个复杂的小attention计算，sglang社区提供attn_res+norm的fuse kernel，本质仍是persistent kernel的实现，本文档不详细展开；
以forward_sp_reduce_scatter为例，核心调用k3_sp_collective.reduce_scatter_attn_res(...)函数，随后调用attn_res_fused_pull_rs(...)函数，本质调用kernel attn_res_fused_pull_rs_kernel(...)；
kernel函数的详细代码如下，显而易见，该fuse kernel实则只是把两个device kernel放入了同一个大kernel中，并没有实现fine-grained的overlap，整体kernel实现流程如下：
- 入口处做barrier，保证o_proj的输出对所有的rank都可见，保证multimem操作正确性；
- 按照block粒度分配token，对每个token，获取其相关的prefix/residual和symmetric memory中的offset;
- 将列均匀分配给block中的所有thread，一次multimem操作单元是16B，因此总共有kDim* 2B/16B个bf16元素，均匀分配给不同的thread，每个thread首先调用ld_multimem_16B进行multimem的ld&reduce操作，获得vec，并将其与对应的residual相加，结果存入prefix中，完成之前defered的prefix_sum的累加操作；
- 调用__threadfence()和__syncthreads()确保全局同步与内存写入一致性；
- 调用高度优化的persistent kernel执行attn_res+norm的fuse kernel；
[Image]
3. 优化方案
显然，目前的fuse kernel并未做到最优，仍有细粒度优化的空间，按照之前的尝试，有两种优化的思路
- 将RS+prefix_add和后续的attn_res+norm的fuse kernel fuse成高度pipeline的fuse kernel，由于两者操作均为行粒度(token)，完全可以以token为粒度，每个token通信完成就立刻触发该token的计算，且可以考虑是否能将输入的参数提前放入SMEM中避免双倍GMEM traffic；
  - 优势：attn_res为sglang直写kernel，不会有大改后gemm性能严重劣化的问题；
  - 劣势：从nsys profling结果来看，attn_res的时间较短，仅4ms，可能无法overlap RS通信；此外，由于decode侧token数量较小，基本上wave数量只为1/2，更加缺失overlap机会，收益只能来源于降低的GMEM带宽和少一次全局同步；
- 将RS+prefix_add与前序的o_proj进行fuse，对于o_proj而言，每个tile计算完后，可以立刻发起对应的multimem通信操作，而无需等待整行完成后才开始通信，因而，RS的通信能够被掩盖在tile的wave计算中；
  - 优势：通信能够逐tile发起操作，不用等待完整RS能够较好的被overlap在计算中，也不用改动高度优化的attn_res+norm的fuse kernel；
  - 劣势：o_proj是高度优化的闭源cublas kernel，需要对tile shape/cluster shape等调参；