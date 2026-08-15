# 概述
本文档概述SGLang flashinfer_allreduce_fusion的触发位置与具体操作；
# Code walkthrough
## 参数解析
在server_args中，核心定义如下的两个变量，来实现TP Allreduce的fusion，其中，enable_flashinfer_allreduce_fusion已经被deprecate，本质上是设置flashinfer_allreduce_fusion_backend为auto；
![[Pasted image 20260812173133.png]]
随后，在`_flashinfer_allreduce_fusion_auto_enable(..)`函数中，给特定的白名单case直接赋值flashinfer_allreduce_fusion_auto_enable，其中，满足如下条件时，可以自动使能AR_tp_fusion，如
1）Hopper或blackwell架构；2）tp size大于1且没使能dp attention；3）节点数为1或者为Blackwell架构；4）moe_a2a_backend为None (没开EP)；同时，在`_enforce_disable_allreduce_fusion(...)`函数中，确认如果开启enforce_disable_flashinfer_allreduce_fusion，那就强制关闭AR fusion；其中，满足自动使能AR fusion的模型架构如下：
![[Pasted image 20260812200838.png]]
## 初始化
BaseRunner.warmup中，调用`_pre_initialize_flashinfer_allreduce_workspace(...)`进行AR fusion相关的workspace的初始化，函数具体如下，其中，FUSE_ALLREDUCE_MAX_BATCH_SIZE被设置为2048，随后调用pre_initialize_workspaces(...)函数进行初始化，该函数必须在cuda graph capture前完成，来保证broadcast/barriers在capture context外；
![[Pasted image 20260812201528.png]]

在`pre_initialize_workspaces(...)`函数中，则核心针对MoE和Attention workspace进行AR workspace的初始化，其中调用`ensure_workspace_initialized(...)`；在该函数中，核心做如下的事情：
* 如果是attention workspace，则更新world_size/rank/cordinator，其中coordinator就是具体的tp group或者ep group；
* 获取对应通信组的device group和cpu group，根据是否use_attn_tp_group调用`_get_workspace_manager(...)`，核心获取对应name的FlashInferWorkspaceManager，并返回；
* 调用对应的FlashinferWorkspaceManager.initialize(...)进行初始化；
  * 调用`is_buffer_size_sufficient(...)`"回答已经分配好的这块 flashinfer workspace，能不能装下当前这个问题规模的 AR，还是必须销毁重建"，workspace 是启动时按 (max_token_num=2048, hidden_dim, dtype) 一次性建好的对称显存，封装check_kw，并调用已经初始化的self.workspace.is_buffer_size_sufficient(...)来判断是否workspace可以被复用；
  * 如果world_size相同，那说明buffer太小了，直接调用cleanup(...)进行旧workspace的清理，也即调用workspace.destory(...)；
  * 调用_preflight_check_workspace_memory(...)检查是否有足够的空间来分配给workspace，如果无，则直接设置_flashinfer_allreduce_unavailable为True；
  * 更新comm_backend，用于后续的跨rank交互显存的barrier等，优先设置为_TorchDistBackend，fallback为_mnnvl_comm_backend (老版本Flashinfer实现)；
  * 更新设置获取check_kw，随后调用_create_allreduce_fusion_workspace(...)创建对应的workspace，该方法直接来自flashinfer.comm；
* 随手，调用一次`_sync_allreduce_unavailable_across_tp(...)`在所有的TP rank上同步_flashinfer_allreduce_unavailable这个flag，只要有一个rank为True，则设置所有rank的该value都为True；
workspace = 一组预先跨 rank rendezvous 好的symmetric memory缓冲区 + 描述它们的指针表/状态字——贵的、必须提前做的部分（分配、handle 交换、multicast 绑定、lamport 初始化）全部固化在这个对象里，每次 AR kernel 只是拿着指针表直接用，至此，初始化完成；
![[Pasted image 20260812213205.png]]
# forward
以Qwen3_moe架构为例，在qwen3_moe.py中，直接关注Qwen3MoEDecoderLayer，其中，核心利用LayerCommunicator的抽象，具体看forward函数，其中，只需要显式的执行self_attn(...)和mlp(...)计算，其中，别的所有的操作，如计算与计算间的通信、相邻的Norm等操作全部交给LayerCommunicator做，而不用在模型的forward里实现重复的逻辑；
```py title="Qwen3MoEDecoderLayer.Forward"
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                **kwargs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            hidden_states = self.mlp(hidden_states, forward_batch)

        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual
```
在foward函数中，可以简单理解为，完整的一次单layer forward分为：
* layer_communicator.prepare_attn_and_capture_last_layer_outputs(...)：计算Residual+RMSNorm；
* self_attn：包含attention计算和o_proj；
* layer_communicator.prepare_mlp(...)：AllReduce+Residual+RMSNorm
* 判断layer_communicator.should_fuse_mlp_allreduce_with_next_layer(...)，是否需要在MLP处fuse Allreduce，如果True，那么延迟ALLreduce到下一个prepare_attn_and_capture_last_layer_outputs中进行fused kernel；
* 调用layer_communicator.should_use_reduce_scatter判断是否可以使用Reduce Scatter而非Allreduce，好处是RMSNorm只用计算自己那部分的切片，随后调用mlp(...)计算；
* 如果使能fuse_mlp_allreduce，赋值hidden_state.\_sglang_needs_allreduce_fusion，不然正常调用layer_communicator.postprocess_layer(...)做Allreduce；
```py title="Flashinfer_allreduce_fused"
════════════════════ 第 N 层 ════════════════════════════════════════════

  hidden_states, residual
        │
        ▼
  ┌─ prepare_attn ──────────────────────────────────────────────┐
  │  ★fusion点A（还上一层的债，见下面第 N+1 层的同款位置）        │
  │  input_layernorm                                             │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  self_attn(...)  ← o_proj 输出 TP-partial，未 reduce
        │
        ▼
  ┌─ prepare_mlp ───────────────────────────────────────────────┐
  │  ★fusion点B（本层内部，attn TP group）:                      │
  │    普通路径: all_reduce(h) → post_attention_layernorm(h,res) │
  │    fused路径: post_attention_layernorm                       │
  │              .forward_with_allreduce_fusion(use_attn_tp=True)│
  │              (communicator.py:1142)                          │    AR+add+norm
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  ⑦ 层尾决策: fuse = should_fuse_mlp_allreduce_with_next_layer(fb)
  │            (batch≤2048? 非最后一层? tp>1? 无DP attention? ...)
  │
  ▼
  with get_forward().scoped(fuse_mlp_allreduce=fuse):   ┐
      hidden_states = self.mlp(hidden_states, fb)       │ flag 只在
        │                                               │ with 块内生效
        │   MoE/MLP 内部深处:                            │
        │   ⑧ RowParallelLinear (linear.py:1620)        │
        │      should_skip_mlp_all_reduce() == True     │
        │      → 跳过 down_proj 后的 all-reduce!         │
        │   ⑧ EP/TP 尾部 (moe/utils.py:582)             │
        │      should_skip_post_experts_all_reduce()    │
        │      → 跳过 experts 后的 EP AR / moe-TP AR!    │
  ◄─────┘                                               ┘
        │
        │   此刻 hidden_states 是 TP-PARTIAL 的（各 rank 只有部分和！）
        ▼
  if fuse:
      ⑨ hidden_states._sglang_needs_allreduce_fusion = True   ← 把"
      （跳过 postprocess_layer, 直接 return）                     写在张量上带走
  else:
      postprocess_layer(...)  ← 正常在这里做 AR

        │
        │   partial hidden_states + 欠条，流向下一层
        ▼
════════════════════ 第 N+1 层 ═════════════════════════════════════

  ┌─ prepare_attn (communicator.py:578-612) ────────────────────┐
  │  ★fusion点A（还第 N 层的债，MoE group）:                     │
  │                                                              │
  │  if hidden_states._sglang_needs_allreduce_fusion:  ← ⑨的欠条 │
  │      if ⑬ apply_flashinfer_allreduce_fusion(bs) 通过:        │
  │         ⑩ input_layernorm.forward_with_allreduce_fusion(     │
  │               h, res, use_attn_tp_group=False)   ← 一个kernel │
  │            └→ ⑫ layernorm.py:189                             │
  │                → flashinfer_allreduce_residual_rmsnorm       │
  │                → flashinfer.comm.allreduce_fusion(           │
  │                     pattern=kARResidualRMSNorm)              │
  │      else (守门没过, 或 flashinfer 失败返回 None):            │
  │         h = moe_tensor_model_parallel_all_reduce(h)  ← 兜底   │
  │         h, res = input_layernorm(h, res)                     │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  self_attn(...)   （后面重复第 N 层的流程）
```
