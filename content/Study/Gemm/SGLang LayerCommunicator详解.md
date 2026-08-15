# Code walkthrough
## prepare_attn_and_capture_last_layer_outputs
该函数负责attention计算前的RMS、REsidual等操作，在其中，核心调用`prepare_attn(...)`方法；
在prepare_attn(...)函数中，核心做如下的几个操作：
* 如果开启--enable-attn-tp-input-scattered，则上一层不做AR，改到这一层做RS+AG，数据量小；
* 判断是否hidden_states.\_sglang_need_allreduce_fusion为True，如果为True，需要补做AR操作，调用input_layernorm.forward_with_allreduce_fusion，其中核心调用`flashinfer_allreduce_residual_rmsnorm(...)` kernel函数
* 调用`_communicate_simple_fn(...)`进行布局转换，将hidden_states从 layer_scatter_modes.layer_input_mode 转到 attn_mode；
* 如果qkv_latent_func非None (MLA架构)，把懒句柄塞进per-forward context，在 input_scattered 下补上那次推迟的 all-gather，使得AG的对象从hidden_size变成了q\_lora\_rank；
## prepare_mlp
核心调用_communicate_with_all_reduce_and_layer_norm_fn(...)函数，其中对于dense模型，调用_gather_hidden_states_and_residual(...)函数，其中，如果不支持allreduce_Fusion，则调用Allreduce+layernorm，如果支持，则直接调用layernorm.forward_with_allreduce_fusion(...)，其中，通过设置use_attn_tp_group=True来与上面prepare_attn的调用进行区分；
## should_fuse_mlp_allreduce_with_next_layer
判断是否允许MLP/MoE后的AR被fuse，首先如果attention_cp_size大于moe_dp_size，则不允许做fusion，因为如果做fusion会跳过postprocess，而其中包含针对CP的scatter，如果这里允许fusion下一层计算shape会对不上；随后，如果开启dp attention起打开eagle投机，那么也直接返回False，如果开启--enable-attn-tp-input-scattered，同样也禁止fusion；
随后记录具体的batchsize大小，然后调用apply_flashinfer_allreduce_fusion(...)判断是否支持fusion；
## should_use_reduce_scatter
如果是dsa/mla模型的cp，则允许使用reduce_scatter
## postprocess_layer
核心调用_communicate_summable_tensor_pair_fn(...)函数，