# 概述
本文档详细walkthrough fa4中在SM100/SM90下实现的forward code (serving暂不考虑backward)。
# Walkthrough
## flash_fwd_sm100.py
flashattention-4采用了warp specialization的技术，具体如 ![[FlashAttention4 精读]]中所讲，采用ping-pong的方式来进行MMA和softmax的overlap，其中，共存在两个softmax warpgroup，每个softmax warpgroup包含128个threads也即4个warp，每个thread负责一行softmax计算，因此前0-7号共8个warp作为两个softmax_warp_group，同时，由于TMEM的存在，rescale不再存在关键路径上，P算完以后不用一直放在寄存器内等待rescale操作完成后再累加到O上，而是可以直接把P丢到TMEM以后，立刻释放register压力，因而，维护correction_warpgroup来专门负责rescale操作，同样的，每一行由一个thread来负责，由于rescale操作并不那么频繁，因而不用额外维护两个warpgroup；
```py title=""
self.softmax0_warp_ids = (0, 1, 2, 3)
self.softmax1_warp_ids = (4, 5, 6, 7)
self.correction_warp_ids = (8, 9, 10, 11)
self.mma_warp_id = 12
self.epilogue_warp_ids = (13,)
self.load_warp_ids = (14,)
self.empty_warp_ids = (15,)
self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100") # 512 coloum
```
### \_\_init\_\_详解
首先详细拆解FlashAttentionForwardSm100的__init__(...)中的输入参数以及具体的设置，具体的输入变量如下：
```py title=""
head_dim: int, # q/k dimension
head_dim_v: Optional[int] = None, # v dimension, needed for MLA
qhead_per_kvhead: cutlass.Constexpr[int] = 1, # GQA
is_causal: bool = False, # causal mask, needed for prefill
is_local: bool = False, # sliding window, local attention
is_split_kv: bool = False, # flash decoding，把超长的kv切成多端，每个CTA算一部分，随后靠着scatter聚合到一起
pack_gqa: bool = False, # 对于decode和短prefill，没法切成tile，此时支持把同一个KV head对应的q head聚合到一起算
q_subtile_factor: int = 1, # 块稀疏 mask 的粒度细分因子
kv_subtile_factor: int = 1, # 块稀疏 mask 的粒度细分因子
m_block_size: int = 128, # tile M
n_block_size: int = 128, # tile N
q_stage: cutlass.Constexpr[int] = 2, # 2代表两个tile ping-pong做overlap
is_persistent: bool = True, # persistent kernel，SM直接按照偏移去拿tile进行计算
score_mod: cutlass.Constexpr | None = None, # 用户自定义分数修改回调
mask_mod: cutlass.Constexpr | None = None, # flex-attention 风格的用户自定义 mask 回调
has_aux_tensors: cutlass.Constexpr = False, # score_mod/mask_mod 是否携带辅助张量(如 ALiBi 的 slopes)。影响 score_vec_size
paged_kv_non_tma: bool = False, # 是否支持用TMA来拷贝KV数据
is_varlen_q: bool = False, # 是否是varlen Q，needed for prefill
use_2cta_instrs: bool = False, # 使用2-CTA MMA单元
use_clc_scheduler: bool = False,
```
在__init__函数中，会把head_dim和head_dim_v都padding到16的倍数，其中，核心通过调用`self.arch = BaseDSL._get_dsl().get_arch_enum()`来获取具体的硬件结构，随后获取具体三个tiler的mma的形状，分别如下，其中对于cta的单个tile而言，由于分为2个stage，因此输出是2MxN，最后一维是d，对于mma_tiler_qk而言，结果是MxN，中间维度是d，对于pv的mma，输出是Mxd，中间维度是N；
```py title=""
self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
# With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
# Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)
self.mma_tiler_pv = (self.cta_group_size * m_block_size, self.head_dim_v_padded, n_block_size)
```
其中，核心注意use_tma_O这个变量，代表是否允许是有TMA来搬运O矩阵，他的开启条件很严苛，如下所示，如果开启pack_gqa，那么必须M可以被qhead_per_kvhead整除，这也是为了方便TMA的索引，同时，如果开启MQA，那么必须不能使用split_kv，同时，不能使用is_varlen_q，因为Q每条序列边界任意,TMA box 罩不住不许越界写的行边界)。放弃后走 gmem_tiled_copy_O:先 smem→寄存器→再逐元素带谓词写 gmem(\_store_O_to_gmem, :2863,每行判断 < seqlen_q 才写)。其中，如果能用TMA把O从shared memory搬回global memory，那么只用epilogue_warp_ids也即第13号warp也就够了，不然，则直接使用correction的四个warp来负责搬运，并把原先的13号warp放入empty_warp_ids中。
```py title=""
self.use_tma_O = (
    not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)
    and not (self.pack_gqa and self.is_split_kv)
    and not is_varlen_q
)
self.use_correction_warps_for_epi = not self.use_tma_O
# ... ...
if self.use_correction_warps_for_epi:
    self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
    self.epilogue_warp_ids = self.correction_warp_ids
```
随后，下面值得注意的变量就是enable_ex2_emu，即利用多项式估计来算softmax，释放MUFU压力；同时，其中存在overlap_sO_sQ，仅在head_dim为192时打开，可以算笔账，hdim 192 时 sQ 要 2×128×192×2B = 96KB,K/V 每 stage 48KB,再单独留一块 sO(2×128×128×2B=64KB)的话流水级数就没了。而 Q 和 O 的生命周期其实错开:Q 在最后一次 QK GEMM 后就没用了,O 在最后才写 smem——所以让 sO 直接骑在 sQ 的地址上(:1039-1042 的 recast_ptr),smem_size_q_o 取 max 而不是 sum，同时，由于persistent kernel 会提前用TMA把下一个Q給load进来，可能和O冲突，因此强制关闭persistent kernel。注意，即使dim是128，如果开split KV，同样也需要overlap，因为

