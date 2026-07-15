# GPU概念回顾
1. TMA：Tensor memory Accelerator：一个加速Global memory与SharedMemory之间拷贝的硬件。它最大的意义是不占用寄存器和 CUDA Core 指令发射带宽。它是异步的，你只需要在指令发射后带一个 mbarrier 来检查拷贝是否完成。
2. UMMA：Blackwell架构下Tensor Core的独立执行引擎，是 Tensor Core 的硬件实体，配置好之后可以自己执行矩阵乘加Matrix Multiply-Accumulate，也就是 MMA，即 D = A × B + C。UMMA 的核心突破是支持了 Microscaling 格式（如 FP4, FP6）。
3. TMEM：Blackwell架构下Tensor Core的专属内存区域，物理上紧挨着 Tensor Core，独立于寄存器堆和 SMEM。可以存放scale factor和MMA的中间结果D等。在 Hopper 中，矩阵乘法的中间结果 D 必须存放在寄存器（Register File）里，这会导致“寄存器压力”巨大，限制了并行的 Warp 数量。
4. UTCCP：SharedMemory到TMEM之间的专属异步拷贝引擎
# 总览
核心思想：把 EP dispatch、linear 1、SwiGLU、linear 2、EP combine 融合到一个 mega-kernel 中，并让 NVLink 通信和 Tensor Core 计算重叠。
[Image]
5. 只支持Blackwell，MoE中weight为fp4，act为fp8
6. 通信暂时只支持nvlink，即单机或者超节点，dispatch使用了nvlink的read，combine使用的是write，技术报告中表明read可以避免write的flag额外延迟
7. dispatch和combine的通信量相对DeepEP变多，DeepEP中一个src rank的token[i]如果命中dst rank多个local_expert，那么从src_rank到dst_rank只会发送一次，但是MegaMoE会按照expert粒度进行传输
# Baseline实现
Mega MoE对比的baseline，分阶段执行：
8. deep_ep.ElasticBuffer.dispatch 做 EP dispatch，把本 rank 的 token 按 topk_idx 发送到对应专家所在 rank。
9. m_grouped_fp8_fp4_gemm_nt_contiguous 做第一层 grouped GEMM。
10. tilelang_ops.swiglu_apply_weight_to_fp8 做 SwiGLU、乘 top-k weight、量化为 FP8。
11. 再调用一次 grouped GEMM 做第二层。
12. ep_buffer.combine 把 top-k 专家输出 combine 回原 token
## Dispatch阶段
输入的token格式为BF16，按hidden维度每32个一组量化为FP8+UE8M0的sf。deepep的通信载荷就是Fp8+sf
## Expert矩阵权重量化
![[MegaMoE 2026-07-10 22.45.48.excalidraw.svg]]
%%[[MegaMoE 2026-07-10 22.45.48.excalidraw.md|🖋 Edit in Excalidraw]]%%w
l1_weights = torch.randn(
    (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16)
    └── num_groups ──┘     └───── n ─────┘         └─ k ─┘
将BF16的(num_groups, n, k)矩阵量化为(num_groups, n, k // 32) UE8M0的w_sf和(num_groups, n, k)的FP4格式
也就是对于token维度不会分组，对于hidden维度每32个一组共用一个UE8M0的scale
GEMM
输入为tokens：FP8+SF
模型权重：FP4+ hidden 32个一组的w_sf的 UE8M0的scale
先分配一个BF16的缓冲区，GEMM的结果为BF16
Activate
SwiGLU的时候，将BF16消费并且写成FP8+SF的形式
Combine
由于GEMM输出直接就是BF16了，不需要再转成BF8+sf的形式再进行Combine，因为在加法的过程中，FP8会损失大量的精度
Token BF16
  └─[per-token_cast_to_fp8, gran_k=32, UE8M0]──▶ FP8 + sf
Dispatch (FP8 + sf 通信)
  │
  ▼
GEMM1:  FP4_w (per-32 K, UE8M0 sf) × FP8_act  ──▶ FP32 accum
  └─ epilogue: SwiGLU ⊙ topk_weight ──▶ 量化回 FP8 + sf
  │
  ▼
GEMM2:  FP4_w × FP8_act  ──▶ BF16 直接输出
  │
  ▼
Combine (BF16 归约) ──▶ y (BF16)
DeepSeek测试数据的形状
This content is only supported in a Feishu Docs
WorkSpace
torch.distributed._symmetric_memory.empty(...)
symm_mem.rendezvous(...) # 同步
symmetric buffer 本质上是 跨 rank/GPU 都能互相寻址的一块 CUDA buffer，用于 Mega MoE fused kernel 内部做 EP dispatch/combine 和中间 workspace。
SymmBuffer 使用 torch.distributed._symmetric_memory.empty 分配一段每个 rank 地址可互相映射的 buffer，再通过 symm_mem.rendezvous 得到跨 rank buffer 指针列表。分配后立即 zero_() 并 barrier 同步，确保所有 rank 的 workspace 初始状态一致。
Symmetric Buffer 内存布局（单个 rank 视角）:
┌──────────────────────────────────────────────────────────────────────┐
│ Workspace (元数据区)                                                 │
│  ├─ barrier signals (32 bytes)                                       │
│  ├─ expert_send_count[num_experts] (uint64)                          │
│  ├─ expert_recv_count[num_experts] (uint64)                          │
│  ├─ expert_recv_count_sum[num_experts_per_rank] (uint64)             │
│  ├─ l1_arrival_count[num_max_pool_blocks] (uint32, padded to even)   │
│  ├─ l2_arrival_mask[num_max_pool_blocks] (uint64)                    │
│  ├─ src_token_topk_idx[experts_per_rank][ranks][max_recv] (uint32)   │
│  └─ TokenSrcMetadata[num_max_pool_tokens] (3 x uint32)              │
├──────────────────────────────────────────────────────────────────────┤
│ x:            FP8 E4M3  [num_max_tokens_per_rank, hidden]            │
│ x_sf:         int       [num_max_tokens_per_rank, hidden/128]  K-maj │
│ topk_idx:     int64     [num_max_tokens_per_rank, num_topk]          │
│ topk_weights: float     [num_max_tokens_per_rank, num_topk]          │
├──────────────────────────────────────────────────────────────────────┤
│ l1_acts:      FP8 E4M3  [num_max_pool_tokens, hidden]               │
│ l1_acts_sf:   int       [num_padded_sf_pool_tokens, hidden/128] M-maj│
│ l1_topk_wt:   float     [num_max_pool_tokens, 1]                    │
├──────────────────────────────────────────────────────────────────────┤
│ l2_acts:      FP8 E4M3  [num_max_pool_tokens, intermediate_hidden]   │
│ l2_acts_sf:   int       [num_padded_sf_pool_tokens, inter_hidden/128]│
├──────────────────────────────────────────────────────────────────────┤
│ combine_buf:  BF16      [num_topk, num_max_tokens_per_rank, hidden]  │
└──────────────────────────────────────────────────────────────────────┘
- num_max_tokens_per_rank表示某个EP实例在Attention之后，进入MoE还没有Dispatch的时候最多有多少个tokens
- num_max_pool_tokens表示当前rank在Dispatch后最多收到多少token，极端情况下所有rank的tokens都会Dispatch过来，因此一个Expert最多接受num_ranks x num_max_tokens_per_rank。另外一个rank还有多个Expert，不考虑tokens去重，还要再乘以最多可以命中多少个local expert，也就是min(num_topk, num_experts_per_rank)。在SymmBuffer中按照这个最大长度去申请的L1和L2 token buffer。 
- num_max_pool_blocks就是num_max_pool_tokens除以BLOCK_M，一个pool_block就是一个BLOCK_M。
This content is only supported in a Feishu Docs

Warp分工
传统 CUDA kernel 所有 warp 干同样的活。
Mega MoE 用的是 CUTLASS 风格的 persistent warp-specialized kernel：kernel 启动后，每个 warp 根据 warp_idx 选一条分支，然后终身只做这一种事。
每个 warp 在入口还会 warpgroup_reg_dealloc<...>() 调整自己的寄存器配额（Dispatch 48、MMA/TMA 40、Epilogue 208）—— 这是"按角色优化"的典型标志。
每个SM内按Warp分角色：
This content is only supported in a Feishu Docs
每个 SM 上都有 dispatch、GEMM、epilogue 这些角色，只是由不同 warp 做。不是把 SM 静态分成几组。
每个SM上都有一个Scheduler，能遍历本 rank 的所有 local experts、所有 waves。
切分warp的好处：
1. 寄存器重分配。epilogue 要做 SwiGLU、amax、cast，需要很多寄存器（208 个/线程）；TMA warp 只是发指令，40 个就够。warpgroup_reg_dealloc / warpgroup_reg_alloc 让用得少的 warp 主动把寄存器还给 SM，腾出来给 epilogue。这叫 register reallocation。
2. TMA 和 UMMA 都是发射-完成的异步指令，让一个固定 warp 一直盯着发，避免调度开销。

Dispatch
不是把token直接push到对端，而是先把这个Expert需要从哪个rank的哪个token/topk拉数据写到目标rank的workspace，目标 rank 知道自己本地 expert 需要哪些 token 后，再由自己的 dispatch warp 用 NVLink/TMA 去 pull 原始 token、scale factor 和 topk weight，整理成本 rank 本地 expert 的连续 token pool，后面 GEMM 直接按 expert pool 算。
这种 “metadata push + payload pull” 的设计让目标 rank 能控制本地 expert pool 的布局。每个 local expert 在 pool 里占一段连续空间，段内按 BLOCK_M 对齐；后面的 GEMM scheduler 只需要知道每个 expert 收到了多少 token，就能把 pool 切成一串 GEMM tile。
expert_send_count[global_ep]：当前 rank 总共要发多少 token-topk 给某个全局 expert。实际是 64-bit packed：低 32 位是 count（累计token数量），高 32 位是“有几个 SM 已完成统计”的完成标记。
expert_recv_count[src_rank][local_ep]：在目标 rank 上，某个本地 expert 要从某个源 rank 接收多少 token-topk。
src_token_topk_idx[local_ep][src_rank][slot]：在目标 rank 上记录src_token_idx * kNumTopk + src_topk_idx，即“这个 slot 应该去源 rank 拉哪个 token/topk”，所以 later 可以反解：
src_token_idx = src_token_topk_idx / num_topk;
src_topk_idx  = src_token_topk_idx % num_topk;
1. 阶段 A：所有 rank 先互相写 metadata/count
每个 rank 会根据自己的 topk_idx，向目标 expert 所在 rank 的 workspace 写：
  - src_token_topk_idx[local_expert][src_rank][slot]
  - expert_recv_count[src_rank][local_expert]
  - expert_recv_count_sum[local_expert]
2. 阶段 B：每个 rank 根据自己的 workspace，从各个 src rank pull token
pull 时用 round-robin/min-peeling 决定“本地 expert pool 的第 N 个 token 应该从哪个 src rank 的哪个 slot 拉”。

Mega MoE是大EP，也就是DP=EP，每个rank有自己独立的token，来自各自的 data parallel shard

1. 写src_token_topk_idx，warp0~warp3 全部 128 个线程一起写，不是 warp0 单干
2. 先grid_sync再跨 rank NVLink barrier，确保所有 rank 的“取货单”完整
3. warp0~3 全切换到 pull，dispatch 开始按 pool token 拉数据。给定 token_idx_in_expert（一个整数），算出它应该去拉哪个 (src_rank, slot_in_rank)。设计了一套min-peeling算法能保证同时均匀地去用TMA拉取不同src_rank的token，而不是所有的warp线程都先一股脑都去拉取rank0，这样吃不满NVLink带宽
4. 计算侧拿到 expert token count，开始调度 L1 blocks
5. 对每个 L1 block，只等自己的 l1_arrival_count 达到 valid_m
6. block0 ready 后立即算 block0；dispatch 同时继续拉 block1/block2/后续 expert 的 token

同步关系
This content is only supported in a Feishu Docs
expert_recv_count_sum是一个 64 位打包计数器：
- 低 32 位：该 expert 从所有 (rank × SM) 上累加得到的 token 数；
- 高 32 位：已经"到齐"的 producer 数量（每次 atomic_add 加 1ull << 32）。

Scheduler
Warp角色讲的是一个SM内部角色的分工，现在介绍的是整张卡128个SM怎么分配任务
Mega MoE采用的是persistent kernel模式，grid为1D，grid_dim = num_sms也就是128，也就是说一个SM上只有一个GPU概念中的block，一个 SM = 一个 CTA = blockIdx.x ∈ [0, 128)。这个 block 是常驻的——kernel 开始到结束，它一直活着，内部用 for_each_block 循环去消费很多工作项。
如果不是 persistent 模式，而是 launch 成千上万个小 block，那就由硬件调度器自由分配，你就不能假设 "blockIdx 就是 SM id"。
scheduler 的返回单位是一块 GEMM tile：
(block_phase, local_expert_idx, m_block_idx, n_block_idx)
block_phase 表示当前是 Linear1 还是 Linear2。scheduler 先从 dispatch 写好的 expert_recv_count_sum 里读到每个 local expert 的 token 数，再按 wave 推进：一个 wave 内先枚举所有 expert 的 L1 tile，再枚举同一批 expert 的 L2 tile。这样 L1 epilogue 产出的 l2_acts/l2_sf 能尽快被同一 wave 的 L2 消费。

所有 SM 要把这些任务block(这里不是GPU中的block)平均分，还要满足"wave 里先做 L1，同 wave 的 L2 才能开始"的顺序。
最朴素的做法是 round-robin：每个 SM 拿走 blockIdx.x, blockIdx.x + num_sms, ... 号 block
block_idx = blockIdx.x;     // 初始 block 号 = 自己的 SM id
// 每分配一个，block_idx += kNumSMs
所以 SM 0 做的是 L1 的第 0、128、256… 号 block，SM 1 做第 1、129、257… 号。
接下来需要考虑的就是怎么把计算任务映射到block_idx。
举例：
rank 0 上有 8 个 local expert，每个 expert 要做 L1 和 L2 两次 GEMM，每个 GEMM 在 (M, N) 上又有很多 block。把所有要做的 block 列出来：
This content is only supported in a Feishu Docs
用 BLOCK_N = 128 切 N 维：
- L1 的 N = 2·intermediate = 4096 → kNumL1BlockNs = 4096/128 = 32
- L2 的 N = hidden = 7168 → kNumL2BlockNs = 7168/128 = 56
各 expert 的 L1 block 数：
This content is only supported in a Feishu Docs
SM 0 的 block_idx 从 0 开始，每次 +128（= kNumSMs）。
(expert=0, phase=L1, m=0..1, n=0..31)  → 2×32 = 64 个 block
(expert=1, phase=L1, m=0..1, n=0..31)  → 64 个
(expert=2, phase=L1, m=0,   n=0..31)   → 32 个
(expert=3, phase=L1, m=0..3, n=0..31)  → 128 个

(expert=0, phase=L2, m=0..1, n=0..55)  → 2×56 = 112 个
(expert=1, phase=L2, m=0..1, n=0..55)  → 112 个
...

kNumL1BlockNs 表示L1沿着N维度上切分多少个block
当前 expert 有 num_m_blocks * kNumL1BlockNs 个 block，如果 block_idx 落在范围内，m_block_idx = block_idx / kNumL1BlockNs、n_block_idx = block_idx % kNumL1BlockNs；否则 block_idx -= 这个 expert 的 block 数，进入下一个 expert 重试。
L1阶段：
This content is only supported in a Feishu Docs
L2阶段
scheduler 会把 expert 指针重置回 wave 起点（align_down(3, 4)=0），但 block_idx 不清零——它保留了 L1 结束时的"余量"，block_idx -= num_m_blocks * kNumL1BlockNs 一路减下去。

Wave
调度需要确保L1做完之后才能做L2，因此为了方便管理，将Expert进行分批，一批叫做wave，只有整个wave的L1全部做完才能进行L2。
wave和SM没有对应关系，所有 SM 一起走 wave 0，一起走 wave 1。换句话说：不存在"SM 0 在 wave 0，SM 1 在 wave 1"这种错开的情况。
举例： num_experts_per_wave = 4，所以：
- Wave 0：expert 0, 1, 2, 3 的 L1 全做完 → 再做 expert 0,1,2,3 的 L2。
- Wave 1：expert 4, 5, 6, 7 的 L1 全做完 → 再做 expert 4,5,6,7 的 L2。
怎么决定Wave的大小？控制每个 expert 的 L1 block 数，尽量将SM喂饱
每个 expert 的 L1 block 数 ≈ ceil(expected_tokens/block_m) × num_n_blocks
num_experts_per_wave ≈ ceil(2 × num_sms / 每 expert 的 L1 block 数)
Wave切换
if next_phase == L1:
    if fetch_next_l1_block() ok → return L1 block
    else:   # 本 wave 的 L1 block 已经分光
        next_phase = L2
        set_expert_idx(本 wave 起点)   # 重置 expert 指针到 wave 开头
        # 注意这里并没有跨 SM 同步！
else:  # L2
    if fetch_next_l2_block() ok → return L2 block
    else:   # 本 wave 的 L2 分光
        next_phase = L1   # 进入下一个 wave
每个 SM 看到"fetch_next_l1_block 返回 false"的意思只是："我自己分到的本 wave L1 block 已经用完"。它不会等其他 SM 也用完，立刻翻到 L2 去拿下一个活。
但是L2开始的真正条件是l2_arrival_mask，也就是这个 M-block 所有 N 方向的 L1 子块都做完了 epilogue（l2_acts 写完）
- 索引粒度是 pool_block_idx，即"专家 pool 里的某一组 BLOCK_M 个 token"（一个 M-block）。
- 每完成 1 个 L1 N-block 的 epilogue，就对 mask 置 1 位。
- 总共有 kNumL1BlockNs 位（在例子里是 32 位）。
- L2 TMA-A warp 在处理 (expert, m) 这个 M-block 前，自旋等：
while (load(l2_arrival_mask[pool_block_idx]) != expected_bitmask) ;
wave的作用
既然真正的同步点是 l2_arrival_mask，那 wave 为什么还有用？
Wave 的作用是：让 scheduler 不要太早切换到L2 phase，这样可能其他SM还没算完，arrival_mask还没就绪。
想象一下没有 wave 的情况（把 wave 设成 1 个 expert）：
没 wave (每个 expert 独立切换):
SM 0:  L1(e0,n0) → 切L2 → L2(e0,n0)等...一直等... → L2 实算
               ↑
       此时 L1(e0,n1..n31) 都还没做，mask 缺 31 位
       SM 0 自旋等很久
wave=1 时，SM 做完 L1(e0,n0) 就立刻去做 L2(e0,n0)，但 l2_arrival_mask[e0,m0] 只有 1 个 bit（自己的），缺的 31 个 bit 要等其他 SM 慢慢做 L1——这期间 SM 0 就浪费在 spin wait 上。
有 wave 之后：
wave = 4 experts (本 wave L1 总 block 数变大):
SM 0:  L1(e0,n0) → L1(e1,n0) → L1(e2,n0) → L1(e3,n0) → 切L2 → L2(e0,n0)
                                                        ↑
                                         这时 SM 1~3 也在做各自 expert 的 L1
                                         l2_arrival_mask[e0,m0] 的 4 位基本凑齐
                                         L2 几乎不用等
wave 的大小就是在赌："到我切 L2 时，mask 已经就绪"。
L1 epilogue
触发条件：当前 (m, n) 块的所有 K 方向（56 个 K-block）都累加完。用 tmem.ld 指令（代码里叫 SM100_TMEM_LOAD_16dp256b1x）直接把 TMEM 搬到 warp 的寄存器
kernel 里有两组 mbarrier：
This content is only supported in a Feishu Docs
tmem_full 是 L1 epilogue 同步点。 它的含义是："这个 (m, n) 块的 56 个 K-block UMMA 全部累加完，TMEM 可以读了"。
SM100 的 SMEM 总共 232KB，扣掉固定区域后，除以每个 stage 需要的字节数，得到 kNumStages。实际算出来大约是 4（heuristic 里算出 num_stages，取 smem / smem_per_stage，并要求 >= 2）。
所以 SMEM 里确实只放得下 4 份 (A, B) tile 对。 4 个 stage 是一个环形缓冲（ring buffer）：stage 0, 1, 2, 3, 0, 1, 2, 3, ...
如果只有 1 份，TMA 搬 → UMMA 算 → TMA 搬 → UMMA 算，串行，硬件利用率低。
4 份让生产者和消费者可以错开：TMA 正在搬第 5 个 K-block 到 stage 0，同时 UMMA 可能还在算第 2 个 K-block（stage 1）。流水线重叠就靠这个。
【一个 cluster (2 SM) 处理一个 (e, m, n) 块】

  scheduler.for_each_block 吐出一个 (e, m, n)
         │
         ▼
  ┌──────────── K 维流水线 (56 轮) ────────────┐
  │  TMA-A warp:  轮转 4 个 SMEM stage 搬 A     │
  │  TMA-B warp:  轮转 4 个 SMEM stage 搬 B     │  ← 并行
  │  UMMA  warp:  等 full barrier → 发 MMA     │
  │               累加到同一个 TMEM 地址        │
  │  (K-block 之间用 full/empty 做流水套圈)    │
  └─────────────────────────────────────────────┘
         │ 第 56 次 UMMA signal tmem_full
         ▼
  ┌──────────── Epilogue (单次触发) ──────────┐
  │  Epilogue warp wait tmem_full              │
  │  读 TMEM → SwiGLU × topk_weight → FP8 量化│
  │  STSM → SMEM                               │
  │  TMA store → l2_acts[...]                 │
  │  tma_store_wait + sync_aligned             │
  │  1 个线程 atomic-OR l2_arrival_mask bit n  │
  └─────────────────────────────────────────────┘
         │
         ▼
  N 个 (e, m, *) 的 bit 齐了 (全 1)
         │
         ▼
  L2 阶段: TMA-A 读 l2_acts[整行 K] → L2 UMMA → L2 epilogue → NVLink combine
This content is only supported in a Feishu Docs

原始权重布局
训练保存的 L1 权重 N 维是：
N 维 = 2 × intermediate = 4096
┌────────────────────────────────┬────────────────────────────────┐
│        gate  (N=0..2047)        │        up  (N=2048..4095)       │
└────────────────────────────────┴────────────────────────────────┘
如果照这样用，BLOCK_N=128 的块会落在要么纯 gate，要么纯 up——你的直觉对，这种情况下必须等两半都算完才能 SwiGLU。
_interleave_l1_weights 的变换（gran=8）
权重转换函数把它重排成：
N 维 = 4096
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──┬──────┐
│gate0 │ up0  │gate1 │ up1  │gate2 │ up2  │gate3 │ up3  │..│gate... │
│ 8个  │ 8个  │ 8个  │ 8个  │ 8个  │ 8个  │ 8个  │ 8个  │  │       │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──┴──────┘
 ↑── 16 个元素的 gate/up 对 ──↑
每 8 个 gate 紧跟 8 个 up。整个 4096 维度上，gate 和 up 均匀交织。
L2 epilogue
L2 epilogue 做的事是：把 (m_block, n_block=128列) 的 FP32 累加 cast 成 BF16，直接用普通 NVLink store（16B/lane）以"token 的 128 列"为粒度撒进 combine_token_buffer[topk_idx][src_token]。本地和远端完全对称，没有本地预求和。一个 token 的 7168 列被 56 个 L2 block 分别写完；一个 token 的 6 个 topk 副本由源 rank 在 Final Combine 里 FP32 累加后 TMA store 到 y。
FP8 × FP4  ──UMMA──►  FP32 (TMEM accumulator, 32-bit per element)
                          │
                          │ SM100_TMEM_LOAD_16dp256b1x
                          ▼
                     uint32_t values[8]    (寄存器，bit 形态 = FP32)
                          │
                          │ __float22bfloat162_rn
                          ▼
                     2×BF16 packed into 32 bits
                          │
                          │ STSM (SM90_U32x4_STSM_T)
                          ▼
                     SMEM (BF16, swizzled)
                          │
                          │ ld_shared<float4> + NVLink store
                          ▼
                     combine_token_buffer (BF16)
