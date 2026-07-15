# New：KVCache Manager
由于在decode阶段引入layerwise的kvcache load/store，显然会严重损害decode侧的吞吐，因而，无法采用上面的在HBM中只驻留部分层的策略，来期望维持bsz同时最大化吞吐；
分析目前引入priority-based scheduler后的问题
KVCache trashing
- Rollout engine对实际驻留在HBM中的非活跃KVCache无感知，当新请求到达时，由于不存在充足的可用kvcache，需要evict HBM中的KVCache，此时应该对KVCache有状态设置，high_ref与low_ref，high_ref>0代表高优Token cache，low_ref>0代表低优token cache，high_ref=0 && low_ref==0代表可被直接换出cache；在需要evict请求时，优先evict无用cache，否则evict低优先级cache，仅当请求为高优先级请求时才允许将高优先cache也evict到HBM中，evict时，采用按需evict策略，从radix tree的叶子结点往上evict，一旦到达所需cache page，则立刻停止evict；
- 若将所有可用的kvcache都evict到DRAM中，仍然无法满足该请求的所需的kvcache，则考虑，高优先级请求停止某个低优先级请求，直到满足存在可用kvcache；
- 若仍然还未满足，则观察目前已partial page存放的kvcache，通过离线profiling结果，判断当前请求改为partial page后，load/store开销小于TPOT，则作为partial page存放在HBM中，将其加入bsz中；
# Code Walkthrough
本节详解新增的RefAwareHiRadixCache的具体代码流程，以及具体实现的新策略；
## HiRadixCache
首先详解下HiRadixCache的具体思路；
HiRadixCache中，在初始化时，首先根据kvcache的类型，进行host DRAM kvcache的初始化，目前仅支持MHATokenToKVPool，NSATokenToKVPool以及MLATokenToKVPool三类，赋值到self.token_to_kv_pool_host中；此外，初始化self.cache_controller，正常情况，设置为HiCacheController；此外，会额外创建多个set来记录目前正在ongoing的不同操作，来避免evict时错误evict正在进行操作的TreeNode；设置write_through_threshold，当write_policy为"write_through"时为1，不然为2，并创建self.evictable_host_leaves来记录可被evict的node set；
```py title=""
# record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
```
其中，核心提供如下方法：
**写回 Host / Storage（GPU → CPU → L3**）                                                                                                                                                                 
  - write_backup (line 653)：把一个 TreeNode 的 device value 复制到 host。约束：write-through 模式下必须父节点已 backup（保证连续前缀）。host    
  内存不足时先 evict_host。成功后保存 host_value、加入 ongoing_write_through，并 inc_lock_ref 防止 GPU 端被驱逐。
  - write_backup_storage (line 688)：进一步把 host 上的 KV 写到 L3 storage，记录 ongoing_backup 并 protect_host。                                
  - _inc_hit_count (line 705)：write-through-selective 策略下，节点命中次数达到阈值才触发 write_backup。                                         
  - writing_check (line 716)：轮询 ack_write_queue，确认异步 DMA 完成后从 ongoing_write_through 摘除节点，发出 store(CPU) event；若启用 storage  
  则继续触发 write_backup_storage。多 TP 用 all_reduce(MIN) 同步进度。
**读取与回填（CPU → GPU）**                                                                                                                                                               
  - match_prefix (line 1222)：先在 radix 树上匹配出 device 命中段；遇到 evicted 节点继续向上追溯，统计 host_hit_length，并定位最近的             
  last_host_node（已 backup 的最深祖先）。返回 device/host 双层命中信息。
  - load_back (line 952)：把命中的 host 节点链 KV 拷回 GPU。先 inc_lock_ref 保护，再申请 device 内存（不足则触发 evict）；过小（<                
  load_back_threshold）或超过 mem_quota 则放弃。成功后 split 回写到各节点的 value，记录 ongoing_load_back。                                      
  - init_load_back (line 1024)：调度入口，如果失败则向上回退到未 evict 的祖先。
  - loading_check (line 765)：处理 ack_load_queue，DMA 完成后释放 lock。
  **驱逐（Eviction）**                                                                                                                            
  - evict (line 844)：基于 eviction_strategy.get_priority 的最小堆驱逐 GPU 端叶子。未 backup 节点：write-back 模式先调用 write_backup(write_back=True) 异步刷盘再驱逐；其他模式直接 _evict_regular（释放 device内存并删除节点）。已 backup 节点：_evict_backuped 只释放 device，保留 host_value（节点变 evicted）。
  - evict_host (line 917)：在 evictable_host_leaves 上驱逐 host KV，要求节点已 evicted 且 host_ref_counter == 0，然后从父节点摘除并维护 leaf状态。
  - _update_host_leaf_status (line 829)：维护"哪些 host 节点是叶子（无 evicted child）"的集合，用于 host 驱逐候选。                              
**锁引用与 Insert**                                                                                                                             
  - inc_lock_ref / dec_lock_ref (line 791/807)：从节点向 root 累加引用，更新 evictable_size_ 与 protected_size_，同步刷新 leaf / host-leaf 集合。
  - insert (line 1405)：在 radix 树插入新 KV。处理三种情况：完整匹配（含 evicted 节点的 KV重算回填）、部分匹配（_split_node）、新增叶子。需要时计算 hash_value 用于 storage 寻址，并触发 store event 与 hit count 更新。                 
  - _split_node (line 1377)：标准 radix split，注意同时切分 value、host_value、hash_value，保留 lock_ref。 
**L3 Prefetch（Storage → Host）**                                                                                                                
  - prefetch_from_storage (line 1258)：发起异步预取。先做 page 对齐与阈值/限速判断，protect_host 防祖先被驱逐，分配 host 内存（不足则evict_host），调用 cache_controller.prefetch 启动后台 op，登记到 ongoing_prefetch。
  - check_prefetch_progress (line 1142)：每个调度 tick 检查 prefetch 是否能终止（受 prefetch_stop_policy：best_effort / wait_complete / timeout，并跨 TP all_reduce 同步）；终止后取已完成 token 数（取所有 TP 的 MIN），通过 _insert_helper_host 把新增的 host 段插入 radix树，释放未使用部分 host 内存，记录"实际从 storage 加载的 token 数"用于 metrics 与 L3 命中统计。
  - can_terminate_prefetch / _prefetch_timeout_check_linear_func (line 1103/1095)：判断停止条件，超时阈值随 page 数线性增长。                    
  - terminate_prefetch / release_aborted_request (line 1205/1489)：主动终止或清理已 abort 请求的 prefetch 状态。                                 
  - _insert_helper_host (line 1308)：仅插入 host 层（不带 device value），用于 prefetch 落地。                                                    
**事件循环入口**                                                                                                                                 
  - check_hicache_events (line 1056)：调度器每个 step 调用，依次执行 writing_check、loading_check、drain_storage_control_queues（合并 prefetch revoke / backup ack / host mem release 三类信号，且只做一次 TP all_reduce 来减少同步开销）以及 storage metrics 上报。
## RefAwareHiRadixCache
首先详解RefAwareHiRadixCache的具体class类；
首先建立class类RefInfo，代表某个请求 (rid)具体绑定的token kvcache的相关信息，包含代表该请求的优先级，以及其reference的TreeNode set；
```py title="RefInfo"
@dataclass
class RefInfo:
    is_high: bool
    nodes: Set[TreeNode] = field(default_factory=set)
```
并分三tier为每个token的kvcache设定优先级，分别是，如果没有高低优先级的请求在使用的token，则优先级为**TIER_UNUSED=0**，如果被某个inflight的低优先级请求使用，则为**TIER_LOW_REF=1**，如果有某个inflight的高优先级请求使用，则为**TIER_HIGH_REF=2**，通过这样的优先级划分，能够让我们更好的在需要evict node kvcache时做决策；
```py title=""
# Eviction tier constants
TIER_UNUSED = 0    # high_ref == 0, low_ref == 0
TIER_LOW_REF = 1   # high_ref == 0, low_ref > 0
TIER_HIGH_REF = 2  # high_ref > 0
```
再来看RefAwareHiRadixCache的具体实现，其中，直接继承自HiRadixCache，提供如HiRadixCache一样的方法， 核心有如下的一些差别：
```py title=""
class RefAwareHiRadixCache(HiRadixCache):
    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self.high_priority_threshold = server_args.high_priority_threshold
        self.unused_evictable_leaves: set = set()
        self.low_ref_evictable_leaves: set = set()
        self.high_ref_evictable_leaves: set = set()
        self.unused_evictable_size_: int = 0
        self.low_ref_evictable_size_: int = 0
        self.high_ref_evictable_size_: int = 0
        self.rid_to_ref_info: Dict[str, RefInfo] = {}
        self._evict_scope_stack: list[tuple[bool, bool]] = []
        super().__init__(params=params, server_args=server_args)
```
核心记录将可被evict的leaves分为三类，未被使用的/被低优先使用的/被高优先使用的；

# Benchmark
## Stress test
staleness=0.5
baseline
```py title=""
64
  High-priority clients:           64/96
  Duration:                        295.08s
  Active decode wall time:         0.00s
  High-priority turns completed:   800
  High-priority completion tokens: 204800
  High-priority out tok (wall):    694 tok/s
  High-priority out tok (decode):  146085770 tok/s
  High-priority engine decode:     N/A (server did not return decode_throughput)
  High-priority cache hit:         0.9087
  High-priority turn>=1 cache hit: 0.9582
  High-priority TTFT mean/p90/p99: 16.770 / 35.162 / 44.818s
  High-priority Lat mean/p90/p99:  16.770 / 35.162 / 44.818s

96
  High-priority clients:           96/144
  Duration:                        451.30s
  High-priority makespan:          451.30s
  Active decode wall time:         0.00s
  High-priority turns completed:   1245
  High-priority completion tokens: 318720
  High-priority out tok (wall):    706 tok/s
  High-priority out tok (decode):  148645512 tok/s
  High-priority engine decode:     N/A (server did not return decode_throughput)
  High-priority cache hit:         0.9117
  High-priority turn>=1 cache hit: 0.9588
  High-priority TTFT mean/p90/p99: 25.939 / 46.898 / 74.079s
  High-priority Lat mean/p90/p99:  25.939 / 46.898 / 74.079s
  High-priority turn0 TTFT p90/p99:74.080 / 74.092s
  High-priority turn>=1 TTFT p90/p99:28.007 / 47.157s
  
128
High-priority clients:           128/192
  Duration:                        1311.54s
  High-priority makespan:          1311.54s
  Active decode wall time:         0.00s
  High-priority turns completed:   1638
  High-priority completion tokens: 419328
  High-priority out tok (wall):    320 tok/s
  High-priority out tok (decode):  149214944 tok/s
  High-priority engine decode:     N/A (server did not return decode_throughput)
  High-priority cache hit:         0.6231
  High-priority turn>=1 cache hit: 0.6558
  High-priority TTFT mean/p90/p99: 82.310 / 172.383 / 196.250s
  High-priority Lat mean/p90/p99:  82.310 / 172.383 / 196.250s
  High-priority turn0 TTFT p90/p99:135.726 / 135.790s
  High-priority turn>=1 TTFT p90/p99:174.618 / 196.371s

  High-priority per-turn breakdown:
    Turn  0: n= 128, avg_out=  256.0, cache_hit=0.0000, ttft=85.744s
    Turn  1: n= 128, avg_out=  256.0, cache_hit=0.9385, ttft=64.239s
    Turn  2: n= 128, avg_out=  256.0, cache_hit=0.8616, ttft=31.156s
    Turn  3: n= 128, avg_out=  256.0, cache_hit=0.5506, ttft=70.023s
    Turn  4: n= 128, avg_out=  256.0, cache_hit=0.3414, ttft=117.115s
    Turn  5: n= 121, avg_out=  256.0, cache_hit=0.2599, ttft=142.194s
    Turn  6: n= 116, avg_out=  256.0, cache_hit=0.2502, ttft=152.476s
    Turn  7: n= 108, avg_out=  256.0, cache_hit=0.1774, ttft=157.121s
    Turn  8: n= 100, avg_out=  256.0, cache_hit=0.3552, ttft=137.530s
    Turn  9: n=  85, avg_out=  256.0, cache_hit=0.4896, ttft=113.922s
    Turn 10: n=  77, avg_out=  256.0, cache_hit=0.8699, ttft=64.822s
    Turn 11: n=  69, avg_out=  256.0, cache_hit=0.9652, ttft=29.412s
    Turn 12: n=  62, avg_out=  256.0, cache_hit=0.9667, ttft=22.810s
    Turn 13: n=  55, avg_out=  256.0, cache_hit=0.9680, ttft=20.393s
    Turn 14: n=  52, avg_out=  256.0, cache_hit=0.9692, ttft=17.691s
    Turn 15: n=  48, avg_out=  256.0, cache_hit=0.9704, ttft=16.785s
    Turn 16: n=  37, avg_out=  256.0, cache_hit=0.9714, ttft=13.712s
    Turn 17: n=  30, avg_out=  256.0, cache_hit=0.9724, ttft=10.484s
    Turn 18: n=  22, avg_out=  256.0, cache_hit=0.9733, ttft=8.280s
    Turn 19: n=  16, avg_out=  256.0, cache_hit=0.9742, ttft=6.282s
```
Proactive KVcache manager
```py title=""
64
  High-priority clients:           64/96
  Duration:                        307.54s
  High-priority makespan:          269.26s
  Active decode wall time:         0.00s
  High-priority turns completed:   800
  High-priority completion tokens: 204800
  High-priority out tok (wall):    666 tok/s
  High-priority out tok (decode):  141443984 tok/s
  High-priority engine decode:     N/A (server did not return decode_throughput)
  High-priority cache hit:         0.9087
  High-priority turn>=1 cache hit: 0.9582
  High-priority TTFT mean/p90/p99: 14.040 / 17.587 / 44.773s
  High-priority Lat mean/p90/p99:  14.040 / 17.587 / 44.773s
  High-priority turn0 TTFT p90/p99:44.775 / 44.778s
  High-priority turn>=1 TTFT p90/p99:15.579 / 17.975s
  
96
  High-priority clients:           96/144
  Duration:                        463.12s
  High-priority makespan:          369.56s
  Active decode wall time:         0.00s
  High-priority turns completed:   1245
  High-priority completion tokens: 318720
  High-priority out tok (wall):    688 tok/s
  High-priority out tok (decode):  120948938 tok/s
  High-priority engine decode:     N/A (server did not return decode_throughput)
  High-priority cache hit:         0.9117
  High-priority turn>=1 cache hit: 0.9588
  High-priority TTFT mean/p90/p99: 19.138 / 25.338 / 67.107s
  High-priority Lat mean/p90/p99:  19.138 / 25.338 / 67.107s
  High-priority turn0 TTFT p90/p99:67.114 / 67.121s
  High-priority turn>=1 TTFT p90/p99:20.059 / 26.865s
  
128
  High-priority clients:           128/192
  Duration:                        606.46s
  High-priority makespan:          474.72s
  Active decode wall time:         0.00s
  High-priority turns completed:   1638
  High-priority completion tokens: 419328
  High-priority out tok (wall):    691 tok/s
  High-priority out tok (decode):  104660024 tok/s
  High-priority engine decode:     N/A (server did not return decode_throughput)
  High-priority cache hit:         0.9109
  High-priority turn>=1 cache hit: 0.9587
  High-priority TTFT mean/p90/p99: 24.897 / 29.171 / 100.443s
  High-priority Lat mean/p90/p99:  24.897 / 29.171 / 100.443s
  High-priority turn0 TTFT p90/p99:100.445 / 100.459s
  High-priority turn>=1 TTFT p90/p99:27.896 / 39.949s

  High-priority per-turn breakdown:
    Turn  0: n= 128, avg_out=  256.0, cache_hit=0.0000, ttft=78.558s
    Turn  1: n= 128, avg_out=  256.0, cache_hit=0.9385, ttft=27.619s
    Turn  2: n= 128, avg_out=  256.0, cache_hit=0.9428, ttft=19.447s
    Turn  3: n= 128, avg_out=  256.0, cache_hit=0.9466, ttft=20.258s
    Turn  4: n= 128, avg_out=  256.0, cache_hit=0.9500, ttft=20.685s
    Turn  5: n= 121, avg_out=  256.0, cache_hit=0.9529, ttft=22.215s
    Turn  6: n= 116, avg_out=  256.0, cache_hit=0.9555, ttft=22.912s
    Turn  7: n= 108, avg_out=  256.0, cache_hit=0.9579, ttft=23.042s
    Turn  8: n= 100, avg_out=  256.0, cache_hit=0.9600, ttft=21.066s
    Turn  9: n=  85, avg_out=  256.0, cache_hit=0.9619, ttft=19.721s
    Turn 10: n=  77, avg_out=  256.0, cache_hit=0.9636, ttft=17.383s
    Turn 11: n=  69, avg_out=  256.0, cache_hit=0.9652, ttft=15.987s
    Turn 12: n=  62, avg_out=  256.0, cache_hit=0.9667, ttft=14.762s
    Turn 13: n=  55, avg_out=  256.0, cache_hit=0.9680, ttft=14.498s
    Turn 14: n=  52, avg_out=  256.0, cache_hit=0.9692, ttft=15.205s
    Turn 15: n=  48, avg_out=  256.0, cache_hit=0.9704, ttft=18.286s
    Turn 16: n=  37, avg_out=  256.0, cache_hit=0.9714, ttft=22.757s
    Turn 17: n=  30, avg_out=  256.0, cache_hit=0.9724, ttft=21.356s
    Turn 18: n=  22, avg_out=  256.0, cache_hit=0.9733, ttft=15.367s
    Turn 19: n=  16, avg_out=  256.0, cache_hit=0.9742, ttft=12.484s
```
