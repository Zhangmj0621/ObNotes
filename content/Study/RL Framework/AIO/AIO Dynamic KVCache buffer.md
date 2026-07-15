# DynamicPagedTokenToKVPoolAllocator
查看alloc函数，其中的核心函数在于_allocate_pages(...)，返回两个，分别是第一个torch.Tensor，代表分配的page，如[A,A,B,B,C]，而out_slot_starts则代表分配的page中对应的layer，假设为低优先级，则可能为[1,3,5,7,1]，其中_allocate_pages(...)会根据目前请求的优先级is_high_priority来分配对应优先级的页，采用的分配策略为，优先从已分配页的未分配部分，分配page，若无已分配相同优先级的页的page可用，则从free_pages分配页；
```py title=""
def _allocate_pages(
        self,
        num_pages: int,
        is_high_priority: bool,
        return_slot_indices: bool = False,
    ):
        # ... ...
```
注意，为了保证计算时，相同的层的kvcache都在同一层上，分配时，会根据前序分配的kvcache所在的实际slot，分配对应的layer的kvcache，采用如下优先级；
 1. 从 partial pool 找匹配 slot_start 的槽
  2. 从已展开的 fresh slots 中找匹配的
  3. 打开一个新 raw page 再试匹配
  4. 降级：接受任意 partial 槽
  5. 降级：接受任意 fresh 槽
如果没有对应的slot_start，则优先分配，partial page中slot_start剩余最多的slot_start；
## 证明：为什么Dynamic Page Layer是2？
能够证明，如果load时间小于forward单层时间，则没有必要分配新的dynamic page layer，如果load时间大于forward单层时间，则在load第 (n+1)层时，第(n-1)层的buffer必定已经释放；所以dynamic Page layer恒等于2，与具体的slot_widths无关；
# 动态加载KVCache到HBM中
核心idea：固定在HBM中只放几层，每个请求分配动态一层page，不断换入新的
首先仔细查看返回实际返回分配的kvcache的位置，在sglang中，Scheduler会通过调用get_new_batch_prefill(...)，随后调用_get_new_batch_prefill_new(...)来遍历waiting_queue，对每个请求调用init_next_round_input(...), add_one_req(...)，并将req放入can_run_list中组成batch，以供创建新的batch，
首先观察init_next_round_input(...)中，核心就是调用tree_cache中的match_prefix(...)函数，返回值分别是，直接命中的device_indices，即在HBM中驻留的kvcache、最后一个device kvcache代表的tree中的last_node、最后一个在DRAM kvcache代表的tree的last_host_node、host命中的kvcache长度；**(注意，当使能DynamicKVCache后，保存的Radix Tree中的value信息应该同时包含slot和page**，此处似乎不需要对HiRadixCache进行修改；
随后调用add_one_req(...)将请求放入can_run_list中，其中，需要将host_hit_length的部分调用self.tree_cache.init_load_back(...)加载到HBM中，注意，此时，是对每个请求调用load将其放入op_queue中，并不代表真正的加载，真正的加载在后续建完batch后统一调用start_loading(...)，在**此处，需要实现新的init_load_back(...)函数**；即**DynamicHiCacheController**，其中，根据alloc分配新页时，需要额外传入优先级以及前面prefix length的slots_start做分配 允许计算时使用动态的KVCache layout
实际尽量将kvcache分配到同一层中；
```py title=""
def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices
```
