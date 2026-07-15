# 概述
本文档详解sglang开启hicache (--enable-hierarchical-cache) 后，目前的pd-colocation模式下，kvcache的实际管理方式，目前考虑采用write_backup策略，仅当kvcache可能被eviction时，才将kvcache写入DRAM中；
主要的代码集中在sglang/python/sglang/srt/mem_cache下
# 调用链
## HBM KVCache分配
HBM的kvcache分配仍然在ModelRunner的`initialize(...)`中调用如下进行初始化；
```py title="init_kvcache_pool"
# Init memory pool and attention backends
self.init_memory_pool(pre_model_load_memory)
```
在init_memory_pool中，核心通过profile实际可用的GPU HBM显存大小，将其封装到config中，随后，调用ModelRunnerKVCacheMixin中的`_init_pools(...)函数`
其中，核心初始化两个变量，分别是self.req_to_token_pool，代表一个请求到token/KV cache的实际映射，ReqToTokenPool的定义在memory_pool.py中
ReqToTokenPool的思路相对简单，最大同时维护size即max_running_requests个请求的req_to_token slot，其中，分配时，会根据是否是chunked prefill来避免重复占用slot；
核心关注Kvcache的分配，以MHA model为例，self.token_to_kv_pool被初始化为MHATokenToKVPool；
```py title="TokenToKVPool"
self.token_to_kv_pool = MHATokenToKVPool(
    self.max_total_num_tokens,
    page_size=self.page_size,
    dtype=self.kv_cache_dtype,
    head_num=self.model_config.get_num_kv_heads(
        get_attention_tp_size()
    ),
    head_dim=self.model_config.head_dim,
    layer_num=self.num_effective_layers,
    device=self.device,
    enable_memory_saver=self.server_args.enable_memory_saver,
    start_layer=self.start_layer,
    end_layer=self.end_layer,
    enable_alt_stream=not self.server_args.enable_pdmux,
    enable_kv_cache_copy=(
        self.server_args.speculative_algorithm is not None
    ),
)
```
核心的kvcache buffer在`MHATokenToKVPool. \_create\_buffers\_`中，具体函数定义如下，其中，k_buffer与v_buffer的分配时slots_size设置为self.size + self.page_size的含义是为了按页进行预留，因为kvcache的存取往往是按page进行的；
```py title="create_buffer"
def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.k_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, self.head_num, self.v_head_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )
```
在分配完具体的kvcache池子后，则初始化self.token_to_kv_pool_allocator，核心决定Token在池子中具体分配的位置，还是以`PagedTokenToKVPoolAllocator`为例
其中，每次alloc时，根据需要的need_size，分配所需的page，注意need_size是page的整数倍，在框架的更上层应该已经提前做了padding，随后，函数输出out_indices，代表每一页的实际连续indices；而对于有命中前缀的alloc_extend，则相对复杂一点，此处，需要保证，如果前缀所在的页并没有写完，则这次从上次的页尾开始一起开始写；
其中，具体写KVCache的kernel用triton实现了高性能的一版本，其中分为三段式：
* 先填旧尾页的剩余空位；
* 再填新分配的整页；
* 最后填新尾页的前半页；
具体的triton函数为sglang/python/sglang/srt/mem_cache/allocator.py中的`alloc_extend_kernel(...)`；
## DRAM KVCache分配
在sglang Scheduler启动的时候，从server_args中记录enable_hierarchical_cache到self.enable_hierarchical_cache中，随后，在`init_cache_with_memory_pool(...)`中，若发现使能hicache，则选择构造HiRadixCache为self.tree_cache
其中，具体的参数封装在params中，class为CacheInitParams；
```py title=""
params = CacheInitParams(
    disable=self.disable_radix_cache,
    req_to_token_pool=self.req_to_token_pool,
    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
    page_size=self.page_size,
    is_eagle=self.spec_algorithm.is_eagle(),
    tp_cache_group=(
        self.attn_tp_cpu_group
        if self.server_args.enable_dp_attention
        else self.tp_cpu_group
    ),
    eviction_policy=server_args.radix_eviction_policy,
    enable_metrics=self.enable_metrics,
    enable_kv_cache_events=self.enable_kv_cache_events,
    enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
    pp_rank=self.pp_rank,
    pp_size=self.pp_size,
    chunked_prefill_size=effective_chunked_prefill_size,
    sliding_window_size=self.sliding_window_size,
)
```
其中，params中的req_to_token_pool与token_to_kv_pool_allocator即为在model_runner中初始化的HBM侧KVCache相关；
```py title="init_kvcache_pool_host"
elif self.enable_hierarchical_cache:
    if self.is_hybrid_ssm:
        from sglang.srt.mem_cache.hi_mamba_radix_cache import (
            HiMambaRadixCache,
        )

        self.tree_cache = HiMambaRadixCache(
            params=params, server_args=server_args
        )
    else:
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        self.tree_cache = HiRadixCache(
            params=params, server_args=server_args
        )
    self.tp_worker.register_hicache_layer_transfer_counter(
        self.tree_cache.cache_controller.layer_done_counter
    )
```
其中，首先具体看一下HiRadixCache的定义，在HiRadixCache中，最重要的两个变量为self.token_to_kv_pool_host和self.cache_controller，其中，self.token_to_kv_pool_host会根据实际的attention类型初始化为MHATokenToKVPoolHost/MLATokenKVPoolHost，而self.cache_controller则类型为HiCacheController，首先以MHATokenToKVPoolHost为例看一下具体的kv_pool_host；
其中，在初始化阶段，核心初始化两个对象，分别是self.allocator和self.kv_buffer，self.allocator对于非mooncake的hicache backedn统一为HostTensorAllocator类，对于kv_buffer，调用init_kv_buffer(...)函数实现，其中根据不同的layout方式，如"page_first", "layer_first"等进行不同的dimension设置，其中默认也最推荐layer_first，每一层的kvcache存放在连续地址，这样有利于laywise的加载kvcache，随后调用alloc_with_host_register进行内存分配，具体代码段如下，其中pin_memory默认为True：
```py title="Allocate\_host\_buffer"
def alloc_with_host_register(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        torch.cuda.cudart().cudaHostRegister(
            buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
        )
    return buffer
```
其中allocator的alloc函数实则为调用torch.empty(dims, dtype=dtype, device=devide)分配内存；
当分配完了Host (DRAM)侧的kvcache后，另一个核心的在HiRadixCache中初始化的为self.cache_controller，初始化class为HiCacheController，我们观察init函数，
```py title="HiCacheController"
class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.enable_storage_metrics = enable_storage_metrics

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        # NOTE: Do NOT reuse `self.stop_event` here since it also guards core HiCache
        # transfer buffers (CPU<->GPU). We want to allow runtime attach/detach of
        # storage without stopping the whole controller.
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()
        
        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e
```
其中，观察到初始化了self.layer_done_counter与`self.mem_pool_device.register_layer_transfer_counter(...)`函数，layerDoneCounter的完整定义如下，可以理解为生产者消费者的信号，producer负责清空event，选取可用槽，而consumer则后续查看该槽位，调用wait来等待具体某层的event完成；设置num_counters=3的目的是为了producer和consumer的overlap；register_layer_transfer_counter(...)做的事情相对简单，即将device KVCache的layer_done_counter指针赋值为刚刚初始化完成的；
```py title="Layerloading"
class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]

class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1
```
其中，在init(...)初始化时，创建如下的load_queue, write_queue与write_buffer等等，用来做CPU kvcache到GPU kvcache之间的传输；
```py title=""
 self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()
```
其中，我们暂时不关注从Storage backend到GPU HBM的prefetch与write_storage，我们关注CPU Kvcache到GPU KVcache到load和store；
对于write操作，即从device kvcache写入host kvcache，核心先从mem_pool_host.alloc分配一块host_indices，随后将对应的device_indices到host_indices的操作连同node_id和priorty一起包装成一个`CacheOperation`，随后调用start_writing(...)函数；
在start_writing(...)中，首先将self.write_queue中的所有CacheOperation包装成一个OP，首先记录start_event，确保后续的写操作是在default stream上所有操作完成才可进行，如move_indices等，随后调用mem_pool_host.backup_from_device_all_layer，此处实则调用一个封装好的triton kernel，将所有层高效的写到CPU DRAM上；注意，其中需要用record_stream(...)来严格保证host_indices和device_indices在write_stream执行操作时，不会被释放；最后，在ack_write_queue中添加ACK；
```py title=""
def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(op)
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))
```
随后来看load操作，意为，从CPU DRAM将KVCache搬运到GPU HBM中，其中，也会首先调用memory_pool_device_allocator，首先分配与host_indices等长的显存，随后封装op至self.load_queue中，但在其中，并不显示的调用start_loading(...)函数；
在start_loading函数中，首先获取之前的self.layer_done_counter中一个可用的producer_id，即可用的LayerLoadingEvents，随后merge op，与写不同的是，load采用layewise的loading方式，每次调用load_to_device_per_layer后，就会在同一条stream (load stream)后插入一个producer_event.loading_event(i)，保证在上层框架做forward时可以laywise的forward，掩盖从DRAM中load的时间；
## 实际触发load/store位置
## 从DRAM往HBM中写KVCache
接下来关注在实际推理时，具体的load/store的触发位置，如start_loading函数的调用位置等等；
首先关注何时走到可能准备kvcache的地方，首先关注Scheduler.dispatch_event_loop(...)函数，其中，由于我们采用colocation模式，所以采用的event loop为event_loop_overlap(...)，其中，在循环中，首先调用recv_requests(...)获取收到的请求，随后通过调用process_input_requests(...)利用type dispatcher把TokenizedGenerateReqInput分发到handle_generate_request()，在handle_generate_request(...)中，根据收到的recv_req，构造Req，随后将其放入waiting_queue中，在时间循环中调用
```py title=""
# Get the next batch to run
batch = self.get_next_batch_to_run()
```
来获取下一个batch来执行推理，其中，核心调用get_new_batch_prefill(...)函数，而该函数调用_get_new_batch_prefill_raw(...)，在其中遍历waiting_queue，对每个候选请求调用req.init_next_round_input(self.tree_cache)，核心在其中调用HiRadixCache的match_prefix(...)函数
```py title=""
if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)
```
其中，match_prefix的逻辑在于，从radixCache中找到最长的前缀匹配，返回values和last_node，其中需要注意的是，对于radix tree，如果evict的话，一定是从树的叶节点逐步往根节点evict，同样的，从DRAM中进一步丢弃，也是从叶子节点逐步丢弃；
```py title=""
def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        page_aligned_len = len(key)
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )
```
match_prefix(...)的返回值分别是，直接命中的device_indices，即在HBM中驻留的kvcache、最后一个device kvcache代表的tree中的last_node、最后一个在DRAM kvcache代表的tree的last_host_node、host命中的kvcache长度；
在init_next_round_input(...)结束后，调用PrefillAddr.add_one_req(...)，核心首先计算实际新增的token数量，并调用ceil_paged_tokens向上取整，self.\_lock\_node即设置从根节点到req.last_node的所有kvcache为被protected，保证不会被换出，随后根据req.host_hit_length是否大于0来判断，是否需要从self.tree_cache中加载kvcache，此处调用init_load_back(...)函数；
```py title=""
with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if req.host_hit_length > 0:
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        last_host_node=req.last_host_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len

            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                return AddReqResult.OTHER

            # ... ...
```
随后仔细查看HiRadixCache中的init_load_back(...)函数，如果last_node被evict了，则直接调用self.load_back(...)函数
```py title=""
def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )
```
接下来详解下load_back(...)函数，其中将node按照正常顺序，放入nodes_to_load队列中，保证后续load时从前往后load，并且记录ancestor_node为protected node；随后判断，除非loading_back的数据量太小或者太大，取消传输，否则，调用cache_controller.load(...)函数加载device_indices，**其中device_indices为实际分配的page的逻辑位移，并不代表真实已经分配了一部分显存**，**但同时，也意味着从逻辑上，这块地址已经被这个请求分走了** (所有layer)；
再回到add_one_req(...)函数中，通过将新分配到的逻辑地址device_indices和原先的req.prefix_indices做torch.cat，实际获得了该req的所有kvcache的page逻辑地址；
在add_one_req(..)结束后，在_get_new_batch_prefill_raw(...)函数下，我们发现下一步会调用ScheduleBatch.init_new(...)创建一个新的batch，其中addr.can_run_list核心为在add_one_req(...)时prefillAddr认为满足运行条件的request集合，其中在new_batch准备调用prepare_for_extend开始准备forward前，如果使能hicache，则开始从DRAM加载KVCache，其中tree_cache.ready_to_load_host_cache(...)实则就是调用`CacheController的start_loading(...)`函数，其中，具体的函数细节已在上文中展示，在此不加赘述；
```py title="Scheduler.py"
 # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()
```
在prepare_for_extend中，会调用alloc_for_extend(...)来为未命中的部分分配资源，核心调用
```py title=""
last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
        )
```
其中alloc_paged_token_slots_extend(...)即为调用allocator.alloc_extend(...)函数，随后，调用write_cache_indices(...)将该req对应的token写入reqToTokenPool中；
在完成了get_next_batch_to_run(...)上述的准备batch的操作以后，核心调用run_batch(...)尝试执行，在run_batch中，会先调用batch.get_model_worker_batch获取封装好的ModelWorkerBatch类，其中包含self.hicache_consumer_index信息，随后，实际调用forward_batch_generation(...)来进行生成；
```py title=""
with self.forward_stream_ctx, self.record_bubble_metrics(batch):
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    with self.record_forward_metrics(batch):
                        batch_result = self.model_worker.forward_batch_generation(
                            model_worker_batch
                            # here pp is not compatible with overlap
                        )
                    # FIXME(lsyin): maybe move this to forward_batch_generation
                    batch_result.copy_done = self.device_module.Event()
                    if batch_result.delay_sample_func is None:
                        self.future_map.store_to_map(future_indices, batch_result)
                        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
                    else:
                        batch_result.future_indices = future_indices
```
其中在forward_batch_generation(...)中，会调用set_hicache_consumer将hicache_consumer_index设置到TP worker上；
实际调用modelRunner.forward(...)函数，即调用_forward_raw(...)函数，随后会根据是prefill还是decode分别做forward_extend(...)与forward_prefill(...)，在其中，首先调用**init_forward_metadata(...)** ，其中的pagetable很重要，初始化metadata，随后调用forward函数，以Qwen2为例，整理链路很简单易懂，最终会调用到RadixAttention.forward(...)函数进行attention计算，而RadixAttention.forward(...)则会根据实际的attention backend来转发到不同的forward_extend(...)和forward_decode(...)函数；
其中不论是forward_decode还是forward_extend(...)都需要调用pool.get_kv_buffer来获取k_buffer和v_buffer，其中则需要按层等待kvcache被成功load
```py title=""
def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)
```
**注意：这样看起来，每次获取buffer时，是laywise的去等待kvcache被加载完成，但这里是否意味着他的加载往往不是请求粒度的load/store？**
## 从HBM把KVCache写回DRAM
