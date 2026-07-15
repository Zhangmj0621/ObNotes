# 概述
本文详解当调用sglang abort_request后，是如何打断具体的请求以及管理KVCache的，并探讨是否能够支持在abort_request时，将请求KVCache全部从HBM中卸载；
# Code walkthrough
在强化框架中，我们往往直接调用tokenzier_manager.abort_request(...)，其中，将请求封装为Req，将其发送给scheduler
```py title="abort_request"
def abort_request(self, rid: str = "", abort_all: bool = False):
        if not abort_all and rid not in self.rid_to_state:
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self.send_to_scheduler.send_pyobj(req)
        if self.enable_metrics:
            # TODO: also use custom_labels from the request
            self.metrics_collector.observe_one_aborted_request(
                self.metrics_collector.labels
            )
```
Scheduler在event_loop中调用process_input_requests处理收到的所有请求，其中，根据req类型为AbortReq将请求转发到abort_request(...)函数，在其中，对于未完成的请求，并不会直接打断该请求，而是设置该req.to_finish为FINISH_ABORT()，需要等待他再跑一个decode forward后再退出；
在SchedulerOutputProcessorMixin.process_batch_result_decode(...)中，会调用req.check_finished(...)，其中，如果发现self.to_finish不为none，则将finished_reason置为self.to_finish，随后，在_handle_finished_req(...)中，会调用release_kv_cache(req, self.tree_cache)，在其中，核心调用tree_cache的cache_finished_req(...)，
```py title=""
def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(req, is_insert=is_insert)

    # FIXME: SessionAwareCache.cache_finished_req sets req_pool_idx = None to
    # transfer KV ownership to the SessionSlot, so we skip the remaining
    # cleanup (overalloc free + pool slot free). This means over-allocated
    # tokens from speculative decoding are NOT freed between turns.
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()

    global_server_args = get_global_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    if spec_algo is None:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    tree_cache.req_to_token_pool.free(req)
```
在cache_finished_req(...)中，会根据page_size获取实际值得插入tree中的新的extend的kv，此处插入时，对于Hicache与write_through模式，会将kvcache同步写入node.value与node.host_value中，随后，在cache_finished_req(...)中，会释放没有对齐的tail，也即对齐pagesize时超出的部分kv，这部分会被直接evict，随后调用dec_lock_ref来将lock_ref为1的kvcache从protected置为evictable；
```py title=""
def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)
```
随后在release_kv_cache(...)函数中，会把req_to_token_pool中的req释放，即释放一个可用的max_running_requests槽位；
注意，HBM到DRAM的写操作并不是同步完成的，而是异步完成，为了防止KVCache被提前释放，此处将node添加到self.ongoing_write_through[node.id]中，并调用inc_lock_ref给node加引用来使其进入protected状态，后续在HiRadixCache.writing_check(...)中，根据具体的ack_list，找到对应的node，调用dec_lock_ref(...)将其解锁；
