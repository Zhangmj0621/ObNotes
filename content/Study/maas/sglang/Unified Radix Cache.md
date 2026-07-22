# 概述
本文档详解Sglang中为Deepseek V4架构提出的 Unified RadixCache 的tree cache结构。
从整体架构上来讲，UnifiedRadixCache的提出主要是为了统一不同的tree cache，似的不需要额外提供不同的tree cache的mixin，而是统一为full attention, sliding window attention, Mamba attention作为**pluggable components**存在于UnfiedRadixCache中。
```py title="UnifiedRadixCache"
┌───────────────────────────────────────────────┐
│              UnifiedRadixCache                │
│            (unified_radix_cache.py)           │
│                                               │
│  root_node ──► UnifiedTreeNode (radix tree)   │
│  components ► {ComponentType → TreeComponent} │
│  lru_lists ─► {ComponentType → UnifiedLRUList}│
└──────────┬───────────┬───────────┬────────────┘
           │           │           │
           ▼           ▼           ▼
   ┌────────────┐ ┌──────────┐ ┌─────────────┐
   │    Full    │ │   SWA    │ │    Mamba    │
   │ Component  │ │Component │ │  Component  │
   └─────┬──────┘ └────┬─────┘ └──────┬──────┘
         │             │              │
         └─────────────┼──────────────┘
                       ▼
               ┌──────────────┐
               │TreeComponent │
               │    (ABC)     │
               └──────────────┘
```
# Code walkthrough
## Key Data Structure
* UnifiedTreeNode: 每个node存储每个component的data独立，根据ComponentType.Full, ComponentType.SWA, ComponentType.MANBA进行区分
* UnifiedLRUList：双链表，连接同一个component_type的UnifiedTreeNode，host LRU和device LRU分别维护两个双链表来保证独立性，支持O(1)原语如insert/remove/promote和O(L)的eviction；
* ComponentData：每个node上存储的per-component的data，
  * value: Tensor | None - 指向component的memory pool的device指针 (TokenToKVPool, SWAKVPool, MambaPool)，‘None'意味着该数据已经被evict了，但是data结构还在 (可能保存在host中)
  * lock_ref: int：记录这个node被active request引用的数量，同原先
  * metadata: dict：component-specific state，比如SWE存储component_uuid来作为window-lock边界追踪
  * host_value：Tensor：存放在host中的cache位置
  * host_lock_ref：int，记录host-side component data被引用的位置，用于保护避免host eviction
## code详解
本节一个文件文件详细拆解具体的Unified tree cache的实现
### tree_component.py
首先查看tree_component.py，其中包含了具体的数据结构，如ComponentType，用来判断是FullComponent还是swa还是mamba，核心维护了ComponentData，用于在每个Node上存储per-component的data，详细数据结构如下图所示，其中，特定component相关信息存放在metadata中。
```py title="ComponentData"
@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    host_value: Optional[torch.Tensor] = None
    host_lock_ref: int = 0
```
其中，特定额外了新增的几个结构，用于判断，如EvictLayer判断目前是否开启hicache，CacheTransferPhase判断目前是否正处于特定的某个阶段，如正在write_back到host，或正在从stroage层prefetch回等等，有LRURefreshPhase判断是否处于特定的节点，如prefix match结束等等，并且，此处还维护了唯一的component uuid，来保持全局的独立性
```py title=""
def next_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER
```
随后查看TreeComponent，其中，维护ComponentType, UnifiedRadixCache, ComponentData等信息，并提供node_has_component_data, refresh_lru等方法，该TreeCompoent本质为一个abstract ABC类，具体根据full attention, swa, mamba会分别实现新的TreeComponent；
### unified_radix_cache.py
随后查看unified_radix_cache.py中的具体实现，其中实现了UnifiedTreeNode, UnifiedLRUList, UnifiedRadixCache等一系列核心函数；
首先查看UnifiedTreeNode，其中相比于普通TreeNode，核心改动在于，UnifiedTreeNode会维护独有的tree_components结构，用于处理UnifiedTreeNode的componentData数据，并且，该结构额外维护了全局统一的last_access_time和creation_time，确保在access和create两个操作上，对所有的TreeNode有先后优先级，并额外维护了lru_prev和lru_next代表来实现双链表，其中，额外重构<方法，用last_access_time来决定每个UnifiedTreeNode的大小关系；
```py title="UnifiedTreeNode"
class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        self.write_through_pending_id: Optional[int] = None
```
随后查看双链表UnifiedLRUList的具体结构，其核心创建self.head和self.tail，利用UnifedTreeNode中的lru_next和lru_prev来实现双链表，其中，采用经典的哈希表+双链表架构，通过hash table记录每个node.id所对应的UnfiedTreeNode在双链表中的位置，实现O(1)的原语；
此外，专门新建\_OngoingWriteThrough, \_OngoingLoadBack和\_OngoingPrefetch类来记录目前正处于的特定状态；
```py title=""
class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node: UnifiedTreeNode
    lock_params: Optional[DecLockRefParams]
    publish_nodes: list[UnifiedTreeNode]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node: UnifiedTreeNode
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node: UnifiedTreeNode
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams
    comp_xfers: dict[ComponentType, list[PoolTransfer]]
```
随后来查看目前实现的UnifiedRadixCache的具体实现，其中，该class仍然继承KVCacheEventMixin, BasePrefixCache，就像RadixCache一样；其中第一个值得注意的函数为`init_hicache`函数，其中，若storage_backend不为None，则会调用**HybridCacheController**.parse_storage_backend_extra_config，获取storage backend相关的config，随后调用`attach_hybrd_pool_to_unified_cache(...)`函数，其中HybridCacheController的具体信息不在该文档赘述；该函数实际上是告诉UnifiedRadixCache他到底要创建多少个device pool/host pool，具体做如下的几件事情
* 策略选择:\_select\_strategy 按 (kvcache 类型, 组件集合) 从 \_STRATEGIES 列表里挑出匹配的装配策略——\_MambaStrategy、\_SwaStrategy、\_DsaStrategy、\_DeepSeekV4Strategy、\_MiniMaxSparseStrategy,兜底是 \_PlainKvStrategy(纯 full-attn 模型)。每个策略知道该建哪些 host pool、层映射怎么算、哪些池要 evict 回调。还有 register_stack_strategy 留给下游 fork 插入自定义组合。
* 构建整个栈:策略的 build() 造出各池的 host pool → 包成 PoolEntry → 组成 HostPoolGroup → 用它构造 HybridCacheController。
* 回填接线(\_apply_stack_result):把 cache_controller、host_pool_group 挂到 cache 上;把每个组件自己的 host pool 塞给对应的 TreeComponent(cache.components[ct] 上 setattr),这样组件的 build_hicache_transfers 钩子才能生成自己那份 PoolTransfer;注册 sidecar 池;把 layer_done_counter 注册到 kvcache 上,让逐层加载能和计算重叠。
其中核心关注和原先RadixCache相关的几个核心函数，==如match_prefix, insert, evict等等==，在match_prefix函数中，尝试调用`StreamingSession.try_match_prefix(...)`，若得到result直接返回 （StreamingSession保存上一轮的last node, req_to_pool idx等等，这样下一轮可以直接O(1)找到对应的node，不需要O(k)匹配），不然则和radixCache一样，通过调用_match_prefix_helper获取具体的value，具体的`_match_prefix_helper`实现分为如下几步：
1. 创建validators，用于判断该node是否合法，如对于FullKv,那么对于match_device_only，需要其value (device_indices)非None才行；
2. 从树节点开始往leaves节点便利，核心通过调用`_update_best_if_valid(...)`记录对应的best_match_node和best_match_device_node，best_match_node可能会比后者更深，原因是他可以记录一个node虽然被evicted但是存在host中；
其中，假设出现prefix_len小于child.key情况，代表着partial匹配了，此时需要split node，具体的`_split_node`函数和先前的RadixCache的有较大差别，具体代码如下：可以看到的时，和原先的split_Node类似，会创建一个新UnifiedTreeNode，继承children的key，将child作为node的parent存在，split node hash 等等，在其中，额外调用`_for_each_component_lru(...)`函数，将child node从双链表中删除，随后将new_node和child插入双链表头；并调用`_update_evictable_leaf_sets(...)`将new_node和child加入evictable set中 (满足无lock_ref且为叶子结点)，其中，还额外对每个component调用`redistribute_on_node_split(...)`函数，用于根据不同的componentType更新parent和child的value和host_value；
可以看到的是UnifiedRadixCache在split_node时，核心只做两件核心的事情，node.value需要针对不同的component进行处理、需要将child和node在双链表中更新；
```py title="_split_node"
def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components, priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.creation_time = child.creation_time

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node
```
在`_match_prefix_helper(....)`后，调用`_match_post_processor(...)`函数，其中，该函数代码如下：其中，该函数做了如下的操作：
* 调用`refresh_lru(...)`，将实际访问过的节点放到双链表表头，为每个节点更新last_access_time;
* 更新last_host_node和device_indices；
* 调用`finalize_match_result(..)`来处理结果，其中对于full attention，计算 host_hit_length——从 best_match_node 向上走到 last_device_node,累加沿途被驱逐节点的 host_value 长度。这个数就是"要从 host 搬回 device 的 token 数",scheduler 据此决定是否发起 load-back；对于SWA直接跳过，对于mamba，则做额外copy-on-write，此处不做额外展开；
```py title="_match_post_processor"
def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
    ) -> MatchResult:
        node_update = best_match_node
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            comp.refresh_lru(LRURefreshPhase.MATCH_END, node_update, self.root_node)

        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        # last_host_node will be used as the starting node for the subsequent
        # `prefetch_from_storage` flow. We directly use best_match_node here,
        # because best_match_node represents the node where all components
        # have reached consensus on both device & host availability.
        last_host_node = (
            best_match_node
            if self.cache_controller is not None
            else best_match_device_node
        )

        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
        )

        for component in self._components_tuple:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_match_device_value_len,
            )
        return result
```
随后来看insert的实现，其中，核心调用_insert_helper(...)函数进行插入操作，具体函数代码如下，具体整个`_insert_helper(...)`核心分为如下几步：
* 更新node的priority，调用`_touch_node(...)`，更新节点的last_access_time，并调用component的refresh_lru，将访问的节点重新放到链表头；
* 从root_node不断往下进行便利，如果prefix_len < node.key，那就split_node，同时，不断更新node的优先级；
* 若node不在device上，即evicted为True，则调用`_unevict_node_on_insert(...)`，将该evicted node的full device value从fresh kv indices中恢复，随后针对每个component调用`recover_after_unevict(...)`负责顺带更新SWA组件信息，因为validator要求所有的组件都通过才代表恢复，其中对于SWA，判断目前的插入node的prefix_len和params.swa_evicted_seqlen的关系，确定是否在窗口内来确定是否恢复；若node在device中，则针对每个component调用`update_component_on_insert_overlap(...)`，主要针对的场景是，full kv在device中，但是SWA cache不在的场景，用来判断是否需要更新窗口内的KV，并返回start_idx作为comp_consumed_from，用来确定可以evict哪段KV；
* 调用`inc_hit_count(...)`判断是否hit_count到达了threhold，来触发write_back(...)；
* 调用`_add_new_node(...)`来讲新节点放入树种，并调用`commit_insert_component_data(...)`来处理特殊的SWA/mamba状态；
* 调用INSERT_END的`refresh_lru(...)`依从性更新所有相关的节点，例如对于Full KV则是从最target_node一直走到root_node，对于SWA则是只统计窗口内的KV node；
```py title="Insert_helper"
def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(node)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)
            node.priority = max(node.priority, priority)

            if node.evicted:
                self._unevict_node_on_insert(node, value[:prefix_len])
                # FULL was restored from the request's fresh KV. Aux
                # components (e.g. SWA) may still hold tombstones and need
                # to rebuild their value from the same slice.
                for component in self._components_tuple:
                    if component.component_type == BASE_COMPONENT_TYPE:
                        continue
                    component.recover_after_unevict(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        params=params,
                    )
            else:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                # Let each component claim ownership of overlapping KV slots
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)

                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

            self._inc_hit_count(node, params.chunked)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        is_new_leaf = False
        # Create new leaf for remaining suffix. A leaf survives on its Full
        # value alone; auxiliary components (SWA, Mamba) may legitimately hold
        # only a tombstone for this span (e.g. the whole leaf is outside the SWA
        # window). Materialize it anyway so the Full KV stays cacheable.
        if len(key):
            target_node = self._add_new_node(node, key, value, priority=priority)
            is_new_leaf = True
        else:
            target_node = node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )

        if target_node is not self.root_node:
            for component in self._components_tuple:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, target_node, self.root_node
                )

        if is_new_leaf:
            self._inc_hit_count(target_node, params.chunked)
        return result
```
随后查看evict的具体逻辑，具体函数如下，其中，首先创建tracker记录每个不同的tree component的访问次数，随后，针对不同的tree_component，调用drive_evcition函数，其中`drive_eviction(...)`分别从不同的component池中evict来释放空间，EvictParams 里分别带 num_tokens / swa_num_tokens / mamba_num，其中，full的evict仍然是经典的从evictable_leaves中不断按照优先级选取leaf进行evict，调用`_evict_device_leaf(...)`函数，该函数核心做如下操作：
1. Full：若没备份，判断是否是backup，若是，先write_backup，writing_check没问题后，在调用_evict_to_host(...)，若非backup，则直接调用`_evict_component_and_detach_lru(...)`函数，对于Full而言，该函数不做任何操作；
2. SWA/mamba：相比于Full，调用_evict_device_leaf(...)的时机完全不一样，扫描的顺序是从LRU的尾部开始往前扫，注意，只对叶子结点调用_evict_device_leaf(...)，对于非叶子结点，直接调用`_evict_component_and_detach_lru(...)`函数时，其中调用`evict_component(...)`释放自己的池子槽位，后续会调用`_iteratively_delete_tombstone_leaf(...)`来递归删除墓碑节点，
```py title="evict"
def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )
```
其中详细解释下不同的tree_component的`drive_eviction`，函数，对于Full而言，非常简单，按照优先级拿到不同的leaves，随后，直接调用不同的_evict_device_leaf(...)函数；而对于SWA/mamba则不一样，按照LRUlist的顺序进行evict，核心参考last_access_time，而且insert插入时，采用mru方式，会保证链表按照child->parent->grand_parent的顺序在双链表中，因而在evict从后向前时，会优先evict parent节点的SWA kv，随后再是child，这也符合SWA的算法特性，最近窗口的KV才有用，因此优先evict parent的SWA KV，然而，这显然没考虑到session_ref，evict时应该按照session_ref来evict，而不是无脑mru，可以作为后续SessionUnifiedRadixCacheMixin的设计原则；当然设计的时候，应当为每个不同的component额外维护自己的ref，在不同的component分别实现各自的session计数维护方法，并在SessionUnfiedradixCacheMixin中封装调用，这样在UnifiedRadixCache层面所有session相关的函数能优先包装到mixin中；
其中`_evict_component`实际上替换的是调用不同的kv_pool的free来释放不同的KV space，把不同的LRU list中的node给删除，这就是_evict_component_and_detach_lru;
```py title="\_evict_component_and_detach_lru"
def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> tuple[int, int]:
        device_freed, host_freed = comp.evict_component(node, target=target)
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed

        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed
```
在evict_device_leaf中，如果是write_through模式，还会调用`_iteratively_delete_tombstone_leaf(...)`来递归删除节点，其中的核心逻辑是：
* 若改node的父亲还有任何的部分还咋被使用，则不允许evict；
* 随后判断该节点的父亲的device是否在HBM中，如果在的话，直接break，不然证明该parent的full已经被evict，那就把别的component也给evict了，如果host_value也不在了，把host伤的别的component也都evict了，此时证明该node已经没用了，将其中各个component都分别调用`_evict_component_and_detach_lru(...)`来进行evict component，随后，如果device和host的full KV都不在了，直接调用`_remove_leaf_from_parent(...)`，将其从树中删除，否则，保留该节点，这也为后续出现可能一个node不存在device value，但是存在SWA/Mamba提供可能性；
其中关注下_evict_to_host(...)函数，在write_through策略下，假设node节点被backup了，则会走这个函数，具体的代码如下，
```py title="evict_to_host"
def _evict_to_host(
        self, node: UnifiedTreeNode, tracker: Optional[dict[ComponentType, int]] = None
    ) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        assert not node.evicted and node.backuped
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        self._cascade_evict(node, trigger, tracker)
        self._record_remove_event(node, medium=StorageMedium.GPU)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)
```
下面详细查看下`_cascade_evict(...)`的具体定义，其中该函数，会根据每个节点的eviction_priority来，如果是非叶子结点，直接按照Full > SWA > mamba的顺序，假设某个KV被evict，低于其优先级的KV会被一起evict；对于叶子结点，由于需要删除整个叶子结点，因此如果trigger是SWA，只要Full没锁，也会级联evict full；注意，在级联的最后，判断如果component_type是Full，再把value置为None，因为evict SWA的时候需要Full；

