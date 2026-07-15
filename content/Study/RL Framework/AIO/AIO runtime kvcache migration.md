虽然全异步能够解决skewness bubble，然而随着训练的进行，rollout worker间的请求高低优先级特性可能发生改变，部分rollout worker上的高优先请求可能由于kvcache压力而被迫排队，部分rollout worker上可能都是低优先级的kvcache，因此，需要runtime的在rollout worker间迁移高优先请求以及其对应计算的kvcache，本文详解具体的实现；
# Sglang runtime KVCache migration
首先在server_args中新增--enable-kv-migration，来判断是否允许开启kvcache迁移；
```py title=""
    # KV cache migration over HTTP (peer-to-peer between sglang servers)
    enable_kv_migration: bool = False
    kv_migration_watchdog_timeout: float = 60.0
```
随后新增和runtime kvcache迁移相关的请求的ReqInput和ReqOutput
```py title=""
@dataclass
class GetTransferSessionInfoReqInput(BaseReq):
    pass

@dataclass
class GetTransferSessionInfoReqOutput(BaseReq):
    success: bool
    tp_rank: int = -1
    pp_rank: int = -1
    session_id: str = ""
    host_kv_data_ptrs: List[int] = field(default_factory=list)
    host_kv_item_lens: List[int] = field(default_factory=list)
    page_size: int = 0
    message: str = ""

@dataclass
class GetRequestExtraTokenSizeReqInput(BaseReq):
    input_ids: List[int]
    extra_key: Optional[str] = None

@dataclass
class GetRequestExtraTokenSizeReqOutput(BaseReq):
    success: bool
    extra_token_size: int = 0
    matched_token_size: int = 0
    total_token_size: int = 0
    message: str = ""

@dataclass
class AllocateTokenForTransferReqInput(BaseReq):
    input_ids: List[int]
    extra_key: Optional[str] = None
    extra_token_size: int = 0
    migration_id: Optional[str] = None  # set by HTTP layer to share id across ranks

@dataclass
class AllocateTokenForTransferReqOutput(BaseReq):
    success: bool
    migration_id: str = ""
    tp_rank: int = -1
    pp_rank: int = -1
    kv_indices: List[int] = field(default_factory=list)
    message: str = ""

@dataclass
class TransferRequestKVCacheTarget:
    tp: int
    pp: int
    session_id: str
    host_kv_data_ptrs: List[int]
    host_kv_item_lens: List[int]
    kv_indices: List[int]

@dataclass
class TransferRequestKVCacheReqInput(BaseReq):
    input_ids: List[int]
    extra_key: Optional[str] = None
    matched_token_size: int = 0
    extra_token_size: int = 0
    target_per_rank: List[TransferRequestKVCacheTarget] = field(default_factory=list)

@dataclass
class TransferRequestKVCacheReqOutput(BaseReq):
    success: bool
    message: str = ""

@dataclass
class CommitTransferRequestKVCacheReqInput(BaseReq):
    migration_id: str = ""

@dataclass
class CommitTransferRequestKVCacheReqOutput(BaseReq):
    success: bool
    matched_after_commit: int = 0
    message: str = ""
```
随后在python/sglang/srt/kv_migration/io_types.py中新增迁移相关的class类，新增类PendingMigration和TransferTarge，分别指示实际inflight的migration操作，放在target端，用于await/commit或者watchdog后释放，TransferTarget代表peer rank的metadata，包括tp/pp信息，session id，分配的host_kv_data_ptrs和kv_indices；
```py title=""
@dataclass
class PendingMigration:
    """In-flight migration on the target side, awaiting /commit or watchdog."""

    input_ids: List[int]
    extra_key: Optional[str]
    full_key: "RadixKey"
    matched_aligned: int
    matched_node: "TreeNode"
    host_locked_nodes: List["TreeNode"]
    host_tail_indices: "torch.Tensor"
    created_at: float = field(default_factory=time.monotonic)
    
@dataclass
class TransferTarget:
    """One peer rank's metadata for /transfer_request_kvcache."""

    tp: int
    pp: int
    session_id: str
    host_kv_data_ptrs: List[int]
    host_kv_item_lens: List[int]
    kv_indices: List[int]
```
随后再来看定义在srt/kv_migration/tree_helpers.py中定义的一些函数，首先是collect_host_pages函数，其中，从root_node开始，不断的匹配key，将中间的所有page添加到pages列表中，并把沿途的TreeNode放入path_nodes中，其中，如果遍历到某个node时，发现