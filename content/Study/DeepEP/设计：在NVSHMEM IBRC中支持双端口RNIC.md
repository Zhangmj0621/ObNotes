# 问题

DeepEP使用NVSHMEM在GPU之间传输数据，使用智谱的平湖集群测试DeepEP的过程中，我们发现，DeepEP的性能不符合预期（[DeepEP测试对比分析-H100](https://infrawaves.feishu.cn/wiki/JnElw8FYhiLABikTJrMcPr7snph)），大概只有单口RNIC的一半。观察交换机的流量，每张RNIC只使用了一个端口。

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MjM1MTBlZDZlMGE3OWYyNTJjMjc0MzJjOTEzZjExNTVfaUFMWjExeHBvcWxKS3hlT2puOU9Cbk81Yks4NTB5RkxfVG9rZW46UEJubmJPMldwb1NKNkp4dGZrVWMzYm0zbmdoXzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)

# 原因

NVSHMEM使用的缺省transportation机制IBRC不支持PE选择多个端口，建立QP连接，所以，在双端口RNIC的场景中，只使用了一个端口。

# 方案

有两个可选的方案解决这个问题：

使用NVSHMEM的IBGDA替代缺省的IBRC机制，但是，在双端口RNIC环境中，IBGDA初始化失败，原因是mlx5dv_devx_umem_reg注册GPU内存时，不成功。短时间内，这个问题，不一定有解。即使这个问题解决了，后续还存在很大的不确定性。

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmRlY2ZjZTlmY2E3NTE3ZGZjOWRiYWRiMzA2NWJhOTBfcmJ5ZkZqVFdBRUhIWFhRUWpIMTllTVQzWEdsRGYzVW1fVG9rZW46UlFSNGIwcUdEb2FWOEd4YzM3V2N3NjliblRkXzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)

所以，我们只能选择第二种方案，即在IBRC中支持双端口RNIC，本质上，PE可以选择两个端口建立QP相互通信。

## 选网卡-已确认信息

按照距离最近的原则，每个PE/GPU选取最近的RNIC的两个端口，代码：

host/transport/transport.cc:nvshmemi_setup_connections(...)

module/topo/topo.cc:nvshmemi_get_devices_by_distance(...)

对于PIX的情况，PE可以正确地选到一个RNIC上的两个端口。

对于PXB的情况，PE可以正确地选到一个RNIC上的四个端口。

## QP建链-已完成

module/transport/ibrc/ibrc.cc:nvshmemt_ibrc_connect_endpoints(...)中

将transport_ibrc_state_t中的ep(qp)数量double

根据实际的num_selected_devs数量，修改ep_handles与local_ep_handles的大小

在当前bootstap连接下，交换双端口ep的信息，并建立RDMA链接。

目前进展：成功在多端口上创建qp，并利用一次bootstrap通信交换双端口ep信息，已经验证

## 内存注册调整为多端口-已完成

为了适配一卡两口环境，对应的内存也需要重新进行注册，原先注册的内存在**handles_**结构中，具体注册的位置为ibrc.cpp:nvshmemt_ibrc_get_mem_handle(...)，函数调用链为mem_transport.cpp:register_mem_handle(...) -> mem_heap.cpp:register_heap_chunk(...)，我们期望将handles_从二维转成三维，进而能够储存多端口的内存注册信息。

```c
class nvshmemi_symmetric_heap {
    //... ...
  - std::vector<std::vector<nvshmem_mem_handle>> handles_;
  + std::vector<std::vector<nvshmem_mem_handle>> handles_[2];

```

与handles_直接相关需要修改的函数接口为(部分需要修改接口参数，涉及上层调用)：

mem_heap.cpp

- `void nvshmemi_symmetric_heap::update_idx_in_handle(...)`
    
    未修改接口，仅修改函数体
    
- `nvshmemi_symmetric_heap::~nvshmemi_symmetric_heap(...)`
    
    未修改接口，仅修改函数体
    
- `int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::cleanup_symmetric_heap(...)`
    
    未修改接口，仅修改函数体
    
- `int nvshmemi_symmetric_heap_static::cleanup_symmetric_heap(...)`
    
    未修改接口，仅修改函数体
    
- `int nvshmemi_symmetric_heap_vidmem_static_pinned::map_heap_chunk(...)`
    
    未修改接口，仅修改函数体
    
- `int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::map_heap_chunk(...)`
    
    未修改接口，仅修改函数体
    
- `int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::exchange_heap_memory_handle(...)`
    
    对P2P连接的pe对，调用该函数，在进程间交换handles信息，我们将接口修改为同时传入多端口的handles信息(目前两个)，以期在不增加socket通信负载下，同时进行多端口的信息交换。
    
    接口修改后如下：
    
    `int exchange_heap_memory_handle(nvshmem_mem_handle_t *local_handles, nvshmem_mem_handle_t *local_handles_2);`
    
    在int nvshmemi_symmetric_heap_dynamic::register_heap_chunk(...)中，同时发送两个端口的handles信息给其余pe
    
    ```c
    NVSHMEMI_IPC_CHECK(
                    ipcSendFd(myIpcHandle, *(int *)local_handles, pid, receiving_process));
    NVSHMEMI_IPC_CHECK(
                    ipcSendFd(myIpcHandle, *(int *)local_handles_2, pid, receiving_process));
    
    // ... ...
    
    NVSHMEMI_IPC_CHECK(ipcRecvFd(
                    recvIpcHandles[sending_process],
                    (int *)&handles_[0].back()[it1->second * state->num_initialized_transports]));
    NVSHMEMI_IPC_CHECK(ipcRecvFd(
                    recvIpcHandles[sending_process],
                    (int *)&handles_[1].back()[it1->second * state->num_initialized_transports]));
    
    ```
    
- `int nvshmemi_symmetric_heap_sysmem_static_shm::register_heap_memory_handle(...)`
    
    未注册两口内存，因而增加函数参数，修改后的函数接口如下：
    
    int nvshmemi_symmetric_heap_sysmem_static_shm::register_heap_memory_handle(..., int index)
    
    在函数中，利用index调用register_mem_handle(...)选择具体的端口进行内存注册
    
- `int nvshmemi_symmetric_heap_vidmem_static_pinned::register_heap_memory_handle(...)`
    
    同上
    
- `int nvshmemi_symmetric_heap_static::register_heap_chunk(...)`
    
    同下，唯一区别在于，相比nvshmemi_symmetric_heap_dynamic，该类函数中，存在cache的概念，因而当访问cache存在时，会直接从cache中读取handle而不需要再次注册
    
    ```c
    if (NVSHMEMI_TRANSPORT_IS_CAP(current, state->mype, NVSHMEM_TRANSPORT_CAP_MAP)) {
        map_handles = handles_[0].front().data();
        local_handles[0][i] = map_handles[i];
      + map_handles = handles_[1].front().data();
      + local_handles[1][i] = map_handles[i];
    }
    
    ```
    
- `int nvshmemi_symmetric_heap_dynamic::register_heap_chunk(...)`
    
    1）为注册两口内存，需要调用两次remotetran.register_mem_handle(...)，对相同的buff用不同的端口进行内存注册，因此，需要进一步修改函数接口
    
    - int nvshmemi_mem_remote_transport::register_mem_handle(..., int index)
        - nvshmemt_ibdevx_get_mem_handle(..., int index)
        - nvshmemt_ibgda_get_mem_handle(..., int index)
        - nvshmemt_libfabric_get_mem_handle(..., int index)
        - nvshmemt_ucx_get_mem_handle(..., int index)
    - ~~buffer_register(...)：未被调用，未来修改~~
    
    **注**：调用register_mem_handle(...)函数位置有多处，但由于我们实际执行时，并未涉及所有函数，因此只在register_heap_chunk(...)中调用时区分index，**其余时候均设置Index = 0**。
    
    2）在register_heap_chunk(...)中，将local_handles数据结构从一维变为二维以存放多端口内存句柄，随后注册内存，并利用两次all_gather进行交换信息(**为什么不一次？如果强行改为一次通信，则存放handles的数据结构会变得不连续，接口改动过于大**）
    
    ```c
    // Export memory for both ports
    status =
        export_memory((nvshmem_mem_handle_t *)(local_handles[0] + idx), mem_handle_in);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "export_memory failed for p2p on heap dynamic \\n");
    status =
        export_memory((nvshmem_mem_handle_t *)(local_handles[1] + idx), mem_handle_in);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "export_memory failed for p2p on heap dynamic \\n");
    
    // ... ...
    
    // register memory on both ports
    status = remotetran.register_mem_handle(&local_handles[0][0], idx, mem_handle_in, buf,
                                            size, current, 0);
    status = remotetran.register_mem_handle(&local_handles[1][0], idx, mem_handle_in, buf,
                                            size, current, 1);
    
    // ... ...
    // Allgather memory handle for remote connected PEs
    handles_[0].push_back(
        vector<nvshmem_mem_handle_t>(state->num_initialized_transports * state->npes));
    handles_[1].push_back(
        vector<nvshmem_mem_handle_t>(state->num_initialized_transports * state->npes));
    
    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles[0], (void *)(handles_[0].back().data()),
        sizeof(nvshmem_mem_handle_t) * state->num_initialized_transports, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed \\n");
    status = nvshmemi_boot_handle.allgather(
        (void *)local_handles[1], (void *)(handles_[1].back().data()),
        sizeof(nvshmem_mem_handle_t) * state->num_initialized_transports, &nvshmemi_boot_handle);
    
    status = remotetran.gather_mem_handles(*(dynamic_cast<nvshmemi_symmetric_heap *>(this)),
                                           physical_heap_size_, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of mem handles failed for remotetransport\\n");
    
    // Exchange send/recv memory handles for p2p connected PEs
    exchange_heap_memory_handle(&local_handles[0][0], &local_handles[1][0]);
    heap_mspace_->add_new_chunk((char *)heap_base_ + physical_heap_size_, size);
    
    ```

**注**：gather_mem_handles(...)只与ibgda相关，因此暂未做修改

mem_transport.cpp

- `void nvshmemi_mem_p2p_transport::print_mem_handle(...)`
    
    仅作打印，暂未修改
    
- `int nvshmemi_mem_remote_transport::gather_mem_handles(...)`
    
    **只与ibgda相关，暂未修改**

nvshmemi_symmetric_heap.hpp

- `inline nvshmem_mem_handle *nvshmemi_symmetric_heap::get_transport_mem_handle(...)`
    
    修改函数接口为：
    
    inline nvshmem_mem_handle *nvshmemi_symmetric_heap::get_transport_mem_handle(...,
    
    int index)
    
    并同步修改上层调用接口nvshmemi_get_remote_mem_handle(...)与nvshmemi_get_local_mem_handle(...)如下
    
    - static inline void nvshmemi_get_local_mem_handle(..., int index)
    - static inline void nvshmemi_get_remote_mem_handle(..., int index)

## 收发 - 已完成

数据收发时，流量平均分布在两个端口的QP上。

在ibrc中，收发主要依赖int nvshmemt_ibrc_rma(...)与int nvshmemt_ibrc_amo(...)两个函数实现，rma用来处理PUT和GET操作，实现RDMA WRITE和READ，amo则主要实现infiniband的原子操作，在其中，传入函数的参数有在上一节内存注册中提到的handle，为了能够同时利用多端口进行收发，我们期望同时传入两个端口的handle信息，因此，我们进一步修改了相关数据结构amo_memdesc_t与rma_memdesc_t，修改后的数据结构如下：

```c
typedef struct rma_memdesc {
    void *ptr;
    uint64_t offset;
    nvshmem_mem_handle_t *handle[2];
} rma_memdesc_t;

typedef struct amo_memdesc {
    rma_memdesc_t remote_memdesc;
    uint64_t retflag;
    void *retptr;
    void *valptr;
    void *cmpptr;
    uint64_t val;
    uint64_t cmp;
    nvshmem_mem_handle_t *ret_handle[2];
} amo_memdesc_t;

```

随后我们在所有调用nvshmemi_get_local_mem_handle(...)与nvshmemi_get_remote_mem_handle(...)获取handle的位置，改为获取双端口的handle信息，将他们存放入同一个修改后的结构体中，在ibrc的int nvshmemt_ibrc_rma(...)与int nvshmemt_ibrc_amo(...)里，可以选择该块数据具体利用哪个端口进行发送，实现流量的平均分布。

```c
    size_t remain_bytes = bytesdesc.nelems * bytesdesc.elembytes;
    size_t send_bytes = 0;

    for (int index = 0; index < ibrc_state->ndevs; index++) {
        if (is_proxy) {
            ep = ibrc_state->ep[index][(ibrc_state->ep_count * pe + ibrc_state->proxy_ep_idx)];
            ep2 = ibrc_state->ep[index ^ 1][(ibrc_state->ep_count * pe + ibrc_state->proxy_ep_idx)];
        }
        else {
            ep = ibrc_state->ep[index][(ibrc_state->ep_count * pe)];
            ep2 = ibrc_state->ep[index ^ 1][(ibrc_state->ep_count * pe)];
        }

        status = check_poll_avail(ep, WAIT_ANY);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \\n");

        op_id = ep->head_op_id & IBRC_REQUEST_QUEUE_MASK; // ep->head_op_id % ibrc_qp_depth

        sr = &(ep->req + op_id)->sr;
        bad_sr = &(ep->req + op_id)->bad_sr;
        sge = &(ep->req + op_id)->sge;

        memset(sr, 0, sizeof(ibv_send_wr));

        sr->next = NULL;
        sr->send_flags = IBV_SEND_SIGNALED;
        sr->wr_id = NVSHMEMI_OP_PUT;
        sr->num_sge = 1;
        sr->sg_list = sge;

        sr->wr.rdma.remote_addr = (uint64_t)remote->ptr + send_bytes;
        assert(remote->handle[index]);
        sr->wr.rdma.rkey = ((struct nvshmemt_ib_common_mem_handle *)remote->handle[index])->rkey;
        if (index != ibrc_state->ndevs - 1)
            sge->length = (bytesdesc.nelems * bytesdesc.elembytes) / ibrc_state->ndevs;
        else
            sge->length = remain_bytes;
        sge->addr = (uintptr_t)local->ptr + send_bytes;
        remain_bytes -= sge->length;
        send_bytes += sge->length;

```

目前已经测试了，分别单独利用两个端口收发时的能力，剩余工作为，同时利用两个端口进行收发。

目前存在问题，观察到，当rma利用端口1发送时，amo必须也利用端口1发送，否则会出现hang的情况，反之亦然，目前在尝试定位两个操作之间的依赖关系。

hang的问题目前已解决，可能的原因为，当原先在单端口发送时，rma和amo下发的wr存在单个qp中，执行顺序保序，而当wr分布到两个端口上后，可能存在，**先于AMO的RMA操作反而晚于AMO操作执行的情况**，因此，在AMO中，增加wait逻辑，等待另一侧端口上已经下发的wr全部完成后，才下发该amo操作wr，此时，hang解决（暂时不理解为什么amo操作需要保证，在其之前的rma需要先于他完成，看deepEP代码没看出关联性）

```c
status = check_poll_avail(ep, WAIT_ANY);
NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \\n");
if (ibrc_state->ndevs >= 2) status = check_poll_avail(ep2, WAIT_ALL);
NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \\n");

```

## PXB支持16个端口收发 - 已完成

为了使得PXB也能够同时利用所有端口进行收发，我们在选择网卡时，对于奇数号的GPU，我们将其的前四个选择的select_dev_id转置，因此，他们会选择共享的四张网卡的后两张，而偶数GPU则继续使用原先的两个端口，此时，流量会均匀的分布在四个端口上。

```sql
if ((gpu_device_id % 2 == 1) && (max_dev_per_pe >= 16)) {
        int left = 0;
        int right = 3;
        while (left < right) {
            int temp = device_arr[left];
            device_arr[left] = device_arr[right];
            device_arr[right] = temp;
            left++;
            right--;
        }
    }

```

# 实现

nic select debug打印：

PXB的情况：

```bash
/workspace/infra/nvshmem_3.1.7/src/modules/transport/ibrc/ibrc.cpp 1457 Select RNIC 13: n_pes: 2, my_pe:1, index:1
/workspace/infra/nvshmem_3.1.7/src/modules/transport/ibrc/ibrc.cpp 1457 Select RNIC 14: n_pes: 2, my_pe:1, index:1
/workspace/infra/nvshmem_3.1.7/src/modules/transport/ibrc/ibrc.cpp 1457 Select RNIC 15: n_pes: 2, my_pe:1, index:1
/workspace/infra/nvshmem_3.1.7/src/modules/transport/ibrc/ibrc.cpp 1457 Select RNIC 16: n_pes: 2, my_pe:1, index:1

```

PIX的情况：

```bash
/workspace/infra/nvshmem_3.1.7/src/modules/transport/ibrc/ibrc.cpp 1457 Select RNIC 14: n_pes: 2, my_pe:0, index:1
/workspace/infra/nvshmem_3.1.7/src/modules/transport/ibrc/ibrc.cpp 1457 Select RNIC 15: n_pes: 2, my_pe:0, index:1

```

# 功能测试

### 两机PIX

原生：

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=NzkwMDQyNjY0NTE2OTYwZjQxMTg4ZDhmZjJiNmFhNTBfYktoTkJKenY5eTJwUXRLN1BVT3A3MDhZMktIQ3ZwM0hfVG9rZW46SEdsZ2I0R3VVb05jMGJ4am01amMycDBPbk5mXzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)

RDMA: 44.01 NVL: 143.95

双端口收发：

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=NGY5NzRkMTFmODk1Yjk4YTY1MDg5NTE5MmFlODk4NDBfMXNUNUs5RGx3M2Nza2t5cTU1VmtxUEM1VVJNQkNZSVBfVG9rZW46T2xRVGJodENhb1Jjd0p4UkNzbGNQYmRnbk15XzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)

RDMA: 51.33 NVL: 167.92

### 两机PXB

原生：

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=NTE2NjU2NGQ2Yzk2ZDkzMTYzOWZiZWVkOTI2Yjg1NDVfYXl6bEhrcU9waHMyb3RlMjBBWXJ5MThwNWp5QWFRcXRfVG9rZW46SG9rdmJUaTdqbzJITG94VERHRWNPTGh2bnlkXzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)

RDMA: 23.31 NVL: 76.25

双端口收发：

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=NTI0ODYzMjZmZjdkNWY2YWUxNDI3N2Y1ZTMxY2VmMjZfZVFKeFNUWDhaUmpxQU1NSXRUREtNU0FQQkI3bEZCMHlfVG9rZW46QUp5NWJGMlUwb1VVTm14akRBdWNROE5sbnhlXzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)

RDMA:52.25 NVL:170.91

# 对比测试

## 两机 node501（PXB）node510（PXB）

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 24, RDMA chunk 32: 39.28 GB/s (RDMA), 128.21 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 20, RDMA chunk 32: **52.26 GB/s (RDMA), 170.59 GB/s (NVL)**

[tuning] Best combine: SMs 24, NVL chunk 3, RDMA chunk 32: 44.13 GB/s (RDMA), 144.05 GB/s (NVL)

## 两机 node506（PIX） node507（PIX）

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 32, RDMA chunk 32: 34.16 GB/s (RDMA), 111.76 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 12, RDMA chunk 32: **47.97 GB/s (RDMA), 156.94 GB/s (NVL)**

[tuning] Best combine: SMs 24, NVL chunk 2, RDMA chunk 32: 45.48 GB/s (RDMA), 148.77 GB/s (NVL)

## 两机 node501（PXB） node507（PIX）

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 20, RDMA chunk 32: 35.48 GB/s (RDMA), 115.81 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 24, RDMA chunk 32: **49.27 GB/s (RDMA), 160.81 GB/s (NVL)**

[tuning] Best combine: SMs 24, NVL chunk 2, RDMA chunk 32: 43.67 GB/s (RDMA), 142.54 GB/s (NVL)

## 两机 node501（PXB） node506（PIX）

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 20, RDMA chunk 32: 35.25 GB/s (RDMA), 115.07 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 20, RDMA chunk 32: **49.10 GB/s (RDMA), 160.25 GB/s (NVL)**

[tuning] Best combine: SMs 24, NVL chunk 3, RDMA chunk 32: 43.63 GB/s (RDMA), 142.40 GB/s (NVL)

## 四机

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 16, RDMA chunk 20: 21.01 GB/s (RDMA), 42.19 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 12, RDMA chunk 12: 20.47 GB/s (RDMA), 41.11 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 2, RDMA chunk 12: 20.02 GB/s (RDMA), 40.20 GB/s (NVL）

# 参考资料

启用双端口发送后交换机流量均匀分布端口

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWQ0NWVkNWMzNjI0ZGY1ZGQ1NGE5NTlmNjEzYzlmZTlfNWo3MXNFZmJjU2ROaTVJemtibFZWdHlsWVdYVkpwczlfVG9rZW46VEFsVmJiQkNYb0s2SUF4S1dQUGNLUDUybkllXzE3NDk4MjA5ODI6MTc0OTgyNDU4Ml9WNA)