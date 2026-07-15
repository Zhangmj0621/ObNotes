## Background

To support dual-ports in nvshmem, we initially create same number qps in both ports and send data in sequence using different qp. However, due to the atomic operaion(use to function as sychonization) in DeepEP, we need to ensure all the rma opeartion(rdma put & get) should complete before the amo operation. When we split data into two qp in different ports, the atomicity is broken. We need to add explict wait before the amo wr posted to guarantee the completion of rma operation, which brings additional overhead. So we plan to restructure our design in a high-level dimension. After consideration, we try to assign channels to different ports to achieve load balancing.

## Interfaces

We design to split channel into two groups by its channel_id. When its channel_id is even, we use first ports to transport. On the other condition, we use the second ports. So we change some interfaces in order to pass channel_id to nvshmem.

### rma

In DeepEP:csrc/kernels/internode.cu, we do rma operation like below in Dispatch.

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OTY2Y2FhZGU5ZjczOGZhNDY1NTFjMjRmMTA2MjAxYmRfN3k2UTduaTBralFiOTFzNXpGUU1VMWphMjI5Rk02OExfVG9rZW46VEIxbWJHUTR3b1lSZGZ4clpuamMzcEdhbm5mXzE3NDk4MjA5NTE6MTc0OTgyNDU1MV9WNA)

`nvshmemx_int8_put_nbi_warp` is defined in nvshmem. It is implemented by the function called `nvshmemi_put_nbi_threadgroup`. Its declaration and definition is as follows:

```
// declaration
#define NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP(NAME, TYPE)                                         \\
    __device__ void nvshmemx_##NAME##_put_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems,  \\
                                                   int pe, int channel_id);                        \\
    __device__ void nvshmemx_##NAME##_put_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, \\
                                                    int pe, int channel_id);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP

// definition
#define NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, Group)                                      \\
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_##Name##_put_nbi_##Group( \\
        Type *dest, const Type *source, size_t nelems, int pe, int channelId) {                  \\
        nvshmemi_put_nbi_threadgroup<Type, nvshmemi_threadgroup_##Group>(dest, source, nelems,   \\
                                                                         pe, channelId);         \\
    }

```

`nvshmemi_put_nbi_threadgroup` is implemented by `nvshmemii_put_nbi_threadgroup`. While the function `nvshmemii_put_nbi_threadgroup` calls `nvshmemi_transfer_rma_nbi` to generate rma request. `nvshmemi_transfer_rma_nbi` calls `nvshmemi_proxy_rma_nbi` to **use proxy to execute rma operation only when the thread is the first in threadgroup(warp/block)**.

```c
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_TRANSFER_STATIC __device__ NVSHMEMI_TRANSFER_INLINE void nvshmemi_transfer_rma_nbi(
    void *rptr, void *lptr, size_t bytes, int pe, int channelId) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    ... ...
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, channel_op, channelId);
    }
}

```

`nvshmemi_proxy_rma_nbi` calls `transfer_dma` directly. The code of `transfer_dma` is as follows. In `transfer_dma`, **we put four requests into proxy_channels_buf, which is used in proxy progress(like what nccl does)**. Four requests contain messages like data address, size and so on. We put channelId into high 32 bytes of put_dma_request_2.

```c
NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __device__ void transfer_dma(
    void *rptr, void *lptr, size_t bytes, int pe, int channel_op, int channelId) {
    // ... ...
    void *buf_ptr = nvshmemi_device_state_d.proxy_channels_buf;
    // ... ...

    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    uint64_t curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    // ... ...

    /* base_request_t
     * 32 | 8 | 8 | 8 | 8
     * roffset_high | roffset_low | op | group_size | flag */
    // ... ...

    /* put_dma_request_0
     * 56 | 8
     * laddr_high | flag */
    // ... ...

    /* put_dma_request_1
     * 32 | 16 | 8 | 8
     * size_high | size_low | laddr_low | flag */
    // ... ...

    /* put_dma_request_2
     * 32 | 16 | 8 | 8
     * channelId | pe | resv1 | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    *((volatile uint64_t *)req) = (uint64_t)(((uint64_t)channelId << 32) | (pe_u16 << 16) | curr_flag);
}

```

In proxy progress, `process_channel_dma` is used to process rma requests. We take out the requests generated before, and get channelId in dma_req_2 like what we do in `transfer_dma`. We pass channelId to function `nvshmemi_process_multisend_rma`, which calls `tcurrs->host_ops.rma`, and which is indeed `nvshmemt_ibrc_rma`.

```
/* ibrc.cpp */
int nvshmemt_ibrc_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                      rma_memdesc_t *remote, rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                      int is_proxy, int channelId) {
    int status = 0;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)tcurr->state;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibrc_ep *ep;
    struct ibrc_ep *ep2;
    struct ibv_sge *sge;
    int op_id;
    int index = channelId % ibrc_state->ndevs;
    int qpIndex;
    qpIndex = ((channelId / ibrc_state->ndevs) % (ibrc_state->ep_count / 2)) * 2 + (is_proxy ? ibrc_state->proxy_ep_idx : 0);

    ep = ibrc_state->ep[index][(ibrc_state->ep_count * pe + qpIndex)];
    // ... ...
}

```

We get port index and qp index by channelId. In this way, we split data by channel into different qp on dual-ports.

### amo

In DeepEP:csrc/kernels/internode.cu, we post amo operation like below in Dispatch.

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ODI1NWU0MGNmNTIyY2ExYjhhY2RiNWQ3OWYxZmM3MzFfZjFKY0w1aFhRZVdNR0RRa0ZqcDNnN09pWlZFeEkxZVpfVG9rZW46VlAzdGJ3Zmx2b3NMQmd4SW5weWMxVmo5bk9iXzE3NDk4MjA5NTE6MTc0OTgyNDU1MV9WNA)

`nvshmemx_signal_op` calls `nvshmemi_signal_op`. `nvshmemi_signal_op` calls `nvshmemi_transfer_amo_nonfetch`. Then `nvshmemi_transfer_amo_nonfetch` calls `nvshmemi_proxy_amo_nonfetch`. And then `nvshmemi_proxy_amo_nonfetch` calls `amo`.

```c
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemx_signal_op(uint64_t *sig_addr,
                                                                             uint64_t signal,
                                                                             int sig_op, int pe, int channelId) {
    nvshmemi_signal_op(sig_addr, signal, sig_op, pe, channelId);
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_signal_op(uint64_t *sig_addr,
                                                                 uint64_t signal, int sig_op,
                                                                 int pe, int channelId) {
    const void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (sig_op == NVSHMEMI_AMO_SIGNAL_SET && peer_base_addr != NULL) {
        // ... ...
    } else if (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        // ... ...
    } else {
        nvshmemi_transfer_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe,
                                                 (nvshmemi_amo_t)sig_op, channelId);
    }
}

NVSHMEMI_TRANSFER_STATIC __device__ NVSHMEMI_TRANSFER_INLINE void nvshmemi_transfer_amo_nonfetch(
    void *rptr, T value, int pe, nvshmemi_amo_t op, int channelId) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_amo_nonfetch<T>(rptr, value, pe, op);
    } else
#endif
    {
        nvshmemi_proxy_amo_nonfetch<T>(rptr, value, pe, op, channelId);
    }
}

template <typename T>
NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_proxy_amo_nonfetch(
    void *rptr, T swap_add, int pe, nvshmemi_amo_t op, int channelId) {
    amo<T>(rptr, 0 /* dummy value */, swap_add, 0, pe, op, channelId);
}

```

In `amo`, we do the same thing in `transfer_dma`. We generate four requests in amo. Due to in these four requests, we have no space for putting channelId. So we define a new request, which calls amo_request_4 and we put channelId in the high 32 bytes of it.

```
/* amo_request_4
     * 32 | 16 | 8 | 8
     * channelId | resv1 | resv | flag */
    idx += CHANNEL_ENTRY_BYTES;
    req = (uint64_t *)((uint8_t *)buf_ptr + (idx & (CHANNEL_BUF_SIZE - 1)));
    curr_flag = !((idx >> nvshmemi_device_state_d.proxy_channel_buf_logsize) & 1);
    *((volatile uint64_t *)req) = ((uint64_t)channelId << 32 | curr_flag);

```

In proxy, we use `proxy_channel_amo` to perform amo operations. We get five requests in reverse and get channelId in req_4. We pass it to `tcurr->host_ops.amo`, which is indeed `nvshmemt_ibrc_amo`.

```
int process_channel_amo(proxy_state_t *state, proxy_channel_t *ch, int *is_processed) {
    // ... ...
    status = tcurr->host_ops.amo(tcurr, pe, NULL, verb, &memdesc, bytes, 1, channelId);
    // ... ...
}

```

In nvshmemt_ibrc_amo, we choose the same qp in nvshmemt_ibrc_rma to ensure atomicity and ensure sequentiality of amo and rma operation.

```c
int nvshmemt_ibrc_amo(struct nvshmem_transport *tcurr, int pe, void *curetptr, amo_verb_t verb,
                      amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc, int is_proxy, int channelId) {
    int status = 0;
    transport_ibrc_state_t *ibrc_state = (transport_ibrc_state_t *)tcurr->state;
    struct ibrc_ep *ep;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    int op_id;
    struct ibrc_atomic_op op;

    int index = channelId % ibrc_state->ndevs;
    int qpIndex;
    qpIndex = ((channelId / ibrc_state->ndevs) % (ibrc_state->ep_count / 2)) * 2 + (is_proxy ? ibrc_state->proxy_ep_idx : 0);

    ep = ibrc_state->ep[index][(ibrc_state->ep_count * pe + qpIndex)];
    // ... ...
}

```

## Experiments

1. Single ports 8 nodes

- qp = 1

[tuning] 64 Best dispatch (FP8): SMs 24, NVL chunk 4, RDMA chunk 12: 32.30 GB/s (RDMA), 60.55 GB/s (NVL)

[tuning] 64 Best dispatch (BF16): SMs 24, NVL chunk 28, RDMA chunk 4: 27.61 GB/s (RDMA), 51.75 GB/s (NVL)

[tuning] 64 Best combine: SMs 24, NVL chunk 3, RDMA chunk 8: 29.32 GB/s (RDMA), 54.95 GB/s (NVL)

- qp = 4

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 12, RDMA chunk 28: 48.85 GB/s (RDMA), 90.84 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 32, RDMA chunk 16: 52.19 GB/s (RDMA), 97.04 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 16: 49.50 GB/s (RDMA), 92.06 GB/s (NVL)

1. Dual ports 4 nodes

- qp = 1

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 32, RDMA chunk 32: 48.11 GB/s (RDMA), 97.02 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 36, RDMA chunk 44: 41.31 GB/s (RDMA), 83.32 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 3, RDMA chunk 12: 38.64 GB/s (RDMA), 77.94 GB/s (NVL)

- qp = 4

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 20, RDMA chunk 16: 55.63 GB/s (RDMA), 111.16 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 16, RDMA chunk 16: 57.50 GB/s (RDMA), 114.89 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 12: 54.50 GB/s (RDMA), 108.91 GB/s (NVL)