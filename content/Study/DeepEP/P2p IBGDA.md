### p2pComm

- initialized: if init or not
    
- rc_handles: qp info, including qpn, lid
    
- rc_handles_len: Size of rc_handles, double size of local handles.
    
- mem_handle: buff register info, including lkey, rkey
    
    ```cpp
    struct nvshmemt_ib_common_mem_handle {
        struct ibv_mr *mr;
        void *buf;
        int fd;
        uint32_t lkey;
        uint32_t rkey;
        bool local_only;
    };
    ```
    
- nvshmemi_ibgda_device_state: some config of ibgda device
    
- **p2pcomm_device_state_d: using for transport(important)**
    
- trans: including some api for call function
    
- pe: server 0; client 1
    
- num_dci_eps: number of dci eps

### P2p完整流程

connect

- 创建p2pComm结构体，init时create qp
- 本地的qp handles信息(use for connect endpoints, created by create_qp)
- buff地址
- 注册buff后的mem_handle信息，包括lkey rkey等
- load对端的qp handles并调用connect_endpoints建联
- load对端的buf地址与mem_handle信息保存(注册的内存信息)

connect后，需要sleep来保证两侧均完成建联后才能收发，如果有mpirun的话可以尝试barrier

send

- ibgda_put_nbi_warp(…)
    
- amo_non_fetch_add(…)
    
    ```cpp
    __global__ void
    p2p_client_put_kernel(void *remote_ptr, void *local_buf, size_t size, int dst_pe, p2pcomm_ibgda_device_state_t *state, int iteration)
    {
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;
    
        size_t chunk_size = (size + num_warps - 1) / num_warps;
        size_t offset = warp_id * chunk_size;
        size_t data_size;
        if (offset < size)
        {
            data_size = min(chunk_size, size - offset);
            // 执行 put
            ibgda_p2p::nvshmemi_ibgda_put_nbi_warp(
                (uint64_t)remote_ptr + offset,
                (uint64_t)local_buf + offset,
                data_size,
                dst_pe,
                warp_id,
                lane_id,
                0,
                state);
        }
        else
        {
            data_size = 0; // ❗越界的 warp 不处理任何数据
        }
        __threadfence_system();
        if (lane_id == 0)
        {
            ibgda_p2p::nvshmemi_ibgda_amo_nonfetch_add((int *)remote_ptr + size / sizeof(int) + warp_id + 1, 1, dst_pe, warp_id, state);
        }
        __threadfence_system();
        if (warp_id == 0 && lane_id == 0)
        {
            int local_flag = 0;
            float *f_recv_buf = (float *)local_buf; // 正确的类型解释
            while (f_recv_buf[size / 4] != 1 + iteration)
            {
                ibgda_p2p::nvshmemi_ibgda_get_nbi_warp(
                    (uint64_t)remote_ptr + size,
                    (uint64_t)local_buf + size,
                    4,
                    dst_pe,
                    warp_id,
                    lane_id,
                    0,
                    state);
                __threadfence_system();
                auto qp = ibgda_p2p::ibgda_get_rc(dst_pe, 0, state);
                ibgda_p2p::ibgda_quiet(qp, state);
            }
        }
        __threadfence_system();
    }
    ```

recv

- p2p_server_poll_kernel(…) just wait for flag_ptr to be writen
    
    ```cpp
    __global__ void p2p_server_poll_kernel(void *recv_buf, size_t size, int iteration)
    {
        const int lane_id = threadIdx.x % 32;
        const auto warp_id = threadIdx.x / 32;
        if (lane_id == 0)
        {
            int *flag_ptr = (int *)recv_buf + warp_id + size / sizeof(int) + 1;
            // printf("before get value flag %d value %d\\n  ",*flag_ptr , (iteration+1)*1073741824);
            while (ibgda_p2p::ld_acquire_global(flag_ptr) != iteration + 1)
            {
            }
            ((float *)recv_buf)[size / 4] = iteration + 1;
        }
        __threadfence_system();
    }
    ```

### Example

client

```cpp
p2p_client_put_kernel<<<1, threads>>>(remote_ptr, local_buf, buf_size, 0, state, i);
```

第一个参数是block数量，意为发送给几个不同的server

第二个参数为每个block的threads数量，其中threads/32得到warp数量，每个warp负责一部分buffer的发送，每个warp负责一部分chunk的发送，每个warp的第一个thread在前面的warp都做完操作以后，执行一次amo

server

```cpp
p2p_server_poll_kernel<<<1, threads>>>(local_buf, buf_size, i);
```

每个warp中的第一天thread等待flag成功被改写