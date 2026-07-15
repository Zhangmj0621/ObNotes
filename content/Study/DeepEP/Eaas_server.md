![[Pasted image 20260408134830.png]]
主要函数

- ibgda_client_put_kernel:
    
    ```cpp
    template <bool do_ibgda>
    __global__ void ibgda_client_put_kernel(
        const uint32_t dspe,                        // 发给server时为0
        uint32_t* batch_count_dptr,                  // server的最大数量
        uint32_t* batch_size_dptr,                  // 批处理大小
        void **remote_buff_dptr_dptr,               // server接收缓冲区地址
        void **send_addrs_dptr_dptr,         // 每个server的发送缓冲区地址
        size_t *send_data_sizes_with_padding_dptr, // 包含填充的总数据大小
        p2pcomm_ibgda_device_state_t **p2p_state_dptr_dptr
    )
    {
    	// ... ...
    }
    ```
    
    具体调用逻辑为：
    
    1. 检测有负载的server数量是否大于最大可用server数量
    2. 计算每个warp需要发送的server数量（向上取整）
    3. **将每个有负载的server分配给具体的warp**执行 assigned_warp，output_global_indices中存放该thread所在的warp内需要负责的几个server
    4. 对于该warp内需要负责发送的server，获取datasize，源地址，目的地址，调用三次nvshmemi_ibgda_put_nbi_warp(…)，一次发送total payload，一次发送kBufferHead，一次发送kCompletionFlag
- launchIbgdaClientPutKernel
    
    提供调用kernel函数的接口
    
    block数量为1，使用32个warp即256个threads来进行发送
    
    ```cpp
    extern "C" cudaError_t launchIbgdaClientPutKernel(
        IBGDA_Remote_Context *remote_context,
        IBGDA_Header_Meta *header_meta,
        IBGDA_Client_Send_Tensor_Info *send_tensor_info,
        IBGDA_Client_Send_Data_Info *send_data_info,
        IBGDA_Client_Recv_Data_Info *recv_data_info,
        bool do_ibgda
    )
    {
        constexpr uint32_t dspe = 0;
        constexpr int num_threads = 256;
        constexpr int num_blocks = 1;
    
        if (do_ibgda) {
            // ibgda_client_put_sequential_kernel<true>
            ibgda_client_put_kernel<true>
            <<<num_blocks, num_threads, 0, remote_context->client_stream>>>(
    	        // ... ...
    	      );
    	  else {
    			  // ... ...
    		}
    }
    ```

### structure

```cpp
struct IBGDAUtilsPtr {
  size_t * send_data_sizes_with_padding_dptr; /* Total data size including padding for each server */
  size_t * recv_data_sizes_with_padding_dptr; /* Total data size including padding for each server */
  void ** server_send_addrs_dptr_dptr;                 /* Send buffer addresses for each server */
  uint32_t * server_padding_sizes_dptr;         /* Padding size for each server */
  uint32_t * server_cumsum_batch_size_dptr;            /* Cumulative sum of batch sizes for each server */
  void** recv_addrs_dptr_dptr;
  void** recv_local_header_dptr_dptr; // 轮训标志位的state buffer.
  uint32_t* batch_count_dptr; // adneed align with uint32_t
};

```

### 调用链

`launchIbgdaClientPutKernelSingleWar` 为上层接口

主要调用两个kernel函数

```cpp
CUDA_CHECK(launchInitializeIbgdaHeadersKernel(
        send_buff_dptr,  // 使用正确的发送缓冲区指针
        &header_meta,
        &tensor_info,
        ibgda_utils_ptr,
        current_stream
    ));
    
CUDA_CHECK(launchIbgdaClientPutKernelSingleWarp(
                    batch_size_dptr,
                    remote_buff_dptr_dptr,
                    ibgda_utils_ptr->server_send_addrs_dptr_dptr,
                    ibgda_utils_ptr->send_data_sizes_with_padding_dptr,
                    p2p_state_dptr,
                    current_stream
                ));
```

- `launchInitializeIbgdaHeadersKernel`启动`initialize_ibgda_headers_kernel_parallel`用1个block执行，避免重复启动kernel，主要用来准备数据，便于RDMA发送，在这个kernel函数里，
    
    - 主要用来将客户端的输入数据(weight, hidden state)打包成适合网络传输格式(计算paddings)以16字节对齐；
    - 将输入按照hidden_state, weight, input_scale data顺序存入s_payload_ptr中，末尾包含第一步计算的padding，将active_experts_per_thread协作拷贝到s_header中;
    - 需要注意，send_addrs_dptr_dptr的地址开头存放Head(BufferHead*)，随后存放s_payload
    - 将数据拷贝到s_payload_ptr中时，所有threads一同协作拷贝，存在向量化拷贝；
    
    ```cpp
    // process and put send_tensor_info into ibgda_utils_ptr
    // send ptr and recv ptr are all in client_send_buf(which need register)
    initialize_ibgda_headers_kernel_parallel<<<num_blocks, num_threads, 0, stream>>>(
            header_meta->client_seed,
            header_meta->hidden_size,
            header_meta->data_type,
            header_meta->layer,
            header_meta->batch_size_dptr,
            ibgda_utils_ptr->batch_count_dptr,
    
            send_tensor_info->hidden_state_dptr,
            send_tensor_info->expert_weights_dptr,
            send_tensor_info->input_scale_dptr,
            send_tensor_info->active_experts_indices_dptr,
    
            ibgda_utils_ptr->server_send_addrs_dptr_dptr,
            ibgda_utils_ptr->send_data_sizes_with_padding_dptr,
            ibgda_utils_ptr->server_padding_sizes_dptr,
            ibgda_utils_ptr->server_cumsum_batch_size_dptr,
            
            ibgda_utils_ptr->recv_addrs_dptr_dptr,
            ibgda_utils_ptr->recv_data_sizes_with_padding_dptr,
            ibgda_utils_ptr->recv_local_header_dptr_dptr,
            
            client_send_buf
        );
    ```
    
    注：北坡准备数据，不存在warp的概念，一个block内的thread每人拷贝一部分
    
- `launchIbgdaClientPutKernelSingleWarp`负责实际发送，操作比较简单，以warp为概念（上层调用时，每个server选用一个warp进行发送(qp)）
    
    北坡的发送意思为，每个server发送只用一个warp，也即单qp
主要函数

- ibgda_client_put_kernel:
    
    ```cpp
    template <bool do_ibgda>
    __global__ void ibgda_client_put_kernel(
        const uint32_t dspe,                        // 发给server时为0
        uint32_t* batch_count_dptr,                  // server的最大数量
        uint32_t* batch_size_dptr,                  // 批处理大小
        void **remote_buff_dptr_dptr,               // server接收缓冲区地址
        void **send_addrs_dptr_dptr,         // 每个server的发送缓冲区地址
        size_t *send_data_sizes_with_padding_dptr, // 包含填充的总数据大小
        p2pcomm_ibgda_device_state_t **p2p_state_dptr_dptr
    )
    {
    	// ... ...
    }
    ```
    
    具体调用逻辑为：
    
    1. 检测有负载的server数量是否大于最大可用server数量
    2. 计算每个warp需要发送的server数量（向上取整）
    3. **将每个有负载的server分配给具体的warp**执行 assigned_warp，output_global_indices中存放该thread所在的warp内需要负责的几个server
    4. 对于该warp内需要负责发送的server，获取datasize，源地址，目的地址，调用三次nvshmemi_ibgda_put_nbi_warp(…)，一次发送total payload，一次发送kBufferHead，一次发送kCompletionFlag
- launchIbgdaClientPutKernel
    
    提供调用kernel函数的接口
    
    block数量为1，使用32个warp即256个threads来进行发送
    
    ```cpp
    extern "C" cudaError_t launchIbgdaClientPutKernel(
        IBGDA_Remote_Context *remote_context,
        IBGDA_Header_Meta *header_meta,
        IBGDA_Client_Send_Tensor_Info *send_tensor_info,
        IBGDA_Client_Send_Data_Info *send_data_info,
        IBGDA_Client_Recv_Data_Info *recv_data_info,
        bool do_ibgda
    )
    {
        constexpr uint32_t dspe = 0;
        constexpr int num_threads = 256;
        constexpr int num_blocks = 1;
    
        if (do_ibgda) {
            // ibgda_client_put_sequential_kernel<true>
            ibgda_client_put_kernel<true>
            <<<num_blocks, num_threads, 0, remote_context->client_stream>>>(
    	        // ... ...
    	      );
    	  else {
    			  // ... ...
    		}
    }
    ```

### structure

```cpp
struct IBGDAUtilsPtr {
  size_t * send_data_sizes_with_padding_dptr; /* Total data size including padding for each server */
  size_t * recv_data_sizes_with_padding_dptr; /* Total data size including padding for each server */
  void ** server_send_addrs_dptr_dptr;                 /* Send buffer addresses for each server */
  uint32_t * server_padding_sizes_dptr;         /* Padding size for each server */
  uint32_t * server_cumsum_batch_size_dptr;            /* Cumulative sum of batch sizes for each server */
  void** recv_addrs_dptr_dptr;
  void** recv_local_header_dptr_dptr; // 轮训标志位的state buffer.
  uint32_t* batch_count_dptr; // adneed align with uint32_t
};

```

### 调用链

`launchIbgdaClientPutKernelSingleWar` 为上层接口

主要调用两个kernel函数

```cpp
CUDA_CHECK(launchInitializeIbgdaHeadersKernel(
        send_buff_dptr,  // 使用正确的发送缓冲区指针
        &header_meta,
        &tensor_info,
        ibgda_utils_ptr,
        current_stream
    ));
    
CUDA_CHECK(launchIbgdaClientPutKernelSingleWarp(
                    batch_size_dptr,
                    remote_buff_dptr_dptr,
                    ibgda_utils_ptr->server_send_addrs_dptr_dptr,
                    ibgda_utils_ptr->send_data_sizes_with_padding_dptr,
                    p2p_state_dptr,
                    current_stream
                ));
```

- `launchInitializeIbgdaHeadersKernel`启动`initialize_ibgda_headers_kernel_parallel`用1个block执行，避免重复启动kernel，主要用来准备数据，便于RDMA发送，在这个kernel函数里，
    
    - 主要用来将客户端的输入数据(weight, hidden state)打包成适合网络传输格式(计算paddings)以16字节对齐；
    - 将输入按照hidden_state, weight, input_scale data顺序存入s_payload_ptr中，末尾包含第一步计算的padding，将active_experts_per_thread协作拷贝到s_header中;
    - 需要注意，send_addrs_dptr_dptr的地址开头存放Head(BufferHead*)，随后存放s_payload
    - 将数据拷贝到s_payload_ptr中时，所有threads一同协作拷贝，存在向量化拷贝；
    
    ```cpp
    // process and put send_tensor_info into ibgda_utils_ptr
    // send ptr and recv ptr are all in client_send_buf(which need register)
    initialize_ibgda_headers_kernel_parallel<<<num_blocks, num_threads, 0, stream>>>(
            header_meta->client_seed,
            header_meta->hidden_size,
            header_meta->data_type,
            header_meta->layer,
            header_meta->batch_size_dptr,
            ibgda_utils_ptr->batch_count_dptr,
    
            send_tensor_info->hidden_state_dptr,
            send_tensor_info->expert_weights_dptr,
            send_tensor_info->input_scale_dptr,
            send_tensor_info->active_experts_indices_dptr,
    
            ibgda_utils_ptr->server_send_addrs_dptr_dptr,
            ibgda_utils_ptr->send_data_sizes_with_padding_dptr,
            ibgda_utils_ptr->server_padding_sizes_dptr,
            ibgda_utils_ptr->server_cumsum_batch_size_dptr,
            
            ibgda_utils_ptr->recv_addrs_dptr_dptr,
            ibgda_utils_ptr->recv_data_sizes_with_padding_dptr,
            ibgda_utils_ptr->recv_local_header_dptr_dptr,
            
            client_send_buf
        );
    ```
    
    注：北坡准备数据，不存在warp的概念，一个block内的thread每人拷贝一部分
    
- `launchIbgdaClientPutKernelSingleWarp`负责实际发送，操作比较简单，以warp为概念（上层调用时，每个server选用一个warp进行发送(qp)）
    
    北坡的发送意思为，每个server发送只用一个warp，也即单qp