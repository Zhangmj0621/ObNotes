### API

```
cuMemCreate 创建一个handle，分配实际的物理内存
cuMemAddressReserve 创建一个地址范围，产生ptr
cuMemMap 将handle映射到ptr

NCCL使用UDS间接调用：
cuMemExportToShareableHandle 导出句柄
cuMemImportFromShareableHandle 导入句柄

```

> 这其中不涉及任何源buffer和目的buffer
> 
> 因此源buffer向目的buffer的拷贝应当是先拷入ptr作为中转

> 使用示例

```
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CHECK_CUDA(call)                                                         \\
    {                                                                            \\
        CUresult err = call;                                                     \\
        if (err != CUDA_SUCCESS) {                                               \\
            const char *err_name;                                                \\
            cuGetErrorName(err, &err_name);                                      \\
            std::cerr << "CUDA error: " << err_name << " at " << __FILE__ << ":" \\
                      << __LINE__ << std::endl;                                  \\
            exit(EXIT_FAILURE);                                                  \\
        }                                                                        \\
    }

size_t BUFFER_SIZE = 1024 * 1024 * 1024; // 1 MB buffer

#define ROUND_UP(x,y) ((x+(y-1))/y*y)

__global__ void set(void *src){
    int *data = (int *)src;
    for(int i=0;i<=10;i++){
        // data[i] = 10 - i;
        data[i] = i;
    }
}

__global__ void get(void *src){
    int *data = (int *)src;
    for(int i=0;i<=10;i++){
        printf("%d ", data[i]);
    }
    printf("\\n");
}

int main() {
    // Initialize the CUDA driver
    CHECK_CUDA(cuInit(0));

    size_t granularity = 0;
    CUmemAllocationProp prop2 = {};
    prop2.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop2.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop2.location.id = 0;
    cuMemGetAllocationGranularity(&granularity, &prop2,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    BUFFER_SIZE = ROUND_UP(BUFFER_SIZE, granularity);
    printf("BUFFER_SIZE = %ld\\n", BUFFER_SIZE);

    // Get device handles for two GPUs
    CUdevice src_device, dst_device;
    CHECK_CUDA(cuDeviceGet(&src_device, 0));
    CHECK_CUDA(cuDeviceGet(&dst_device, 1));

    // Create contexts for both GPUs
    CUcontext src_context, dst_context;
    CHECK_CUDA(cuCtxCreate(&src_context, 0, src_device));
    CHECK_CUDA(cuCtxCreate(&dst_context, 0, dst_device));

    // Switch to source device context
    CHECK_CUDA(cuCtxSetCurrent(src_context));

    // Create a virtual address range
    CUdeviceptr virtual_address;
    CHECK_CUDA(cuMemAddressReserve(&virtual_address, BUFFER_SIZE, 0, 0, 0));

    // Allocate physical memory on the source GPU
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dst_device;

    CUdeviceptr physical_memory_src;
    CHECK_CUDA(cuMemCreate(&physical_memory_src, BUFFER_SIZE, &prop, 0));

    // Map the physical memory to the virtual address
    CHECK_CUDA(cuMemMap(virtual_address, BUFFER_SIZE, 0, physical_memory_src, 0));

    // Set access permissions for the source device
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = src_device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CHECK_CUDA(cuMemSetAccess(virtual_address, BUFFER_SIZE, &access_desc, 1));

    CHECK_CUDA(cuCtxSetCurrent(dst_context));
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = dst_device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CUDA(cuMemSetAccess(virtual_address, BUFFER_SIZE, &access_desc, 1));

    CHECK_CUDA(cuCtxSetCurrent(src_context));
    CUdeviceptr data, ans;
    cuMemAlloc(&data, BUFFER_SIZE);
    cuMemAlloc(&ans, BUFFER_SIZE);
    set<<<1,1>>>((void *)data);
    cudaMemcpy((void *)virtual_address, (void *)data, BUFFER_SIZE,            cudaMemcpyDeviceToDevice);
    cudaMemcpy((void *)ans, (void *)virtual_address,  BUFFER_SIZE, cudaMemcpyDeviceToDevice);
    get<<<1,1>>>((void *)ans);
    CHECK_CUDA(cuCtxSynchronize());

    // Clean up
    CHECK_CUDA(cuMemUnmap(virtual_address, BUFFER_SIZE));
    CHECK_CUDA(cuMemRelease(physical_memory_src));
    CHECK_CUDA(cuMemAddressFree(virtual_address, BUFFER_SIZE));

    CHECK_CUDA(cuCtxDestroy(src_context));
    CHECK_CUDA(cuCtxDestroy(dst_context));
    return 0;
}

```

> 执行结果

```
./test3
BUFFER_SIZE = 1073741824
0 1 2 3 4 5 6 7 8 9 10

```

> 内存占用

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGY5MmNkMGMwYWE3OTcxN2ZmYjE5Yjc1ODkwNTIwNTZfOU1kNkNXYU9vRWZ3WVEwUTZVN3I0eXo5amZVQ2xZalJfVG9rZW46VW5qRWI0cXRYb2RCTWR4N1RYTWNSeWRZbkliXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

### Chunck buffer

> send/recv端分别分配一块内存，通过vmm技术将recv端内存[ global ]在两端映射，生成两f端的dptr
> 
> 从最终映射的地址看proxy在做send chunck到dptr的拷贝 下图中[ set ]为dptr的地址

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MmZjNWJhZTE4MDQxNDZhMGUyMGQ0Mjc1NWNlMjAxZmRfMmFHNFJnOHdxRnB2SE1RSU5PcUxaeWp4MlhTSHRPUG9fVG9rZW46R3duN2JZMVlUb1dOTHB4NGdiYWNuM05OblVIXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

> send端
> 
> ncclCudaCalloc

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OGUzODZlZTZiMGZjNTY4ZTBhOWFhZTk1Zjg1MDM5ZjJfU2xTR0w0TlBGQTRWU1pUWVdNVGJUVFRIbU9kb29xOTRfVG9rZW46UnhyRmJOMU1ub0RpZ0x4UkdtV2NQZjc1bmhkXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

> dptr

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=N2Q1ODZiNDVkYjk3ZGNlYTllM2VhMmM0MDgyMTA4MjJfVkh2VU1aY0tLYnUwQ1FiT2c0eTN5UXlhQ25DWVBsVkZfVG9rZW46U2lqcGI4c2Vyb1FjZ0l4djZuM2NkRGR2bnhmXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

> recv端
> 
> ncclP2pAllocateShareableBuffer

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTkzYTg1MzNmNGE1NzYzMGJmZWU1ZmFmZGY1YmUxZThfVTRjSGxSemdwdWM4OXNWOU1JUk0wRmdubnhYcTFJak9fVG9rZW46Rk1QTWJmdHdpb2FhMGZ4bUlCS2NiMlZDbkZlXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

### kernel 拷贝的同步 共享主机内存

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OWI1NGIyOTEwMDUyYzg3ZWI5NTdhZDllNmI2YTVlMmRfRHB0bHY0M2tsTjFyZnRsQ01QZ0J6dVdqVVptZHZzYmZfVG9rZW46RkVXV2JWUW1lb3dTTnR4eFhQaWMwaHlIbmhnXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

### Shm 步进控制(demo)

> 生产

```
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void process1() {
    const char* shm_name = "/shared_mem";
    size_t size = 1024 * sizeof(int); // 共享内存大小

    // 创建共享内存对象
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, size); // 设置共享内存大小

    // 映射共享内存到进程地址空间
    int* ptr = (int *)mmap(0, size, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    for(int i=0; i < 10; i++){
        ptr[i] = 0;
    }
    int idx = 0;
    while(1){
        if(ptr[idx] == 0){
            // Copy
            printf("copy [ %d ]\\n", idx);
            ptr[idx] = 1;
            idx = (idx + 1) % 10;
        }
        sleep(1);
    }
}
int main(){
        process1();
}

```

> 消费

```
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

void process2() {
    const char* shm_name = "/shared_mem";
    size_t size = 1024 * sizeof(int); // 共享内存大小

    // 打开共享内存对象
    int shm_fd = shm_open(shm_name, O_RDWR, 0666);

    // 映射共享内存到进程地址空间
    int* ptr = (int *)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    int idx = 0;
    while(1){
        // Paste
        if(ptr[idx] == 1){
            ptr[idx] = 0;
            idx = (idx + 1) % 10;
        }
    }

    // 释放共享内存（进程结束时）
    munmap(ptr, size);
    close(shm_fd);
    shm_unlink(shm_name);
}
int main(){
        process2();
}

```

### 设计

> 1、recv端没有proxy，需要两端启动proxy
> 
> 2、完成源buffer到全局buffer(dptr)，全局buffer到目的buffer的拷贝
> 
> 3、buffer地址和收发参数在结构体中的传递
> 
> 4、recv端需要同样找到这些信息

> send端获取到全局buffer

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MWQxYzAwMDQwMjI2N2EzOTIzYjQxNzU2MWY2MzkxZTFfUkg1eTZlWFJuaTZHVXBNRTVmbUJEYk9hRjhRUDViRlpfVG9rZW46UklBRWI2Qlp3b0drUnJ4YlVtd2MxSlh4blhlXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

### 最新进展

```
src->全局buffer / 全局buffer->dst获取了buffer的地址
需要添加两端的同步控制

```

> 这里验证了recv端可以从全局buffer获取到预制的数据

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OGY4YTdjMGQ1NGIxMjIxOWUzMDMzN2NlYzRhNTdlM2FfbU5DdzZ0RnYydFYybUQwRlFiQ2RkTENweTNwNjJwSXFfVG9rZW46VlNGcWI2dmJzb3NCWWV4TUdPNGM5b0NsbkxnXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

> 临时使用睡眠增加延时代替同步，4字节send/recv稳定跑通

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OGM1ZmUxMzI3MGI5YmRhMmE1ZWY3OGNlYzExYTIzMjBfSklUQTVVY0c2azhMVEMxYzZIS1MyVnBrTlJkTWhXY05fVG9rZW46Qkx5d2JpajFRb3ByTzh4R2pIZmNzdkJhbjVzXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

### code

> d2:/root/waibibabu/p2p
> 
> 验证脚本
> 
> d2:/root/waibibabu/runp2p

### 2025.2.6

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=NDQ5YjMzNWVlZGM0ODgxYzEzYWNiMTcwMjhiOTY2NDZfZjZNT256Zkt2MTIyVGU0SlA2YXN4a0NTb2RHWFUxbVFfVG9rZW46T0plRWJaU0Y3b29tUG14OFg2MGMyc3dYbjBjXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

> TODO：找到chunck buffer size，合并机内机间代码

### 2025.2.12

> 机内机间打通

```
<http://183.207.7.174:8081/moon/vccl_2.21.51x/-/tree/nokernel?ref_type=heads>

```

> 执行脚本

```
#! /usr/bin/bash

MPIRUN=/root/waibibabu/mpich-4.2.3/build/bin/mpirun
LIBRARY_PATH=./mpich-4.2.3/build/lib/:./gitlab/vccl_2.21.51x/build/lib/
$MPIRUN -np 4 -host node131:2,node132:2 -genv NCCL_BUFFSIZE=33554432 -genv NCCL_DEBUG= -genv NCCL_DEBUG_SUBSYS=p2p -genv NCCL_IB_HCA=mlx5_0,mlx5_1 -genv NCCL_P2P_USE_CUDA_MEMCPY=1 -genv NCCL_PROTO=Simple -genv NCCL_IB_MERGE_NICS=0 -genv NCCL_MAX_NCHANNELS=1 -genv LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH nccl-tests/build/sendrecv_perf -b 4M -e 512M -f 2 -w 0 -n 1

```

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjVhOTgxZDdhYTU2NjM4ZTU1OWRjYTgwYTY5MzM3MDdfZHpQcXUwZTNVUXN1QUZYVTRKNGhPU1V5NHhRckJoREVfVG9rZW46TDNLbWJrWEhib0FFclh4NTB1Z2NEbnRFbmplXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)

### 2025.2.14

> 支持alltoall

```
#! /usr/bin/bash

MPIRUN=/root/waibibabu/mpich-4.2.3/build/bin/mpirun
LIBRARY_PATH=./mpich-4.2.3/build/lib/:./gitlab/vccl_v2/build/lib/

$MPIRUN -np 4 -host node131:2,node132:2 -genv NCCL_BUFFSIZE=33554432 -genv NCCL_DEBUG= -genv NCCL_DEBUG_SUBSYS=init -genv NCCL_IB_HCA=mlx5_0,mlx5_1 -genv NCCL_P2P_USE_CUDA_MEMCPY=1 -genv NCCL_PROTO=Simple -genv NCCL_IB_MERGE_NICS=0 -genv NCCL_MAX_NCHANNELS=1 -genv LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH nccl-tests/build/sendrecv_perf -b 4 -e 4G -f 3 -w 1 -n 1

```

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWM1NTUxN2E3NzAxMjA2NjFhYjU5Mjg0YWQ5MmQ3ZjhfQnYwb1EyOHZMWFhmdlVpdjZJVk93YmFNYjRnMkpaQ2FfVG9rZW46TDk2dmJPYnQyb3prT2l4S0J3SGNRWlZ6bnNmXzE3NDk4MTk5MDg6MTc0OTgyMzUwOF9WNA)