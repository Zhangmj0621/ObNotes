# 1. 系统内存与设备内存

## 1.1 内存架构
![[Pasted image 20260408144601.png]]

系统存储

- L1/L2/L3：多级缓存，其位置一般在CPU芯片内部
- System DRAM：片外内存，内存条；
- Disk/Buffer：外部存储，如Disk

GPU设备存储

- L1/L2 cache：多级缓存，在GPU芯片内部
- GPU DRAM：通常所指的显存

传输通道

- PCIE BUS：PCIe标准数据通道，数据从该通道从显卡到达主机
- BUS：总线，计算机内部各个存储之间交互数据的通道
- PCIe-to-PCIe：显卡之间通过PCIe直接传输数据
- NVLINK：显卡之间专用数据传输通道

## 1.2 传输通道速度

NVLINk > PCIe

# 2 内存之间操作

设备之间数据传输操作主要包括两类：系统到设备的数据操作、GPU间数据操作

## 2.1 数据从disk/系统内存到GPU

从磁盘 (Disk/SSD) 传数据到GPU内存，需要经过：硬盘 → 系统内存 → 设备内存的过程，其中传输速度慢，受限于BUS速度，内存速度，PCIe速度等等影响；因此尽量会避免从硬盘中频繁读写数据到GPU的操作，同时减少系统内存的换页操作，如使用pinned memory；

## 2.2 Pinned Memory

Pinned memory（Page-locked memory）页锁内存，能够提高数据在系统内存与GPU之间的传输速度，**其具体的做法是将数据在系统内存中锁住，避免数据在系统环境切换（如线程更换）时，数据从内存转移到硬盘**；

实际上改操作就是在段页式管理系统内存时，将特定的页表常驻在内存中，避免被任何换页算法换出；

```bash
root@node201:/workspace/infrawaves/zhangmj/BasicCUDA/memory_opt# ./testHost2Device 
[Host and Device Memory Opt Demo:] - Starting...
Test data transfer with pageable memory
>. hostToDeviceTransfer bandwith: 9.729803 GB/s
>. deviceToHostTransfer bandwith: 14.678223 GB/s
Test data transfer with pinned memory
>. hostToDeviceTransferWithPinned bandwith: 55.145630 GB/s
>. deviceToHostTransferWithPinned bandwith: 54.867783 GB/s
```

## 2.3 Zero Copy

Zero Copy是GPU计算单元直接从系统内存读取数据，不需要讲数据先从系统内存转移到GPU显存，可能的示意图如下：
![[Pasted image 20260408144615.png]]
具体做法是，直接申请pinned memory，然后直接将该内存指针传递给正在运算的kernel：

```cpp
float *a_h, *a_map; // 定义两个指针：a_h 内存原指针，a_map映射指针 
...
cudaGetDeviceProperties(&prop, 0);       // 获取GPU的特性，看是否支持地址映射 
if (!prop.canMapHostMemory) 
    exit(0);
cudaSetDeviceFlags(cudaDeviceMapHost);    // 设置设备属性，打开地址映射
cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);  // 开辟pinned memory
cudaHostGetDevicePointer(&a_map, a_h, 0);    // 地址映射 a_h ->  a_map.
kernel<<<gridSize, blockSize>>>(a_map);   
```

注意：

- zero copy 需要借助pinned memory ；
- zero copy 适用于只需要一次读取或者写入的数据操作，要频繁读写的数据，不建议用zero copy (HBM带宽往往在几百GB/s甚至TB/s级，远大于PCIe带宽，如要频繁读写，不如先拷贝到设备显存中)

```bash
root@node201:/workspace/infrawaves/zhangmj/BasicCUDA/memory_opt# ./testZeroCopy 
[Zero Copy Opt Vector Add] - Starting...
>. Data tranfer via global memory.  VectorAdd throughput: 0.317975 GB/s
>. Data tranfer via  zero copy.     VectorAdd throughput: 869.565186 GB/s
```

## 2.4 卡内异步拷贝

卡内的异步拷贝（Async-copy）是一种高效的数据传输操作，能够提高数据从全局内存到共享内存之间传输速度，同时降低了L1缓存和寄存器的使用量。其传输原理如下所示，正常情况下数据从全局内存（DRAM）上传输到共享内存（SMEM）需要先经过L2传输到L1，然后由L1写入寄存器（RF），再由寄存器传输给共享内存。如果使用异步拷贝，数据能够直接经由L2传输到SMEM，后者也可以用L1做个中转。这种绕开（Bypass）的操作可以让传输速度提升**2倍**以上。
![[Pasted image 20260408144628.png]]
卡内异步拷贝是在Ampere架构之后的GPU上面才支持的操作，主要是通过CUDA 里面的cooperative_groups::memcpy_async 操作完成，使用示例如下所示：

```cpp
/// This example streams elementsPerThreadBlock worth of data from global memory
/// into a limited sized shared memory (elementsInShared) block to operate on.
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024;
    const size_t elementsInShared = 128;
    __shared__ int local_smem[elementsInShared];

    size_t copy_count;
    size_t index = 0;
    while (index < elementsPerThreadBlock) {
        cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        cg::wait(tb);
        // Work with local_smem
        index += copy_count;
    }
}

```

## 2.5 设备之间数据传输

设备与设备（GPU-GPU）之间的内存数据传输有两种，方式1：经过CPU内存进行中转，方式2：设备之间直接访问的方法，这里主要讨论方式2。

设备之间（peer-to-peer）直接访问方式可以降低系统的开销，让数据传输在设备之间通过PCIE或者NVLINK通道完成，而且CUDA的操作也比较简单，示例操作如下：

```cpp
float* p0, *p1;
cudaSetDevice(0);                   // 将GPU0设置为当前设备
size_t size = N * sizeof(float);    // size设置为N个 float
cudaMalloc(&p0, size);              // GPU0开辟内存
cudaSetDevice(1);                   // 将GPU1设置为当前设备
cudaMalloc(&p1, size);              // GPU0开辟内存
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
```

```bash
root@node201:/workspace/infrawaves/zhangmj/BasicCUDA/memory_opt# ./testDevice2Device 
[Device to Device Memory Opt Demo:] - Starting...
>. Device to itself transfer.             Bandwith: 2856.096924 GB/s
>. Device to device transfer without p2p. Bandwith: 37.549473 GB/s
>. Device to device transfer with p2p     Bandwith: 326.174713 GB/s
```

设备之间的数据传递，还涉及到内存地址和传输速度的问题需要考虑，如内存地址可以使用UVA：

### 2.5.1 UVA的使用

UVA( Unified Virtual Address)，是对系统以及设备的内存统一管理的一种机制，在没有UVA的情况下，如下图所示，系统内存地址、两个设备GPU地址都是从0x0000开始到0xFFFF结束， 各个设备的内存的独立意味着需要显示地完成多个设备指针地址之间的映射(转换)操作。
![[Pasted image 20260408144640.png]]
有UVA的情况下，系统内存与设备内存认为进行了统一的编排。如示意图中，三个内存块统一在一起，从0x0000开始到0xFFFF结束：

![[Pasted image 20260408144650.png]]
### 2.5.2 NVLINK与PCIe的差距

在CUDA API中，我们用cudaMemcpy可以测试数据拷贝的速度差异，拷贝的形式有三种：

- cudaMemcpyHostToDevice： 主机到GPU；
- cudaMemcpyDeviceToHost：设备到主机；
- cudaMemcpyDeviceToDevice： GPU到GPU；

# 3 设备内存硬件

## 3.1 GDDR

GDDR（Graphics Double Data Rate, SDRAM）是一种针对显卡的存储介质。了解GDDR可以用DDR作为参考对比。目前在我们的电脑里面主流的内存条主要是DDR，使用较多的是DDR3/DDR4/DDR5。DDR作为一个为CPU服务的RAM，满足CPU运算的特点，针对的场景是：**小数据、多操作**，因此DDR的内存条一般设计为时延小，不太计较是否有大的带宽（bandwidth）；而GPU的特点是**数据大、操作少**，或者说单个操作内要进行大批量数据处理，所以在普通的DDR基础上，GDDR增加了带宽。

## 3.2 HBM

HBM(High Bandwidth Memory)高带宽存储，是另一种常用显存介质。 顾名思义这个存储介质有着"High Bandwidth"，参考NVIDIA P100所用的HBM来说明，该系列的显卡采用HBM第二代存储芯片，如下图是P100的硬质电路侧面视图，其中许多HBM2存储介质堆叠在基板（BASE DIE）上，且基板位置通过无源硅板（passive silicon interposer）紧邻P100芯片。

这种3D设计空间堆叠方式将存储介质（HBM，DRAM）层层拼接起来，这样的3D结构更能**接近芯片**单元，同时**空间占比相对较小**。该设计使得HBM的带宽和存储量都得到了提升，对比GDDR：

注：HBM做显存的优势这么明显，为什么没有完全取代GDDR？**因为生产一个3D结构的HBM的难度相比于生产一个平面结构的GDDR难度更大，所以HBM一般价格高。**

# 4 设备内存存储

前面提到GPU的内部存储分为片上存储和片下存储，指的硬件所在位置，为了满足GPU的应用场景，对存储功能进行了细分，包括：局部内存(local memory)、全局内存（global memory）、常量内存（constant memory）、图像/纹理（texture memory）、共享内存（shared memory）、寄存器（register）、L1/L2缓存、常量内存/纹理缓存（constant/texture cache），下面逐个介绍一下。

其中涉及到一些名词，可以参考CUDA手册/NVIDA芯片手册理解，这里先**通俗**地解释一下：

- SM（Streaming Multiprocessors）：理解为一个GPU内数据处理的大单元，好比多核的CPU芯片里面的一个核，CPU的一个核一般是运行一个线程，而SM能够运行多个轻量线程；
- nvcc：GPU程序的编译器，其实就是针对CUDA特殊化的gcc编译器；
- block： thread线程的集合单位。比如让GPU完成一个矩阵数据的运算 ，然后我们给参与运算的thread编个队，队名叫做block，对多个block编队就成了grid单位。
- warp： SM里面的运算执行单位，理解为运算时一个warp抓一把thread 扔进了计算core里面进行计算。

以英伟达的典型芯片Volta为例，首先全局概览一下芯片的存储单元的架构：
![[Pasted image 20260408144711.png]]
## 4.1 全局内存

全局内存（global memory）是数据常用的内存，它能被设备内的所有线程访问、全局共享，为片下（off chip）内存，前面提到的硬件HBM中的大部分都是用作全局内存。跟CPU架构一样，运算单元不能直接的使用全局内存的数据，需要经过缓存，其过程如下图所示：
![[Pasted image 20260408144721.png]]
在CUDA runtime中，全局内存申请一般是cudaMalloc开头的函数。

## 4.2 L1/L2 cache

L1/L2缓存（Cache）数据缓存，这个存储跟CPU架构的类似。L2为所有SM都能访问到，速度比全局内存块，所以为了提高速度有些小的数据可以缓存到L2上面；L1用于存储SM内的数据，SM内的运算单元能够共享，但跨SM之间的L1不能相互访问。

对于开发者来说，需要注意L2缓存能够提速运算，比如CUDA11 A100 上面L2缓存能够设置至多40MB的持续化数据(persistent data)，L2上面的持续化数据能够拉升算子kernel的**带宽和性能**，设置持续化数据的举例如下（摘取自CUDA 官网的exmaple）：

```cpp
cudaGetDeviceProperties( &prop, device_id);
// Set aside 50% of L2 cache for persisting accesses 
size_t size = min( int(prop.l2CacheSize * 0.50) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 
// Stream level attributes data structure 
cudaStreamAttrValue attr ;

attr.accessPolicyWindow.base_ptr = /* beginning of range in global memory */ ;

attr.accessPolicyWindow.num_bytes = /* number of bytes in range */ ;

// hitRatio causes the hardware to select the memory window to designate as persistent in the area set-aside in L2 
attr.accessPolicyWindow.hitRatio = /* Hint for cache hit ratio */

// Type of access property on cache hit 
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
// Type of access property on cache miss
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&attr);
```

通过persistent data可以使得运算提速1.5倍；

## 4.3 局部内存

局部内存(local memory) 是线程独享的内存资源，线程之间不可以相互访问，硬件位置是off chip状态，所以访问速度跟全局内存一样。局部内存主要是用来解决当**寄存器不足**时的场景，即在线程申请的变量超过可用的寄存器大小时，nvcc会自动将一部数据放置到片下内存里面。

注意，局部内存设置的过程是在编译阶段就会确定。

## 4.4 寄存器

寄存器（register）是线程能独立访问的资源，它所在的位置与局部内存不一样，是在片上（on chip）的存储，用来存储一些线程的暂存数据。寄存器的速度是访问中**最快**的，但是它的容量较小。以目前最新的Ampere架构的GA102为例，每个SM上的寄存器总量256KB，使用时被均分为了4块，且该寄存器块的64KB空间需要被warp中线程平均分配，所以在线程多的情况下，每个线程拿到的寄存器空间相当小。寄存器的分配对SM的占用率（occupancy）存在影响，可以通过[CUDA Occupancy Calculator](https://link.zhihu.com/?target=https%3A//xmartlabs.github.io/cuda-calculator/) 计算比较，举例：如图当registers从32增加到128时，occupancy从100%降低到了33.33%：

## 4.5 共享内存

共享内存（shared memory) 是一种在block内能访问的内存，存储硬件位于芯片上（on chip），访问速度较快，共享内存主要是缓存一些需要反复读写的数据。可以通过一个矩阵运算的例子说明shared memory的作用，比如完成矩阵运算C = A X B， Ai_row表示A的第i行数据， Bj_col表示B的第j列数据，cij表示第i行 第j例的数值，有：

$$ cij  = A_irow \times B_jcol $$

假设要得到C矩阵的第i行Ci_row的数据，**上述运算需要进行N次**，N为：B矩阵列宽大小。

对于该计算而言，运算中的Ai_row保持不变，Bj_col进行迭代更新。Ai_row假设使用global memory，则每次运算都需要重新加载，数据重复加载了**N次**。然而Ai_row数据是可以复用的，所以将Ai_row放入共享内存中，这样相同的数据避免反复加载（Ai_row数据加载是要**1次），**从而提高运算效率。相比只用全局内存，共享内存在上述矩阵运算上可以提升20~50GB/s的速度。

```cpp
root@node201:/workspace/infrawaves/zhangmj/BasicCUDA/memory_opt# ./testSharedMemory 
[Shared Memory Application: Array Sum.] - Starting...
Sum array with shared memory.       Elapsed time: 0.264493 ms 
Sum array without shared memory.    Elapsed time: 0.012185 ms
```

注：共享内存与L1/L2存在差异。共享内存与L1的位置、速度极其类似，区别在于共享内存的控制与生命周期管理与L1不同，共享内存的使用受用户控制，L1受系统控制，CUDA编程的时候，shared memory更利于block之内线程之间数据交互。

## 4.6 常量内存

常量内存(constant memory) 是指存储在片下存储的设备内存上，但是通过特殊的常量内存缓存（constant cache）进行缓存读取，常量内存为只读内存。为什么需要设立单独的常量内存？直接用global memory或者shared memory不行吗？

主要是解决一个warp内多线程的**访问相同数据**的速度太慢的问题，如下图所示：**(Shared memory一定做不到！)**
![[Pasted image 20260408144737.png]]
所有运算的thread都需要访问一个constant_A的常量，在存储介质上面constant_A的数据只保存了一份，而内存的物理读取方式决定了这么多thread不能在同一时刻读取到该变量，所以会出现先后访问的问题，这样使得并行计算的thread出现了运算时差。常量内存正是解决这样的问题而设置的，它有对应的cache位置产生多个副本，让thread访问时不存在冲突，从而提高并行度。

```cpp
__constant__ int c1 = 10;  // 声明__constant__ 即可。
__global__ void kernel1(int *d_dst) {
   int tId = threadIdx.x + blockIdx.x * blockDim.x;
   d_dst[tId] += c1;
}
```

需要说明的是，在硬件上面，constant单元也分了多级（L1/L1.5/L2），而且存在线程访问延时，比如上述例子中的广播操作，当线程数量增加时延时也会随之增加。
![[Pasted image 20260408144748.png]]
## 4.7 texture memory

图像/纹理（texture memory）是一种针对图形化数据的专用内存，其中texture直接翻译是纹理的意思，但根据实际的使用来看texture应该是指通常理解的1D/2D/3D结构数据，相邻数据之间存在一定关系，或者相邻数据之间需要进行相同的运算。 texture内存的构成包含 global + cache + 处理单元，texture为只读内存。texture的优势：

- texture memory 进行图像类数据加载时， warp内的thread访问的数据地址相邻，从而减少带宽的浪费。
- texture 在运算之前能进行一些处理（或者说它本身就是运算），比如聚合、映射等。