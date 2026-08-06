# 1 Abstract
本文章解决的问题是，在模型训练Tensor Parallelism带来的通信与计算overlapping，其中，提出了利用更细粒度的并行，将tp通信组内的reduce_scatter和all_gather实现到一个fuse kernel内，在不compromise GPU利用率的同时，最小化Effective Communication Time (ECT)；
一些Perliminary如下：
核心AllReduce被拆分成Reduce_scatter和All gather，对于SP而言，优势在于RMSNorm不用算n次，
# 2 Motivation
之前采用的通信与计算overlapping策略，大多是粗粒度的方式，需要把一个Gemm拆分成多个小Gemm，然后做Overlap，然而，具体的示意图如下，然而，他可能会有如下的三个核心问题：
* 多个kernel并行的时候，SM竞争随机，难以预测执行时间；
* 拆成多个小kernel后，由于存在reduce操作，因而有数据依赖，不一定能很好的并行起来；
* 拆分成多个Gemm以后，可能影响GPU的计算效率；
![[Zotero 2026-08-05 19.46.01.png]]
# 3 Design
因此，本文提出Flux，相比于之前gemm粒度的方案，核心采用tile粒度来做计算和通信的overlap，分别针对RS和AG分别提出了和GEMM fuse kernel的方法，首先先看ReduceScatter，其中，每个SM计算完对应的tile后，在epilogue warp中，获取这个tile对应结果对应的accumulator地址，随后，直接执行reduce操作将结果累加到那块地址上，注意，对于具体的write操作，如果是intra-node，直接调用st (register->GMEM)或 cp.async.bulk.tensor(TMA, register->SMEM->GMEM)，如果是inter-node，则利用NVSHMEM的put接口，直接进行操作；
![[Pasted image 20260805200820.png]]
对于AllGather的Fuse Gemm，则核心分为两部分，由于和RS不同，AG的数据不是已经由Kernel产生放在register内的，因而直接把AG移动到Host中来执行，利用DMA来进行拷贝操作，其中，tile的通信按照 $tile_{comm}$的顺序来进行发送，这个$tile_{comm}$是按咋实际的网络拓扑来决定，同时，为了避免Thread Block空转的问题 (每个SM调度上block后，会先等待signal被置为True后才能往下进行GEMM计算)，其中，在host function中，核心分为pull-based和push-based两种实现方式，核心都是通过`GetRemotePtr(...)`和`GetLocalPtr(...)`获取对应的本端tile的地址以及其该去往的tile地址，区别在于是由接收方主动的去拉取，还是由发送方主动的去send，发送的时候调用DataTransfer接口，这个接口在后续会讲一些具体的Optimization方法，在发送完成以后，调用`cuStreamWriteValue`，把signal置为True；
在AllGather的Gemm中，每个Thread Block首先等待对应的Tile的signal，注意，等待的顺序和tile发送的顺序对齐，利用`cuStreamWaitValue`等待signal为True后，则计算；
![[Pasted image 20260805202913.png]]
# 4 Optimization and Implemention details
## 4.1 Tile coordination swizzling
cutlass已有自己的swizzling方法，核心逻辑是假设为了避免memory traffic冲突，假设所有的TP rank采用完全相同的swizzling，则会出现每个rank同时往同一个rank写reduce的memory traffic contention，因此对于RS的fused Kernel，则直接把每个TP rank根据rank号偏移一位，即Rank1先写rank 1-7再0，而rank0则写rank0-7，这样子刚好能避开所有的memory traffic contention；
对于AG的fused kernel，则又不太一样，核心的点在于怎么选择$tile_{comm}$，其中，分为如下的几种情况：
* 如果机内通信且有NVLINK：则fullmesh连接，直接一次性全发
* 如果机内通信且只有PCIe：则采用经典的ring-based算法
* 如果机间通信有NVLINK：则将intra-node通信和inter-node通信overlap起来
* 如果机间通信只有PCIe：先做inter-numa通信，在intra-numa通信和intra-node通信overlap
## 4.2 Implementation details
* DataTransfer：如果允许GPU p2p，则直接cudaMemcpy，不然就用NCCL的send/recv api；
* signals：32 bit的contiguous内存，初始化时分配，在gemm结束时才销毁
* Communication tile size：在AG 中，由于发送操作由host function发起，且DMA数量有限，频繁的小tile传输可能会导致性能劣化，因而需要根据workload调参；