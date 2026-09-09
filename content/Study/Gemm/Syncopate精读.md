# 概述
本文档精读Syncopate: Efficient Multi-GPU AI Kernels via  Automatic Chunk-Centric Compute-Communication Overlap；
# Motivation
为了overlap，现有的distributed compiler会把GEMM切成多个小kernel，然后把不同的通信计算kernel放到不同的stream来overlap，然而这有如下三个问题：
* 每个小kernel都有launch和同步开销；
* 把大GEMM拆成很多小GEMM会降低SM利用率；
* 通信的尾部很难被隐藏；
因此本文提出syncopate，不拆成很多个compute kernel，而是在一个fuse kernel内部，让不同tile在通信过程中逐步进行，其中核心提出chunk-level抽象进行通信；
# Design
## Why chunk？
核心抽象以chunk为级别通信，而不是tile，因为通信可能并不适合一次传输一个tile；
现在的通信模式主要分为如下几种：
* Copy engine：带宽利用率最高，但不适合传输小数据；
* TMA：需要更大的tensor才能达到高吞吐，但仍然不如CE，且需要占用SM；
* load/store：同步操作，不能异步通信，需要强制占用SM；
![[Pasted image 20260909163835.png|0]]
同时，通信粒度太小，会导致带来大量的signal/wait；
因此论文，引入了中间层chunk，将tensor切分为多个chunk，以chunk粒度进行通信；
![[Pasted image 20260909163920.png]]
## How to express communication？
核心可以通过P2P和Collective communication来表达所有的通信，具体的几种不同的通信操作可以视作如下的表达：
![[Pasted image 20260909164434.png]]
其中，syncopate的输入需要有人来提供通信计划，这个计划可以来自：
1. 用户手写；
2. 论文提供的1D/2D Allgather、ReduceScatter模版；
3. 现有的分布式compiler，如Alpa、Domino、Mercury；
4. 某些collective synthesis算法；
## How to express computation？
计算kernel仍是用户写的普通triton kernel，需要加一些额外注释；论文需要如下的三类信息：
1. tile size，告诉编译器每个tile对应tensor的哪一块；
2. tile identifier：告诉编译器当前循环当前处理哪个tile；
3. tile scheduler：告诉编译器tile原本是如何遍历的；
## What compiler realy does？
编译器具体是如下五步：
* 解析通信计划：把高层计划转成具体的chunk操作；
* 解析tile与chunk的对应关系：判断每个tile负责读取哪个chunk？会产生哪个chunk？是否要等通信完成？
* 构建依赖图：依赖图中同时包含compute tile/communication chunk/communication operation；
* 插入同步：如果一个tile的数据还没到，就插入wait，不同的通信方式同步方式不一样；
* 重写tile调度顺序：chunk-level reorder+intra-chunk swizzle；
## How to auto tune？
最重要的问题是如何确认最优的chunk_size、通信backend、split config、chunk_size等等；
