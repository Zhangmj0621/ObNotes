# Overview
* 一个GPU的kernel执行首先被他的多层线程定义：thread、warp、warpgroup、CTA、cluster和grid；很多Blackwell的操作有他们自己的天然scope，一个TMA拷贝被一个单线程launch、一个TMEM accumulator被四个warp一起read back，利用32线程窗口工作、一个2-CTA协作MMA横跨两个CTAs；
* 数据可能在很多地方存在：global memory、shared memory、tensor memory和寄存器提供了不同的tradeoff，比如容量、延迟和access scope (带宽)，同时，cluster存在DSMEM能够让一个CTA去访问另一个CTA的shared memory；一个高性能kernel的核心任务就是在这些不同的空间内移动数据；
* 计算和数据移动被不同的硬件engine处理，cuda core处理地址计算、控制流、scalar逻辑；tensor core实现MMA计算；TMA负责异步搬运数据；我们在这章节最后简单展示一个GEMM的数据pipeline来展示不同的engine怎么通过overlap来同时保证busy；
![[Pasted image 20260729215253.png]]

# 执行层级
GPU不是把所有threads都平铺执行的，他把它们组织到多层中，每个有一些不同的协作关系，下一张图展示不同的Blackwell中的层级；
![[Pasted image 20260729215412.png]]

* Thread：执行的基本单元，每个thread有自己的程序计数以及寄存器 (最多256个)，在warp中，通过lane ID进行标识
* Warp：32个threads，利用SIMT的方式进行执行 (single instruction, multiple threads)，一个warp的lanes一同执行相同的instruction，但每个保留他们自己的寄存器，但也可能各走各的不同逻辑，这也是为什么一个warp的不同lane可能会有分支；
* Warpgroup：四个相邻的warps，128个threads，Hopper使用warpgroup作为单元来触发warpgroup-level的MMA (wgmma)，而在blackwell上，四个warp刚好盖住4个32-lane的TMEM；
* CTA：CUDA通常也叫它block；是硬件调度的基本单元，一个CTA在一个单SM上运行，并且拥有一块私有的SMEM，很多CTA能够同时驻留在同一个SM上，如果这样，那么这个SM上的shared memory会被这么多个CTA给切分；
* Cluster：在不同SMs上存活的协作CTAs，一个cluster的CTAs能够和别的同步，并且可以读其他的SMEM，叫作DSMEM，2-CTA MMA就是利用了这个特性；
# Memory Space
多层thread告诉我们计算是怎么组织的，我们接下来决定数据存放在哪里，一个GPU提供很多内存区域和不同的在容量、延迟和access scope上的tradeoff，一个高性能kernel必须在不同的介质间有效的搬运数据；
![[Pasted image 20260729220440.png]]
TMEM是B卡带来的新的片上存储，早期MMA accumulators存活在寄存器上，当MMA tile增大， 这些accumulator占据了很大的寄存器空间，B的tcgen05直接把MMA结果写到TMEM中，减少寄存器压力；
TMEM包含128行，对应128TMEM lanes，总共512列，每个4字节，共256KB，逻辑上，他们属于CTA，物理上在SM上；
程序显式管理TMEM，一个kernel必须分配和释放他，而且一个epilogue必须显式的的从TMEM把MMA accumulator读取到寄存器中。
# DSMEM
一个cluster能够