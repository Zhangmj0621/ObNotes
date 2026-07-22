# 概述
本文档详解flashattention4的具体实现和设计，其中，将核心分为paper read和code walkthrough两个部分；
# Paper Read
相比fa3主要为Hopper GPU进行优化，通过异步的执行和warp特殊调度，fa4主要针对Blackwell架构进行优化，其中主要面对不对称的硬件性能scaling，分别是，tensor core的吞吐翻倍了，然而其他的功能性组件如shared memory带宽、指数单元增长却很慢，因此，fa4主要提出了如下三个方面的核心创新，分别是
* 重新设计pipeline来完全利用异步的MMA操作以及更大的tile size；
* 软件模拟的指数运算与条件性 softmax 缩放重算，以减少非矩阵乘法操作；
* 利用Tensor内存和2-CTA MMA模式来降低共享内存冲突和backward pass中的atomic add操作；
* 整个fa4是利用CuTe-DSL (Python)实现的，比传统基于C++的方案快20-30倍；
## Introduction
Transformer是近期LLM的核心架构，其中attention非常重要，然后实现高效的attention面对一个不对称的硬件升级，即tensor core的计算能力成倍增长，然而其他组件如shared memory大小/特殊计算单元增长却很慢，导致了pipeline的设计需要很careful；
最开始TriDao提出Flashattention，通过tile和kernel fusion来消除中间到global memory的read/write；在基础上提出的fa2，核心在seq_len维度并行，fa3通过实现细粒度的warp specialization执行、并支持FP8，然而fa3主要优化的还是Hopper架构，在