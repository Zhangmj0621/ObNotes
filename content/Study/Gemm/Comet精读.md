# Motivation
* 在MoE层中，通信占比开销47%，如何overlap计算和通信很重要；
* 过往工作往往采用粗粒度的overlap，有如下的两个问题：1）将大GEMM切成小GEMM会有损计算效率；2）MoE的负载很随机，每个expert计算量不同，简单把计算通信封装到不同stream伤的独立kernel可能导致不确定性的表现；
* MoE workload有非常复杂的数据依赖和动态的计算通信负载，很难实现有效的overlap；
为此提出了Comet，核心具有两个系统组件：
1） A dependency resolving method：识别计算和通信的数据依赖，实现最优的pipeline组成；
2）An adaptive workload assignment method：根据workload，动态的在计算/通信间分配SM；
Comet核心维护要一个shared tensor作为计算和通信操作的中转buffer，将tile级别的计算和token级别的通信的mismatch消除，同时利用动态的thread block specialization来隔离计算通信的影响；
在实现细粒度计算和通信overlap中，核心存在两个问题，让这件事件intrivival：
* 计算是tile级别的，通信是token级别的，除非收到了完整的tile (可能对应128个token)，不然计算不会开始；细粒度的通信差别很大，如remote通信很慢，但机内走TMA又很快，因此实现细粒度通信overlap很难；
* 计算和通信负载随机，怎么做资源分配很重要；
# Design
核心提出COMET，具有两个核心机制，分别是
* **Shared tensor based dependency resolving**：将shared tensors沿着特定的dimension解耦，来打破粗粒度的数据依赖，如expert0计算token...，expert1计算token...，改为，每个token实际上划分为(expert_id, source_rank)，而不是原先的纯expert_id；随后，利用reschedule，把同属于同一个source_rank的token聚合到一起做GroupGEMM，而不用像原先每个expert的GEMM一个一个算各自的；
* **Adaptive workload assignment**：动态根据workload来给计算和通信分配SM资源；
MoE layer0和layer1的decomposition和reschedule如下图所示：
![[Pasted image 20260811114751.png]]
![[Pasted image 20260811114802.png]]
计算通信的overlap实际上还是利用cuda stream的并行实现，核心数据交互用shared tensor (GMEM)实现
![[Pasted image 20260811115025.png]]
