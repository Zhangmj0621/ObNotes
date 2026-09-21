# 概述
本文档概述cutlass persistent kernel 在sm90和sm100架构下的具体实现和区别，本文档主要针对sm100进行详细代码分析，sm90的persistent kernel采用一个producer warpgroup两个consumer warpgroup的实现，且由于没有TMEM的存在，warp会被reigster数量bound，代码的pipeline也相对更简单清晰一点；
# Code walkthrough
sm100的persistent kernel在cutlass的文件sm100_gemm_tma_warpspecialized.hpp中，其中，比起sm90清晰的warpgroup调度粒度，sm100则完全采用了warp粒度的specialized，其中的warp分配如下：
![[Pasted image 20260917150236.png]]
其中，sched, mma, mainloopLoad, epilougeLoad均为一个warp，鉺Epilogue warp则占据128个threads共4个warp，核心原因是因为TMEM为128x512的内存空间，对其操作的时候严格要求4warp对齐，其中，相比于sm90，每个mma warp需要以warpgroup为粒度存在 (wgmma的需求)，sm100的tcgen05指令只需要一个thread就能发射mma操作，因此，其中，只占据了一个warp；
具体而言，每个warp负责的功能如下：
![[Pasted image 20260917150316.png]]
不同的warp间通过pipeline进行mbarrier同步，具体而言的pipeline如下：
![[Pasted image 20260917150331.png]]
其中，详细逐段分析代码如下：
