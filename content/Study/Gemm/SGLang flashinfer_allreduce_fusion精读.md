# 概述
本文档概述SGLang flashinfer_allreduce_fusion的触发位置与具体操作；
# Code walkthrough
## 参数解析
在server_args中，核心定义如下的两个变量，来实现TP Allreduce的fusion，其中，enable_flashinfer_allreduce_fusion已经被deprecate，本质上是设置flashinfer_allreduce_fusion_backend为auto；
![[Pasted image 20260812173133.png]]
随后，在`_flashinfer_allreduce_fusion_auto_enable(..)`函数中，给特定的白名单case直接赋值flashinfer_allreduce_fusion_auto_enable