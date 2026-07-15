直接引用kaiyuan的知乎文章中的一些图片，[https://zhuanlan.zhihu.com/p/698447429](https://zhuanlan.zhihu.com/p/698447429)

其中，CP切分仅切sequence_length，然而，seq_len的切分贯穿QKV三个不同的Linear层，即每个CP Rank只会计算各自切分过的Q、K、V矩阵，也即Q、K、V的linear weights直接在rank间做了切分，随后当每个rank计算得到各自的Q，K，V后，采用ring attention的方式获取 $O_0$, $O_1$, …并最终修正得到$O_{ls}$，可以理解为每个rank计算的attention大小为 $[b, sq/3, np, hd]$ ；
![[Pasted image 20260408131311.png]]
# PCP & DCP