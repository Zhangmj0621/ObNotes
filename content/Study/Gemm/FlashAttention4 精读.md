# 概述
本文档详解flashattention4的具体实现和设计，其中，将核心分为paper read和code walkthrough两个部分；
# Paper Read
相比fa3主要为Hopper GPU进行优化，通过异步的执行和warp特殊调度，fa4主要针对Blackwell架构进行优化，其中主要面对不对称的硬件性能scaling，分别是，tensor core的吞吐翻倍了，然而其他的功能性组件如shared memory带宽、指数单元增长却很慢，因此，fa4主要提出了如下三个方面的核心创新，分别是
* 重新设计pipeline来完全利用异步的MMA操作以及更大的tile size；
* 软件模拟的指数运算与条件性 softmax 缩放重算，以减少非矩阵乘法操作；
* 利用Tensor内存和2-CTA MMA模式来降低共享内存冲突和backward pass中的atomic add操作；
* 整个fa4是利用CuTe-DSL (Python)实现的，比传统基于C++的方案快20-30倍；
## 1/2 Introduction
Transformer是近期LLM的核心架构，其中attention非常重要，然后实现高效的attention面对一个不对称的硬件升级，即tensor core的计算能力成倍增长，然而其他组件如shared memory大小/特殊计算单元增长却很慢，导致了pipeline的设计需要很careful；
最开始TriDao提出Flashattention，通过tile和kernel fusion来消除中间到global memory的read/write；在基础上提出的fa2，核心在seq_len维度并行，fa3通过实现细粒度的warp specialization执行、并支持FP8，然而fa3主要优化的还是Hopper架构，在Blackwell上并不是最优，下面提供一些具体的Blackwell和Hopper的差别
* Tensor core：Blackwell相比Hopper两倍了tensor core吞吐，BF16 MMA 吞吐==8192 ops/clock/SM==，Hopper 4096，这个值可以从理论最大FLOPS中计算得出，2.25PFLOPS/1850Mhz clock speed / 148SMs = 8192 ops/clock/SM；
* Exponential unit，B200/GB200和Hopper一致，都是16 opus/clock/SM，B300/GB300将这个值提升到32 ops/clock/SM；
* SMEM：Blackwell和Hopper保持一致，shared memory读写吞吐都是128 bytes/clock/SM；
同时，Blackwell提供额外的架构创新，保持扩每个SM新增**256KB的TMEM (Tensor memory)** 来直接储存中间的tensor core结果，MMA的**tiles从Hopper的64 x 128提升到了 128 x 128**，并直接给**TMEM新增了完全异步的Tensor core操作**；这些值为后续理论分析attention计算的bottleneck提供重要支撑；
具体的Attention forward和backward计算示意图如下：
![[Pasted image 20260728173904.png]]
![[Pasted image 20260728173913.png]]
## 3 Algorithm
### 3.1 Algorithm Forward Pass
本节系统分析目前在Blackwell架构上的系统pipeline bottleneck，并详细介绍为什么会提出一系列的flashattention的优化改进；
#### 3.1.1 Feeds and speeds
在attention计算时，设计算Q和K的tile dimension为M x N，设置head dimension为d，随后分别分析MMA计算 /读写SMEM流量 /指数单元计算 (Softmax)的总时间开销
* MMA计算：attention计算时核心存在两个MMA操作，分别是$QK^T$ (MxN output, Mxd and dxN)和 $PV$ (Mxd output, MxN and Nxd)，每个MMA需要2MND浮点数操作 (乘一次，加一次)，考虑到前面介绍的8192 FLOPS/cycle，因此执行时间为 ==$T_{MMA} = (4MND) / 8192    \text{cycles}$；
* SMEM流量：对于两个MMA操作，第一个操作Q和K都需要从Shared memory读，因此具体要read M/128 x N/128 x 256d，对于PV操作，P在TMEM内，因此没有访寸开销， 此时总开销位 M/128 x d/128 x 128N，因此考虑到每元素为BF16，读写速度为128 bytes/clock/SM，因此总时间为 ==3MND/8192 cycles==；
* 指数单元：指数单元需要在M x N总操作上做softmax，考虑到B200上该数值为 ==MN/16 cycles==；
对于常见的head dimension=128，MxN为256x128和128x128，具体的forward pass的数据如下
![[Pasted image 20260727212045.png]]
因此可以看到，这给我们kernel设计的时候有三个如下的motivation
1. 需要更大的tile size和最大化overlap MMA计算操作和Softmax 操作；
2. 需要增大指数单元的吞吐，利用其他的硬件单元；
3. 减少不必要的non-matmual操作
#### 新pipeline来overlap matmul和softmax
由于tensor core能力翻倍，因此在Blackwell上，如何更好的把tensor core操作和softmax操作做overlap比Hopper更重要，整体上仍然采用类似FA3的==ping-pong==的方式，two tile的output互相overlap。一个tile执行MMA操作，另一个tile则执行softmax操作，Hopper会把accumulator结果放在寄存器中，而Blackwell会把他放在TMEM中，另外，Blackwell的tile是128x128大小，而Hopper的tile是64x128大小；
因此，一个自然的方法也就是利用2个总共有128 threads的warpgroup来处理softmax的每一行，我们这里叫他softmax warpgroup，和FA3，这里在两个softmax warpgroup的关键段落处同步来保证他们不要同时去执行softmax操作，==因为指数单元能力固定，两个并行执行不如串行执行==，能够让某个tile尽快进入下一个stage。
和FA3最大的区别在于，P是放在TMEM内，而不是放在寄存器里，==因而采用单独的一个correction warp来对O做rescale==，因为P的存储不再处于critical path上了；
```py title=""
Softmax WG:
计算新 row max
      ↓
得到 scale α
      ↓
继续计算并写出 P_j
          ──────────────────────┐
                                │ 并行
Correction WG:                  │
TMEM 读取旧 O                   │
      ↓                         │
寄存器中分块执行 O ← αO         │
      ↓                         │
写回 TMEM                       │
          ──────────────────────┘
                    ↓
MMA WG 执行 P_j V_j，并累加到 TMEM 中的 O
```
为了实现这样的pipeline overlap，如何管理TMEM的内存分配非常重要，因为每个SM只有256KB的TMEM大小，首先Output必须得放在TMEM里，格式为FP32，元素大小为128x128x4x2=128KB，因而，O就要占据TMEM中一半大小的空间，剩下的S和P在TMEM中的存储只能放剩下的一半，S格式是FP32，P是BF16，因此每个S要占据64KB，每个P要占据32KB，因此128KB可以放得下2个S或者4个P；我们有两种做法来存放S和P，一种是放一个S和2个P，==一种是放2个S和P做overlap==，显然采用第二种，一个可能的TMEM示例图如下，由于一部分S算完以后就能腾出空间放P，而S的大小要大于P，因而显然，TMEM中还能剩余一部分空间来给correction warpgroup通信rescale信息。
```py title=""
TMEM:
┌──────────┬──────────┬────────────────┬────────────────┐
│ Oᴴ FP32  │ Oᴸ FP32  │ Sᴴ → Pᴴ 复用区 │ Sᴸ → Pᴸ 复用区 │
└──────────┴──────────┴────────────────┴────────────────┘

启动：
MMA WG        : QᴴKᵀ → Sᴴ     QᴸKᵀ → Sᴸ

并行：
Softmax WG-H  : load Sᴴ → max → exp → 分段写 Pᴴ
Softmax WG-L  : load Sᴸ → max → exp → 分段写 Pᴸ
Correction WG : 根据统计量 rescale O
MMA WG        : P 前 3/4 ready 后提前开始 PV
```
另一个Blackwell tile size的issue是，显然，计算max时，寄存器需要把完整的128个元素放在寄存器里 (每个thread)，由于同时存在四个warpgroup，两个softmax warpgroups、1个correction warpgroups和一个触发tensor core和TMA单元的warpgroup，因此分配充足的寄存器给softmax很重要；对于BF16的input，需要128个input寄存器和64个output寄存器 (S是FP32，P是BF16），每个thread最多只有256个寄存器，此外还要保存别的如旧 max 和 rescale factor、exponentiation 临时变量、多项式 exp 的中间值等一系列结果到寄存器中，因而 寄存器很容易逼近上限。如果寄存器不足发生 spill，数据会被放到 local memory，最终通常落入 L1/L2，性能会严重下降。
因此，fa4采用**stage out storing P**，具体做法是，P未按照dimension维度，切成3/4和1/4两个部分，前面3/4先算P，算完以后可以直接触发MMA操作，把P从寄存器搬到TMEM中，因此，不需要全量在寄存器中放所有128个输出P，也不需要完整的放128个S，而是流式的把S从TMEM搬到寄存器，然后算P，算完96个，就先搬走到TMEM中算MMA；
#### 3.1.3 Emulation of the exponential function
指数操作一般是MUFU单元来做的，指数单元能力 (16)比MMA能力 (8192)差特别多，因此，指数操作是attention kernel中的一个bottleneck。本节的核心是一部分元素继续使用硬件 MUFU.EX2，另一部分元素使用普通浮点 FMA 指令，通过多项式近似计算指数，从而同时利用两类执行单元。GPU 的硬件指令通常是 MUFU.EX2，直接计算：$2^x$，因此，实际上是先计算$x=(S_{ij}-m_i)\log_2 e$，再计算$2^x$，注意此处softmax已经减去了行最大值，因此不需要担心正方向溢出。
本论文采用==经典的cody-waite range reduction==，核心分解非常简单，即$x=n+f$，其中$n=\lfloor x\rfloor,\qquad f=x-\lfloor x\rfloor\in[0,1)$，于是有$2^x=2^{n+f}=2^n2^f$，因此原问题转化了两个更好计算的子问题 1）计算整数次幂；2）计算$2^f$：只需要在固定区间$f\in[0,1)$上近似。IEEE 754 FP32 的正规浮点数可以写为：$(-1)^s \times 2^{E-127} \times(1.M)$，其中E是8位指数域，127是FP32的exponent bias，M是尾数域，对于纯粹的 $2^n$，有效数字正好是1.0，尾数全0，因此其指数域只需要写为E = n + 127，对应的FP32 bit pattern位 (n+127) << 23 (23即最后的尾数全0)，例如 $2^3=8.0$，FP32指数域是 $3+127=130$，因此不需要真正执行指数函数，只需要构造指数位。而对于浮点数位，由于$f\in[0,1)$，因而计算目标$2^f$一定$\in [1, 2)$ ，所以q的指数位一定是127，因而bit pattern可以表示为 $\operatorname{bits}(q)=(127\ll23)+\operatorname{mantissa}(q)$，于是有$\operatorname{bits}(2^nq)=((127+n)\ll23)+\operatorname{mantissa}(q)$，于是就有$\operatorname{bits}(2^nq)=\operatorname{bits}(q)+(n\ll23)$，因而，最终的浮点数计算其实不需要额外的浮点乘法，可以采用如下三步骤来解决：
* 把q reinterperet 成整数
* 给exponent field加上n
* 在reinterpret回FP32
为了快速计算整数部分n，==引入了magic bias==：$C=2^{23}+2^{22}=1.5\times2^{23}$，即$C=12582912$，然后通过指定向下舍入模式完成$n=(x+C)-C$，这里利用了FP32尾数有23个显式fraction bits的特点，当给一个较小的x加上很大的C后，结果所在数量级的ULP约为1，因此自然就保留下来的就是n了 (小数点被23位給mask掉了)；
关于小数部分计算，核心采用多项式：$2^f \approx p_0+p_1f+p_2f^2+\cdots+p_nf^n$，其中p0为1，且为了减小误差，这些系数都不是简单的泰勒展开，而是使用Sollya工具求出的优化系数，目标是在整个$[0, 1)$区间最小化相对误差，即，它更接近minimax polynomial。
在此基础上，进一步采用==Horner方法==优化计算，直接计算$p_0+p_1f+p_2f^2+p_3f^3$较慢，需要乘法构造$f^2,f^3$，而Horner形式可以写成$p(f)=((p_3f+p_2)f+p_1)f+p_0$，这样相当于只用做3次FMA操作就行了。
论文通过实际数据证明degree-3的误差已经够用：具体的FP32原始误差如下
| 方法         |              最大相对误差 |              平均相对误差 |
| ---------- | ------------------: | ------------------: |
| `MUFU.EX2` | $(1.41\times10^{-7})$ | $(3.04\times10^{-8})$ |
| degree 3   | $(8.77\times10^{-5})$| $(5.43\times10^{-5})$ |
| degree 4   | $(3.05\times10^{-6})$ | $(1.84\times10^{-6})$ |
| degree 5   | $(1.44\times10^{-7})$ | $(5.48\times10^{-8})$ |
可以看到，degree3确实远不如MUFU.EX2，然而，Flashattention的需要把P转换成BF16，此时光光BF16的误差就达到了3.9x10^{-3}，因此degree3的误差比BF16自身的量化误差要小44倍，因此在degree3下，degree的结果再约99%输入上 与硬件结果相差不超过1个BF16 ULP；
Partial emulation：虽然走多项式估计能够节约MUFU指令，然而，他会带来额外的寄存器开销、更大的寄存器带宽花费、更高的延迟，因而，只对10%-25%的entry做多项式估计，剩余的entry仍然通过MUFU.EX2进行计算，==具体的fraction根据经验进行微调==，在给定的tile配置下根据MMA和指数单元的吞吐。
#### 3.1.4 Skipping online softmax rescaling
直接切换rescale为如下的公式，仅当rescale差距超过256的时候，才进行更新一次，不然就延迟更新，能够大幅度减少rescaling操作。在实际运行中，为了避免warp内的threads多样性，不让他们走不同分支，只要32个threads有任何一个线程需要resacle，整个warp的的所有线程都执行rescale。
![[Pasted image 20260728172427.png]]
### 3.2 Attention backward pass
#### 3.2.1 Feeds and Speeds
和forward pass类似，我们首先提供指导关于我们的kernel设计和优化的motivation，基于具体计算出的tensor core, smem访问和指数单元的所需时间；
* MMA计算：backward pass需要5个MMA操作，每个MMA包括一个MxN的matrix，一个Mxd的matrix和一个dxN的matrix，需要2MND的浮点数操作，总共10MND的浮点数操作，因此需要时间为 10MND / 8192 cycles。
* SMEM流量：其中三个操作= KQ⊤, dP⊤ = VdO⊤, and dQ = dSK是shared-shared操作，而剩下两个操作dV = P⊤dO and dK = dS⊤Q是Tensor-shared操作，因而Shared memory总带宽为 (4md+3ND+MN) / 64 cycles，同时，考虑到算法还有额外把立即数梯度dDS (MxN)写到Shared memory，
* 指数单元