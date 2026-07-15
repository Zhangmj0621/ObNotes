tensor core和cuda core是没有训练和推理任务的区别，他们都是在进行运算，至于是不是训练/推理他们并不感知。这个问题要解释清楚，需要点基础知识。分几个方面：

1、用户是怎么样使用到[NVIDIA显卡](https://zhida.zhihu.com/search?content_id=359873869&content_type=Answer&match_order=1&q=NVIDIA%E6%98%BE%E5%8D%A1&zhida_source=entity)的计算资源（core）；

2、训练和推理的区别在哪；

2、tensor core和cuda core的相关硬件架构知识。

先说**问题1**，我们怎么样调用到显卡的计算资源的，对于一般使用者来说，路径如下（只说AI相关的应用）：

```cpp
用户代码  -> AI框架（PyTorch/Tensorflow/Caffe等）-> CUDA lib -> Driver -> 显卡
```

在这个链路中，一般是不会感知到tensor core和cuda core的区别的，因为有框架这一层帮你做好了封装；框架是调用CUDA的库完成的运算，至于调用什么core完成，CUDA lib已经设计好了，用户不用操心。当然，要是想成为高级用户，自己开发运算算子，要看CUDA 库，如CUDA runtime、CUDA driver：

那对于使用者来说感受到区别在哪？**算力 。**具体来说，就是训练、推理的时间的差异。 至于是用cuda core算的还是tensor core算的，就无需关注了。

**问题2：**训练和推理的运算的区别。在大部分情况下训练与推理差异如下：

- 训练过程： 前向传播 -> 后向传播 -> 梯度更新。（迭代重复）
- 推理过程： 前向传播 。 （迭代重复）

推理只是完成了训练的一部分内容，当然推理还可以剪枝、压缩让前向传播更快。但总体而言完成一次迭代（同batch_size），训练需要的运算量更多，但前向传播的底层运算基本相同。

**问题3：**tensor core和cuda core 的概念

tensor core和cuda core 都是运算单元，是硬件名词，其主要的差异是**算力**和**运算场景**

**场景**：cuda core是全能通吃型的浮点运算单元，tensor core专门为深度学习矩阵运算设计。

**算力**：在高精度矩阵运算上 tensor cores吊打cuda cores。

先来看一下core里面有什么吧，上一张十年前的设计图：

![image.png](attachment:004cb85a-162e-4673-ae27-7d4ebce999fe:image.png)

一个cuda core 包含了一个整数运算单元integer arithmetic logic unit (ALU) 和一个浮点运算单元floating point unit (FPU)。然后，这个core能进行一种fused multiply-add (FMA)的操作，通俗一点就是一个加乘操作的融合。特点：在不掉精度的情况下，**单指令**完成乘加操作，并且这个是支持32-bit精度。

更通俗一点，就是深度学习里面的操作变快了；

虽然cuda core可以分工干活，但是对于一些场景，比如混合精度的矩阵操作，core算得似乎不是那么高效，于是NVIDIA就开始琢磨专门针对tensor的硬件单元了：

**tensor core 第一代是在[volta架构](https://zhida.zhihu.com/search?content_id=359873869&content_type=Answer&match_order=1&q=volta%E6%9E%B6%E6%9E%84&zhida_source=entity)上推出，**专为深度学习而设计的。也就是说他在tensor矩阵场景下算得更快，先上一张图片感受一下，在4X4 矩阵乘法运算时的差异：

![image.png](attachment:57d9ad87-ac81-4deb-88cc-9a1e38002b36:image.png)

<aside> 💡

> 通过 FP16 和 FP32 下的混合精度矩阵乘法提供了突破性的性能 – 与 NVIDIA Pascal 相比，用于训练的峰值 teraFLOPS (TFLOPS) 性能提升了高达**12 倍**

</aside>

首先，这个Tensor Core基础能力在于，**一个时钟周期内可以完成一个64 flating point 的FMA**，而cuda core是搞不定的，分多次。

其次，Tensor Core也能堆叠，V100上面就堆了640个。而且tensor core经过了几次升级，其操作的精度更加丰富了：