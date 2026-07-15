本文主要介绍用CUDA实现[矩阵乘法](https://zhida.zhihu.com/search?content_id=215526824&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95&zhida_source=entity)运算（C = A x B）的几个基本方法，帮助大家理解矩阵在GPU上面的运算与CPU上的有何异同，通过实践上手CUDA的优化计算，相比基础方法，**能提速10倍以上**。 本文内容涉及到CUDA矩阵1D运算、2D运算、共享内存、CUBLAS的使用。

# 1 CPU矩阵乘运算

矩阵 C = A x B的数学运算，是[线性代数](https://zhida.zhihu.com/search?content_id=215526824&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0&zhida_source=entity)里面最基本的内容， 计算的基本公式如下：

$$ c_{ij} = \sum_{0}^{wA}{a_{ik} * b_{kj}} $$

矩阵C中每个元素 $c_{ij}$为A的第i行与B的j列进行元素对应相乘再求和。

若：A 宽wA 高：hA；B 宽wB 高：hB；C 宽wC 高：hC 有：

$$ wA=hB; hC=hA; wC=wB; $$

们能够很容易的得到运算部分的代码，如下：

```cpp
for (unsigned int i = 0; i < hA; ++i){
    for (unsigned int j = 0; j < wB; ++j) {
      float Cij = 0;
      for (unsigned int k = 0; k < wA; ++k) {
        Cij += A[i][k] * B[k][j];
      }
      C[i][j] = Cij ;
    }
   }
```

进一步，我们还需要了解矩阵的一维数据运算方式。矩阵的数据在内存中存储的格式是线性格式（行优先/列优先），如下所示，展示的是一种行优先的存储方式。 可以通过索引计算来定位矩阵中的某个元素，比如第i行第j列的元素，在线性内存中的位置： i * w + j。 w为矩阵的宽度。
![[Pasted image 20260408145007.png]]
显然，如果用单线程的CPU运算，该过程的计算时间是

$$ T = hA * wB * wA * deltaT $$

其中hA、wA是矩阵A的高和宽，wB是矩阵B的宽度，deltaT表示每次运算消耗的时间。

由于过程只有一个CPU线程在串行计算，所以矩阵越大耗时越久。为了优化这个过程，我们采用GPU来计算，GPU有大量的线程，通过增加更多的线程来并行计算，降低运算时间。理论上当我们用N个线程来运算时，整个运算时间为：

$$ T = hA * wB * wA * deltaT / N $$

# 2 一维块构建运算

## 2.1 代码基本实现

```cpp
__global__ void MatMulKernel1D(float *C, float *A, float *B, const int wh, const int wC, const int hC)
{
    const int totalSize = wC * hC;
    int thID = threadIdx.x + blockIdx.x * blockDim.x; // 索引计算
    while (thID < totalSize) {
        int Cx = thID / wC;             //数据坐标 与 thread索引的映射
        int Cy = thID % wC;
        float rst = 0.0;
        for (int i = 0; i < wh; i++) {
            rst += A[Cx * wh + i] * B[i * wC + Cy];
        }
        C[Cx * wC + Cy] = rst;
        thID += gridDim.x * blockDim.x;
    }
}
```

## 2.2 共享内存优化

```cpp
template <int shWASize>
__global__ void MatMulKernel1DWithShMem(float *C, float *A, float *B, const int wA, const int wC, const int hC)
{
    __shared__ float sRow[shWASize]; // 定义共享内存的大小
    int blockID = blockIdx.x;
    while (blockID < hC) {
        int thIdx = threadIdx.x;
        while (thIdx < wA) {
            sRow[thIdx] = A[blockID * wA + thIdx];   //数据转移到共享内存
            thIdx += blockDim.x;
        }
        __syncthreads();

        thIdx = threadIdx.x;
        while (thIdx < wC) { // wB = wC;
            float sum = 0.0;
            for (int i = 0; i < wA; i++) {
                sum += sRow[i] * B[wC * i + thIdx];
            }
            C[blockID * wC + thIdx] = sum;
            thIdx += blockDim.x;
        }
        blockID += gridDim.x;
    }
}
```

采用了共享内存后，通过实测会发现，矩阵运算的速度不增反降。其实原因很简单，因为共享内存使用的成本高于其节约的时间。这样我们需要进一步优化，比如采用2D block 并配合共享内存。

- 每个block只算一行，可能浪费thread，打不满GPU
- 只有A放入了shared memory，B仍然还是global memory

# 3 二维块 (2D Block) 优化运算

2D block相比1D block，最大差异是让thread编号由1维变成了2维，在矩阵乘法中，我们可以将矩阵拆成子矩阵，每个block对应计算一个子矩阵；

Cs矩阵的具体运算可拆解为：Cs = As0 x Bs0 + As1 x Bs2 + ... + Asm x Bsm. 如下图所示，我们用宽度为M的方块去分割数据。这样每个小矩阵的大小都是M * M。 那么，为什么要进行分割运算，直接运算不是很简洁？ 实际上就是为了使用共享内存，减少数据的加载次数。上面运算中，例如As0 x Bs0运算由于As0与Bs0矩阵可以足够小，都能加载到共享内存中，每个数据可减少M - 1次全局内存读写。
一般而言M * M设置的大小与CUDA中2D Block的大小一致，这样能够简化运算：
![[Pasted image 20260408145053.png]]
优化的代码关键如下：

```cpp
template <int BLOCK_SIZE> __global__ void MatMulKernel2DBlockMultiplesSize(float *C, float *A, float *B, int wA, int wB)
{
    // ... omit init ...
 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
       // As与Bs 加载到共享内存中:
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        //让As Bs的数据初始化，从原始数据中映射：
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

#pragma unroll
        // 子矩阵的运算数据相加
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    // 最终结果让汇总：
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
 
}
```

```cpp
root@node201:/workspace/infrawaves/zhangmj/BasicCUDA/matrix_multiply# ./matMul wA=1000 hA=312 wB=11 hB=1000
[Matrix Multiply Test] - Starting...

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
MatrixA(1000,312), MatrixB(11,1000)
========================= 1D blocks without shared memory =================
Computing result using MatrixMul1DTest Shared Mem: 0
Warmup  operation done
Performance= 64.65 GFlop/s, Time= 0.106 msec, Size= 6864000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
========================= 1D blocks with shared memory ===================
Computing result using MatrixMul1DTest Shared Mem: 1
Warmup  operation done
Performance= 6.63 GFlop/s, Time= 1.036 msec, Size= 6864000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
========================= 2D blocks with any size ========================
Computing result using MatMul2DTest Kernel. 
Spport any size, e.g. wA=1000 hA=312 wB=11 hB=1000.
Warmup  operation done
Performance= 190.31 GFlop/s, Time= 0.036 msec, Size= 6864000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS
========================= CUBLAS Sgemm kernel ========================
Computing result using CUBLAS Sgemmm Kernel. 
Warmup  operation done
Performance= 909.94 GFlop/s, Time= 0.008 msec, Size= 6864000 Ops,Checking computed result for correctness: Result = PASS
root@node201:/workspace/infrawaves/zhangmj/BasicCUDA/matrix_multiply# 
```