# 1 threads相关概念

CUDA里面用Grid和Block作为线程组织的组织单位，一个Grid可包含了N个Block，一个Block包含N个thread。

相关单位参数：

- gridDim: blocks在grid里面的数量维度；dim3;
- blockDim: threads在一个block的数量维度；dim3；
- blockIdx: block在grid里面的索引；dim3；
- threadIdx：thread在block里面的索引;dim3;

gridDim，规定blocks的形状，blockDim规定了threads的形状。 这些参数在kernel写好后，**global**/___device__定义的函数里面能够直接访问到变量。我们在创建kernel时，通常会传入blocks的维度、threads维度，示例如下

```cpp
__global__ void kernel(float *src){
   //do something
}

// 定义grid 和block的形状：
dim3 BlocksPerGrid(N, N, N);  // gridDim 对应gridDim.x、gridDim.y、gridDim.z
dim3 threadsPerBlock(M, M, M);  // blockDim 对应blockDim.x、blockDim.y、blockDim.z
// invoke code:
kernel<<<BlocksPerGrid, threadsPerBlock>>>(*src);
```

下图给出了一个实例，其中gridDim .x=2, .y=2, .z=3; blockDim .x=4, .y=2, .z=4

这样，一共包含的block数量：2**2**3 = 12；每个block线程总数：4**2**4 = 32； 线程总数：12 * 32 = 384
![[Pasted image 20260408144933.png]]
若要索引一个threads，可通过索引坐标, 如图标记为蓝色的线程，其索引表示方式：blockIdx的 x=1, y=0, z=2; threadIdx的x=3, y=0; z=3

而线程的全局idx的求解，还需要按照坐标计算公式求得：

# 2 坐标计算公式

现在需要让线程通过全局索引拿取各自的数据，则需要通过公式转换计算求得。首先需要知道全部3D的坐标下的idx如何计算。3D3D的索引计算即找出一个线程在所有线程中的位置。由于线程的组织结构包含了两层，所以可以拆分计算：

- step1: 计算线程thread在block的位置：
- step2: 计算该block在grid中的位置；
- step3: 计算block有多少线程；求解位置索引。

## 2.1 3D结构

$$ threadInBlock = threadIdx.x + threadIdx.y_blockDim.x + threadIdx.z_blockDim.x*blockDim.y $$

$$ blockInGrid=blockIdx.x + blockIdx.y_gridDim.x + blockIdx.z_gridDim.x*gridDim.y; $$

$$ oneBlockSize = blockDim.x_blockDim.y_blockDim.z $$

$$ idx = threadInBlock + oneBlockSize*blockInGrid $$

## 2.2 2D结构

```cpp
__global__ void kernel2D2D(float *input, int dataNum)
{
    // int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    // int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
    // int idx = threadInBlock + oneBlockSize*blockInGrid;
    // when:
    // threadIdx.z = 0; blockIdx.z = 0;
    // blockDim.z = 1; gridDim.z = 1;
    // then:
    // int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x;
    // int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x;
    // int oneBlockSize = blockDim.x*blockDim.y;
    int idx = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*(blockIdx.x + blockIdx.y*gridDim.x);
    // thread overflow offset = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
}
```

## 2.3 1D结构

令 threadIdx.y = 0; threadIdx.z = 0; blockIdx.y= 0; blockIdx.z = 0; blockDim.y = 1; blockDim.z = 1; gridDim.y = 1; gridDim.z = 1; 带入3D公式中简化得到1D计算：

```cpp
__global__ void kernel1D1D(float *input, int dataNum)
{
    // int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    // int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    // int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
    // int idx = threadInBlock + oneBlockSize*blockInGrid;
    // when:
    // threadIdx.y = 0; threadIdx.z = 0; blockIdx.y= 0;  blockIdx.z = 0;
    // blockDim.y = 1; blockDim.z = 1; gridDim.y = 1; gridDim.z = 1;
    // then:
    // int threadInBlock = threadIdx.x;
    // int blockInGrid = blockIdx.x;
    // int oneBlockSize = blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // thread overflow offset = blockDim.x*gridDim.x;
}

```