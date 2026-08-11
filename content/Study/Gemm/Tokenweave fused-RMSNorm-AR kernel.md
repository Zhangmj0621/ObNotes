# 概述
本文档详解tokenweave中fused-RMSNorm-AR kernel的具体实现；
# Code walkthrough
首先查看tokenweave中的具体fused kernel代码，以及对应的一些常见的multimem和同步的代码。
首先直接查看定义在csrc/tokenweave_fused_kernels.cu中的向上提供的接口`fused_rs_ln_ag_cta(...)`，该函数通过torch bindings直接提供给上层python api使用，其中，定义block 数量为MAX_CTAS，默认值为8，定义单个block的大小为1024 (32个warp)，随后，核心调用LAUNCH_FUSED_RS_LN_AG_CTA也即fused kernel `fused_rs_ln_ag_cta_kernel(...)`。
直接来看该fused kernel的实现，代码非常浅显易懂，其中，计算vec_hidden_size，即最多需要多少次搬运操作 (multimem一次性最大搬运128也即16字节，对于BF16来说，一次能搬运8个元素，widths即代表一次性搬运几个元素)，随后，计算tokens_per_iter，即每个block负责多少个token，然后，调用 `sync_remote_blocks(...)`进行同步，本质调用`put_signal(...)`和`wait_signal(...)`以及底层的PTX cas来确保同一个symmetic memory group的rank的同号SM会同步进入kernel，保证后续reduce操作的正确性，注意，开始时的cas均是Relaxed，因为kernel刚启动，不存在所谓的内存读写；随后，对本Thread block负责的token，创建寄存器变量variance和SMEM s_variance，每个thread累加计算各自的variance，随后调用blockReduceSum (经典块内规约实现) device function进行reduce操作，此时第一个warp的variance已经变成了原先所有元素的累加和，因此对于0号thread，直接做一次计算，将最终结果放入SMEM中，随后同步一次，随后，对结果计算最终的Normalization和施加weight，并调用multimem_st将结果写会symmtric memory中，最后，调用`sync_remote_blocks(...)`，这一次，对于`put_signal`来说是Release，要求在这个操作前的所有写操作必须在这个signal前完成，对`wait_signal`来说则是Acquire，保证在这个之后的所有读操作，都在这个signal之后，保证了数据的读写一致性；
```py title=""
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ input,        // [..., hidden_size]
    scalar_t *__restrict__ mcptr,        // [..., hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    uint32_t **signal_pads,
    size_t rank,
    size_t world_size,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{

  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ input_v = reinterpret_cast<vec_t *>(input);
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);
  int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

  #pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++)
  {
    int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens)
      continue;
    float variance[1] = {0.0f};
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    int offset = token_id * vec_hidden_size;
    int offset_scalar = token_id * hidden_size;
    auto input_o = input_v + offset;
    auto residual_o = residual_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      vec_t temp = *(reinterpret_cast<vec_t *>(&mtemp));
      temp += residual_o[idx];
      variance[0] += temp.sum_squares(); // FP32 accumulation
      residual_o[idx] = temp;
    }

    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0)
    {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx)
    {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}
```
可以看到，fused kernel本身非常的简单易懂，核心这里用到的两个PTX指令就是用来做同步的cas以及multimem_st/ld相关的PTX指令。
下面代码是经典的块内规约：
```py title=""
template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T *val)
{
#pragma unroll
  for (int i = 0; i < NUM; i++)
  {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T *val)
{
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T, NUM>(val);

  if (lane == 0)
  {
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++)
  {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}
```