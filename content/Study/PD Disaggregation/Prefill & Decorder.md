# Definition

prefill和decorder是推理的两个不同阶段

- prefill：用户输入完prompt到生成首个token的过程，也即我们所说的TTFT
- decode：生成首个token到推理停止的过程，也即TBT

可以吧prefill理解成阅读理解阶段，一次性处理整个问题，把decode理解为写作过程，一个词一个词回答
![[Pasted image 20260408135016.png]]
# Difference between Prefill & Decode

**Prefill阶段:** 由于能够并行处理所有输入token，主要受限于GPU的计算能力，优化重点在于最大化利用GPU的并行计算核心和高效处理批量请求。

**Decode阶段:** 面临的是自回归生成的串行限制，无法像Prefill那样充分利用GPU的并行计算能力，其性能主要受限于内存带宽和延迟。

# 优化策略

## Prefill

### Batching

批量处理(Batching)是一种常见的优化手段，通过将多个用户的请求组合成一个batch进行处理，从而提高GPU利用率和吞吐量（增加并行度）

**静态批量处理（Static Batching）**：将同时到达的所有推理请求组成一个批次，然后一次性进行prefill计算。这种方法能有效利用GPU的并行计算资源，提高整体吞吐量，但可能导致处理时间短的请求需等待批次中最慢的请求完成，增加延迟。

**连续批量处理（Continuous Batching）**：也称为动态批量处理或飞行中批量处理，在推理的迭代层面进行batch管理。当批次中某个请求完成后，可以立即用新请求替换它，无需等待整个batch完成。这种方法更有效地利用GPU资源，减少空闲时间，并可能降低平均延迟。

**最大化解码批量处理（Decode-Maximal Batching）**：将一个prefill请求分割成等大小的块，然后将一个prefill与多个decode请求组合成一个批次处理。这种方法允许从单个prefill请求创建多个batch，优化decode请求的处理，但可能增加decode延迟。

### TP(张量并行, tensor parallel)

将模型计算负载分布到多个GPU上，通过将模型各层（如transfromer中的self-attention和SVM）分割成更小计算快，这些快可以在不同GPU上运行

## Decode

Decode阶段由于串行性（单个token单个token输出）以及对内存带宽高度依赖，需采用不同优化策略，主要目标是减少内存访问延迟，避免计算冗余，并降低每输出token的时间(TPOT)和整体延迟

### KVCache

KVCache对decode非常重要，在自回归decode阶段，每生成一个新的token，模型都需要基于包括当前token在内的所有先前token计算KV值（self-attention)。KVCache通过将先前已处理的token的KV值存储在内存中。避免重复计算，模型**只需要计算新生成的token的KV值**
![[Pasted image 20260408135029.png]]
**为什么不需要Q cache？**

为什么加速LLM推断有KV Cache而没有Q Cache？ - 风之魔术师的回答 - 知乎 [https://www.zhihu.com/question/653658936/answer/3545520807](https://www.zhihu.com/question/653658936/answer/3545520807)

KVCache虽然能减少计算冗余，但会带来额外内存开销，因为需要存储所有先前token的KV值。KV缓存大小会随序列长度、注意力层数、dimension和batch_size增加而线性增加

### Attention优化

[**FlashAttention**](https://zhida.zhihu.com/search?content_id=255117291&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity)：一类优化的注意力计算方法，通过改进内存访问模式和减少冗余计算提高效率。这种优化方法能显著加速注意力计算，特别是在处理长序列时更为明显。

**多查询注意力**（[Multi-Query Attention](https://zhida.zhihu.com/search?content_id=255117291&content_type=Article&match_order=1&q=Multi-Query+Attention&zhida_source=entity), MQA）：通过在多个注意力头之间共享键和值投影矩阵，有效减少模型参数数量和KV缓存占用，从而加快解码速度并允许处理更大批次。