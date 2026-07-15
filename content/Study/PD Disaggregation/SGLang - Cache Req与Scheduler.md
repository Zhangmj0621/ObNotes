## SGLang KVCache管理模块

cache类定义在_python/sglang/srt/mem_cache.py_下，包含两大结构，基于BasePrefixCache的类(chunked cache, radix cache)以及memory_pool。

- BasePrefixCache：俗称的KVCache实体
- Memory_pool：包含两种映射关系，reqToToken，tokenToKv，前者为生成请求和对应token的映射关系，后者则是生成的token与其KVCache的映射关系

不同的KVCache操作差异(MLA, MHA, double sparsity)有不同的实现

上层结构基本都在manager层，代码目录_python/sglang/srt/managers_，这是一个完整的LLM serving框架，包括tokenizer/detokenizer, session管理, scheduler等，KVCache的核心涉及角色是scheduler以及tp_worker，sglang对于执行逻辑的抽象关系，有过说明

```python
The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
```

推理核心实现model_executor，目录位于_python/sglang/srt/model_executor_下，具体forward前的调度和处理工作集中在model_runner，包括sampling、rope、cuda_graph、kvcache的选择与配置等，通过forward_batch_info.py我们可看到sglang支持常规prefill、decode、prefill with prefix cache外，也支持投机推理（**可能是local prefix cache？**）

真正的