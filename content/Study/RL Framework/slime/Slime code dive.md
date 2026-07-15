## 1. 概述

本文主要详解slime启动脚本train.py与train_async.py，[本文着重详解train.py](http://xn--train-xx0k68g3u0c3b9aehe4st.py)

## 2. 主干pipeline
![[Obsidian 2026-04-08 14.02.29.png]]
## 3. 分配创建资源

在`create_placement_groups(…)`函数中，主要负责ray资源的预留与分配，其中根据具体不同的模式分配资源，例如对于colocate模式，rollout与actor共用资源，而对于disaggregated模式，则两者使用不同的资源，根据offset决定rollout资源的开始索引

```python
def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    // ... ...  
```

其中`_create_placement_group(…)`中，利用info_actors首先分配了num_gpus数量的bundle，每个bundle占有一个GPU和一个CPU，随后根据他们的ip与gpu index进行排序，尽可能吧相同节点的GPU排到一起，随后释放info_actors占用资源，根据offset记录分配的actor gpu资源与rollout gpu资源

```python
def create_placement_groups(args):
    """Create placement groups for actor and rollout engines."""
    
    // ... ...
    
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)

    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]

    return {
        "actor": (pg, actor_pg_reordered_bundle_indices),
        "rollout": (pg, rollout_pg_reordered_bundle_indices),
    }
```

随后初始化wandb相关信息，随后根据先前分配给actor和rollout的gpu资源信息，调用`create_actor_group(…)`与`create_rollout_manager(…)`启动actor和rollout

在`create_actor_group(…)`中，根据对应的训练backend选择对应的TrainActor，例如对于megatron后端而言则是MegatronTrainRayActor，通过ray.remote()统一封装为TrainRayActor，并根据world_size的数量调用T`rainRayActor.options(…).remote(…)`，其中world_size的数量即为actor的gpu数量，也即 self._num_nodes * self._num_gpus_per_node。

注：其中有一个非常有意思的点，在options选项中指定num_gpus=num_gpus_per_actor，该参数在前面被指定为0.8，观察发现rollout时则被指定为0.2，则猜想该设计原因应该主要是为了针对colocated架构，并未rollout和actor分别分配0.2与0.8的GPU推荐显存防止OOM

```python
def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, wandb_run_id):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        wandb_run_id=wandb_run_id,
        num_gpus_per_actor=0.8,
    )
```

`create_rollout_manager(…)`则主要负责rollout部分的初始化工作，主要流程图如下：
![[Pasted image 20260408140243.png]]
其中，在RolloutController创建中，会生成RolloutDataSourceWithBuffer对象，并且根据自定义的路径定义generart_rollout, eval_generate_rollout和custom_reward_post_process_func函数，下面看下RolloutDataSourceWithBuffer函数

## 3. DataSourceWithBuffer

核心代码在slime/ray/rollout_data_source.py中，其中主要定义两个核心类RolloutDataSource和RolloutDataSourceWithBuffer，负责为rolloutEngine提供数据，下图展示获取数据的完整流程
![[Pasted image 20260408140253.png]]
其中RolloutController调用`get_sample(…)`从Data source处获得返回的sample，随后调用`generate_rollout(…)`进行rollout生成；

### RolloutDataSource

- init()：初始化中会定义是否启用rollout_global_dataset，默认为True，从—prompt-data中获取真实数据集，不然则设置self.dataset=None，用户可自行维护。
    
- get_samples():
    
    ```python
     def get_samples(self, num_samples):
            samples = []
    
            # TODO unify the two branches
            if self.dataset is not None:
                if self.sample_offset + num_samples <= len(self.dataset):
                    prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                    self.sample_offset += num_samples
                else:
                    prompt_samples = self.dataset.samples[self.sample_offset :]
                    num_samples -= len(prompt_samples)
                    self.epoch_id += 1
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(self.epoch_id)
                    prompt_samples += self.dataset.samples[:num_samples]
                    self.sample_offset = num_samples
                for prompt_sample in prompt_samples:
                    group = []
                    for _ in range(self.args.n_samples_per_prompt):
                        sample = copy.deepcopy(prompt_sample)
                        sample.index = self.sample_index
                        self.sample_index += 1
                        group.append(sample)
                    samples.append(group)
            else:
                for _ in range(num_samples):
                    group = []
                    for _ in range(self.args.n_samples_per_prompt):
                        sample = Sample(
                            index=self.sample_index,
                        )
                        self.sample_index += 1
                        group.append(sample)
                    samples.append(group)
    
            return samples
    ```
    
    根据num_samples取出对应的所需sample(batch)，如果到达边界，则取最后剩余部分，并根据所需数量从头开始取样，并记录epoch++，同时支持rollout_shuffle；随后，为sample复制n_sample_per_prompt数量，为GRPO服务
    
- save & load:
    
    save保存目前取样的情况，包括sample_offset与epoch_id等等
    
    load则是从save的取样情况读取，确保数据一致性

### RolloutDataSourceWithBuffer

- init: 通过设置buffer_filter_path指定选取数据策略，默认为pop_first
- get_samples: 优先从buffer获取数据，如果不够则从原始数据集补充
- add_samples：将sample添加至buffer中，添加时保证每个prompt有n_samples_per_prompt个sample

## 4. RL主流程

主要关注`async_generate(…)`是如何生成prompt所需的response的完整流程，其中关注multi-turn与tool-call方式；

`aysnc_generate(…)会`调用`RolloutController.generate(…)`函数，在其中，会调用`generate_rollout(…)`函数获取实际的输出，并且如果获取的data长度不能整除global_batch_size，则trim多余的尾部数据，随后调用`_convert_samples_to_train_data`将sample转换成满足训练的格式；

其中`generate_rollout(…)`为在default配置下，为定义在路径/slime/rollout/sglang_rollout.py函数下的generate_rollout；

### generate_rollout_async

下面详解该函数中的完整逻辑，其中首先定义核心对象GenerateState，用来管理生成状态，其中包括sampling_params等等

```python
async def generate_rollout_async(args, rollout_id: int, data_source) -> list[list[Sample]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        list[list[Sample]]: a list of samples generated by the rollout, the length of the list is exactly the same as the `rollout_batch_size`
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)
    // ... ...
```

记录target_data_size与dynamic_filter等值，进入while循环，直到获取的data总长度达到target_data_size，即rollout_batch_size

注意，其中每有一个group完成(一个prompt的所有sample，共n_samples_per_prompt个），则打印第一个group的第一个rollout sample，随后判断是否需要dynamic_filter，并将group添加至data list中

当获取满足要求的data后，输出最后一个prompt的第一个sample，并abort剩余还在生成的多余请求(over_sampling_batch_size ≥ target_data_size)，根据每个group的第一个prompt的index进行排序

### generate_rollout_async

在generate_rollout_async中，主要通过在submit_generate_tasks(…)中的generate_and_rm_group来获取sample与reward值，函数定义如下

```python
async def generate_and_rm_group(args, group: list[Sample], sampling_params: dict, evaluation=False) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    group = await asyncio.gather(
        *[generate_and_rm(args, sample, sampling_params.copy(), evaluation=evaluation) for sample in group]
    )

    # for the rm that need the whole group, we will not do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group
```

调用`generate_and_rm`并发生成所有的sample和reward model，对于需要整个组执行reward model的，则调用`batch_async_rm`生成reward；

```python
async def generate_and_rm(args, sample: Sample, sampling_params: dict, evaluation=False) -> Sample:
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            sample = await custom_generate_func(args, sample, sampling_params)
        else:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample

        sample.reward = await async_rm(args, sample)

    return sample
```

其中`custom_generate_function_path`为自定义的generate函数，对于工具调用类，则定义在slime/examples/retoo/generate_with_retool.py中；判断是否需要group计算reward，如果不是group生成reward，则调用`batched_async_rm`获取rewards函数；

`batched_async_rm`同样根据自定义的custom_rm_path获取自定义reward函数，对于工具调用类同样定义在slime/examples/retool/generate_with_retool.py中

### Tool-call流程

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds

    for turn in range(TOOL_CONFIGS["max_turns"]):
        # Simple: just send prompt + response
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<interpreter>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload, use_http2=args.use_http2)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        # Record current response tokens
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Check if maximum tool call count reached
        if turn >= TOOL_CONFIGS["max_tool_calls"]:
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_masks = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample
```

目前在ToolRegistry中注册的工具只有code_interpreter一种，具体的注册定义如下
![[Pasted image 20260408140322.png]]
目前在slime中工具调用的格式分为三种，system_content、sample.prompt、messages(from previous turn)，转化为标准格式，调用format_conversation_with_tools(…)获取对应的prompt；

直接调用http请求的post将请求发送给sglang_router，对返回的请求进行后处理，随后，调用execute_prefictions函数，从response中分割出对应的action和content，如果action为”code”，则调用tool_registry.execute_tool(…)执行工具，将对应的result存入next_obs中，返回，若发现模型response格式有错误，则添加引导性的prompt
![[Obsidian 2026-04-08 14.03.32.png]]
具体的执行在tool_registry.execute_tool(…)中，其中具体的函数执行流程为：

- 检查内存使用率
- 检查代码安全性
- 增添必要的wrapper来限制内存使用，同时将用户代码用try进行保护
- 创建环境_create_sage_environment(…)并临时创建code.py文件，将内容写入，使用python3执行代码，将结果返回