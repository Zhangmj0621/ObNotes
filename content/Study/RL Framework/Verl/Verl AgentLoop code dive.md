# 概述

本文详解verl agentLoop的完整执行逻辑，以sandbox_fusion tool为例

# 架构图

架构图来自verl 0.4.2文档
![[Pasted image 20260408135614.png]]
# 代码

从架构图来看，整个流程相对比较清晰了，最上层为定义在verl/experimental/agent_loop/下的agent_loop.py中AgentLoopManager，负责管理所有的AgentLoopWorker，初始化在verl/trainer/ppo/ray_trainer.py中的RayPPOTrainer类中，在init_workers(…)函数中，如果rollout mode为async，则创建AgentLoopManager，也从侧面说明了，目前verl只支持async mode下启用multi-turn，且必须为colocated模式

```python
if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )
```

在AgentLoopManager中，根据函数_init_agent_loop_workers()创建共rollout.agent.num_workers个AgentLoopWorker，创建时采用round-robin的方式，每个agent_loop_worker尽可能分布在不同的节点上

```python
def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.rm_executor)
            )
```

其中传入的server_handle的初始化为server.init_hybrid()，在函数中执行launch_servers()，随后在函数中直接执行SGLangHttpServer. launch_server()启动sglang server，其中_server_handle为server 0，_server_address与server_port同样，每一个SGLangReplica都有一个自己的server0也即master node；

```python
# get http server address from first server
server_address, server_port = await self.servers[0].get_server_address.remote()
self._server_handle = self.servers[0]
self._server_address = f"{server_address}:{server_port}"
```

在AgentLoopWorker中，创建AsyncLLMServerManager对象，负责管理所有的LLM Server，其中维护一个最小堆，记录每个server上分配的请求个数，确保请求server时时刻访问堆顶元素

```python
class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
```

接下来看上层框架如何调用进AgentLoop中

在RayPPOTrainer的主执行函数fit(…)中，如果async_rollout_mode为true，则执行async_rollout_manager.generate_sequences(…)，也即AgentLoopRolloutManager；

在generate_sequences(…)中，将batch按照AgentLoopWorker数量均分为chunk，调用每个worker的generate_sequence(…)函数

```python
async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output
```

其中，核心在于对于batch中的**每个sample**，都分别调用self._run_agent_loop(…)函数创建任务，

在_run_agent_loop(…)中，获取对应的AgentLoopBase，对于multi-turn来说则是定义在/verl/experimental/agent_loop/tool_agent_loop.py中的ToolAgentLoop类，随机调用他的run方法，核心相关代码如下，其中维护状态机实现agent_loop

```python
# State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
                agent_data.assistant_turns += 1
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
                agent_data.user_turns += 1
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
```

而在_handle_generating_state(…)中，则调用AsyncLLMServerManager.generate(…)方法，其中包含 choose负载最低的server，并调用该server的generate函数，即SGLangHttpServer.generate(…)，其中直接调用tokenizer_manager.generate_request()返回结果

```python
async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        output = await self.tokenizer_manager.generate_request(request, None).__anext__()
        if return_logprob:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)
```

## 初始化工具

在每个ToolAgentLoop初始化的时候，会通过调用

```python
tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
```

初始化工具，其中详细看一下内部的实现，其中针对非mcp和mcp的tool有两种截然不同的初始化方式

```python
	def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    # Use a temporary event loop in a new thread because event
    # loop may already exist in new async architecture while retaining
    # backwards compatibility
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=tmp_event_loop.run_forever, name="mcp tool list fetcher", daemon=True)

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()

        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        for tool_config in tools_config.tools:
            cls_name = tool_config.class_name
            tool_type = ToolType(tool_config.config.type)
            tool_cls = get_tool_class(cls_name)

            match tool_type:
                case ToolType.NATIVE:
                    if tool_config.get("tool_schema", None) is None:
                        tool_schema = None
                    else:
                        tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                    tool = tool_cls(
                        config=OmegaConf.to_container(tool_config.config, resolve=True),
                        tool_schema=tool_schema,
                    )
                    tool_list.append(tool)
                case ToolType.MCP:
                    mcp_tools = run_coroutine(initialize_mcp_tool(tool_cls, tool_config))
                    tool_list.extend(mcp_tools)
                case _:
                    raise NotImplementedError
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()

    return tool_list
```

其中，启动一个thread，不断执行临时事件循环来注册mcp tool

先看普通tool register，也即ToolType.NATIVE，其中直接根据tool_cls初始化对象，将初始化的tool放入tool_list列表中

再来看MCP tool register

```python
case ToolType.MCP:
                    mcp_tools = run_coroutine(initialize_mcp_tool(tool_cls, tool_config))
                    tool_list.extend(mcp_tools)
```

其中启动一个协程将其丢入事件循环中，具体执行函数initialize_mcp_tool(…)函数

```python
async def initialize_mcp_tool(tool_cls, tool_config) -> list:
    from verl.tools.utils.mcp_clients.McpClientManager import ClientManager

    tool_list = []
    mcp_servers_config_path = tool_config.mcp.mcp_servers_config_path
    tool_selected_list = tool_config.mcp.tool_selected_list if "tool_selected_list" in tool_config.mcp else None
    await ClientManager.initialize(mcp_servers_config_path, tool_config.config.rate_limit)
    # Wait for MCP client to be ready
    max_retries = 10
    retry_interval = 2  # seconds
    for i in range(max_retries):
        tool_schemas = await ClientManager.fetch_tool_schemas(tool_selected_list)
        if tool_schemas:
            break
        if i < max_retries - 1:
            logger.debug(f"Waiting for MCP client to be ready, attempt {i + 1}/{max_retries}")
            await asyncio.sleep(retry_interval)
    else:
        raise RuntimeError("Failed to initialize MCP tools after maximum retries")
    # mcp registry
    assert len(tool_schemas), "mcp tool is empty"
    for tool_schema_dict in tool_schemas:
        logger.debug(f"tool_schema_dict: {tool_schema_dict}")
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config, resolve=True),
            tool_schema=tool_schema,
        )
        tool_list.append(tool)
    return tool_list
```

来看一下MCPClientManager.initialize(…)是如何初始化配置的，首先，加载config文件，获取其中的所有mcpServers，调用SSETransport建立连接，其中具体的url可以从各个官网上拿到

```python
class MCPClientManager:
    rootServerName = "mcpServers"
    initialized = False
    clients = []
    tool_client_mapping = {}
    rate_limiter = None

    async def initialize(self, config_path, rate_limit: float = 10.0):
        if self.initialized:
            return
        """Initialize the MCP Client Manager and start all clients"""
        result = self._load_config(config_path)
        servers = result[self.rootServerName]
        exclude_sse_servers = {self.rootServerName: {}}
        for server_name in servers.keys():
            server = servers[server_name]
            if "auth_token" in server:
                transport = SSETransport(url=server["url"], headers={"Authorization": f"Bearer {server['auth_token']}"})
                client = Client(transport)
                self.clients.append(client)
            else:
                exclude_sse_servers[self.rootServerName][server_name] = server

        if exclude_sse_servers[self.rootServerName]:
            self.clients.append(Client(exclude_sse_servers))

        # Initialize rate limiter
        self.rate_limiter = TokenBucket(rate_limit)
        self.initialized = True
```

当获得了所有的clients，后续在主函数中，尝试获取tools，存放入tool_schemas对立中，尝试max_retries次，一旦成功，则像通用tool一样初始化，后续调用工具时则直接利用先前建立连接的client即可

```python
async def call_tool(self, tool_name, parameters, timeout):
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        client = self.get_client_with_tool_name(tool_name)
        async with client:
            return await client.call_tool_mcp(tool_name, parameters)
```