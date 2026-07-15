# 概述

rStar中，通过master node维护一个transferQueue和32个send workers来实际执行任务dispatch，在其他每个node上运行一个task schedular和1024个execution worker来实际执行代码，大致架构图如下：
![[Pasted image 20260408144443.png]]
# 执行命令

启动redis-server

```bash
redis-server --daemonize yes --protected-mode no --bind 0.0.0.0
```

在master node启动

```bash
tmux new-session -d -s server \\
  'cd $WORKSPACE/code-judge && \\
   MAX_EXECUTION_TIME=4 \\
   REDIS_URI="redis://$MASTER_ADDR:6379" \\
   RUN_WORKERS=0 \\
   uvicorn app.main:app --host 0.0.0.0 --port 8088 --workers 16 \\
   2>&1 | tee server.log'
```

在其他worker node启动

```bash
tmux new-session -d -s worker \\
  'cd $WORKSPACE/code-judge && \\
   MAX_EXECUTION_TIME=4 \\
   REDIS_URI="redis://$MASTER_ADDR:6379" \\
   MAX_WORKERS=64 \\
   python run_workers.py \\
   2>&1 | tee worker.log'
```

# 代码详解

## Tool的视角（请求下发后如何聚集成batch？）

首先看tool配置，定义在rstar2_agent.tools.code_judge_tool.PythonTool中，定义如下，简单实现了所verl agentLoop需要的create与execute函数，先来看`create(…)`函数

```python
class PythonTool(CodeJudgeTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        await self._start_request_processor()

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # ... ...
```

其中，request_processor为定义在rstar2_agent.tools.request_processor.RequestProcessor类，其中共存在max_concurrency(在config中为8个)个send_workers，对应尝试发送batch数据的send worker

```python
async def start(self):
        """
        Starts the concurrent sender worker tasks. Must be called within loop.
        """
        if self._running:
            print(f"[{time.monotonic():.4f}] RequestProcessor is already running.")
            return
        self._running = True
        self._sender_workers = [asyncio.create_task(self._sender_worker()) for _ in range(self._concurrency)]
        print(f"[{time.monotonic():.4f}] RequestProcessor started {self._concurrency} sender workers.")
```

其中每个`_sender_worker()`函数中，不断轮训RequestProcessor._submission_queue（所有的send_worker进程共享同一个队列），取出第一个item，如果成功，则一直尝试取满_batch_size个item（在config文件中为32个），如果取出失败或者队列为空，则打印log并等待下次重新进入while循环，如果成功获取了batch_size的item或者超时，则调用`self._perform_send_batch(…)`批量发送tool-call item

```python
if batch_item_ids:
   # Acquire semaphore permit before starting the potentially long-running batch submission
   async with self._semaphore:
   # Perform the actual batch sending using the injected function
   await self._perform_send_batch(batch_item_ids)
```

总的来说，PythonTool创建时，核心便是创建数量为concurrency的sender_worker，每个sender_worker不断check任务队列，凑够batch_size或者超时则尝试发送；

下面具体看下`_perform_send_batch(…)`的执行流程

首先，删去不在self._pending_requests（该list在调用execute时会将req和对应的future捆绑放入）中的request，并获取对应的payload信息

```python
valid_item_ids_for_batch = [req_id for req_id in batch_item_ids if req_id in self._pending_requests]

        if not valid_item_ids_for_batch:
             # print(f"[{time.monotonic():.4f}] Batch contains no valid pending items after worker picked them up.")
             return # Nothing valid to send

        # Build the payload list for the injected function using only valid IDs
        for req_id in valid_item_ids_for_batch:
             req_info = self._pending_requests[req_id] # Should exist based on valid_item_ids_for_batch
             batch_info.append({"request_id": req_id, "payload": req_info["payload"]})
             payloads_for_server.append(req_info["payload"])
```

随后，将提交的batch数量+=1，存入实际的该batch req数量，记录开始时间，阻塞等待调用`_batch_submit_func(…)`函数发送payload到对应的_session，其中_session中建立的最大连接数同样等于max_concurrency，随后根据返回的结果进行判断，如果没有拿到所有的result，则打印log，随后对于所有完成的request，将其的future设置成完成，并删去_pending_requests列表项

```python
try:
            # Call the function provided during initialization
            # It must return results in the same order as input payloads_for_server
            results_list = await self._batch_submit_func(payloads_for_server, self._session)

            submission_duration = time.monotonic() - start_time
            self._stats["total_batch_submission_duration_seconds"] += submission_duration
            self._stats["num_successful_batches"] += 1
            self._stats["total_items_processed_in_batches"] += len(payloads_for_server)

            # Process the results returned by the injected function
            # The order of results_list is assumed to match the order of payloads_for_server
            if len(results_list) != len(batch_info):
                 print(f"[{time.monotonic():.4f}] Warning: Injected function returned {len(results_list)} results, but batch had {len(batch_info)} items. Cannot reliably match results.")
                 match_count = min(len(results_list), len(batch_info))
            else:
                 match_count = len(batch_info)

            for i in range(match_count):
                 req_id = batch_info[i]["request_id"] # Get the original ID
                 result = results_list[i]          # Get the corresponding result

                 if req_id in self._pending_requests:
                      req_info = self._pending_requests[req_id]
                      future = req_info["future"]
                      if not future.done():
                           future.set_result(result)
                           del self._pending_requests[req_id]
                      else:
                           if req_id in self._pending_requests:
                                del self._pending_requests[req_id]
                 else:
                     print(f"[{time.monotonic():.4f}] Warning: Received result for unknown or already completed request ID {req_id[:6]}... Result: {result}
```

其中没有特殊制定情况下，则`_batch_submit_func`的定义如下，这也是在初始化CodeJudgeTool时初始化的func定义，其中实际将请求打成batch的执行函数，将在下面一节详解

```python
 run_jupyter_tool_calls_on_server_async = partial(
            run_tool_calls_on_server_async,
            generate_tool_call_code=generate_tool_call_code,
            generate_tool_call_input=generate_tool_call_input,
            host_addr=host_addr,
            host_port=host_port,
        )
```

再来看具体的执行及`execute(…)`函数

```python
class PythonTool(CodeJudgeTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        # ... ...

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        input = parameters.get("input", "")
        tool_call = {
            "name": "python_code_with_standard_io",
            "arguments": {
                "code": code,
                "input": input,
            },
        }
        result_text = await self.request_processor.send_request(tool_call)
        return ToolResponse(text=result_text), 0.0, {
```

其中`execute(…)`的调用时机往往是上层的`generate_sequence(…)`函数中调用，根据有对应的</tool> context存在，而将请求交给execute(…)代理执行，其中函数中将code与input从parameters中分离，将其封装进tool_call中，随后调用`request_processor.send_request(tool_call)`下发请求

send_request(…)函数实现相对简单，主要为每个req增添future，并将其与payload绑定好封装进_pending_requests列表中，随后阻塞将其放入_submission_queue中即可

```python
async def send_request(self, request_payload: Any, timeout: float = None):
        """
        Adds a single request to the buffer and waits for its result.
        This call is awaitable and provides the synchronous-like pattern.
        """
        if not self._running:
             raise RuntimeError("RequestProcessor is not running. Call .start() first.")

        request_id = str(uuid.uuid4())

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = {
            "future": future,
            "payload": request_payload
        }

        await self._submission_queue.put(request_id)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            if request_id in self._pending_requests:
                 del self._pending_requests[request_id]
            print(f"[{time.monotonic():.4f}] Request {request_id[:6]}... timed out waiting for result.")
            raise
        except Exception as e:
            if request_id in self._pending_requests:
                 del self._pending_requests[request_id]
            print(f"[{time.monotonic():.4f}] Request {request_id[:6]}... encountered error while waiting: {e}")
            raise
```

至此，tool视角的代码部分看完了，下面看看具体batch_submit_func的定义

## batch_submit_func详解（发送的请求如何执行？）

在上一小节中，没有提到server的概念，纯粹为cilent侧req下发后如何执行处理逻辑，回过头看CodeJudgeTool的初始化函数，其中，run_jupyter_tool_calls_on_server_async()的定义实际上指定了对应的judge server所在的host与port入口，即为localhost:8080，也即具体该函数的定义应当在该端口上执行，随后为其分配了max_concurrency个session(每个sender_worker一个）来进行请求

```python
class CodeJudgeTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        host_addr = self.config.get("host_addr", "localhost")
        host_port = self.config.get("host_port", "8088")
        run_jupyter_tool_calls_on_server_async = partial(
            run_tool_calls_on_server_async,
            generate_tool_call_code=generate_tool_call_code,
            generate_tool_call_input=generate_tool_call_input,
            host_addr=host_addr,
            host_port=host_port,
        )
        request_processor_batch_size = self.config.get("request_processor_batch_size", 1)
        request_processor_concurrency = self.config.get("request_processor_concurrency", 1)
        request_processor_batch_timeout_seconds = self.config.get("request_processor_batch_timeout_seconds", 30)
        tool_connector = aiohttp.TCPConnector(limit=request_processor_concurrency, force_close=True, enable_cleanup_closed=True)
        tool_timeout = aiohttp.ClientTimeout(total=60)
        tool_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
        self.request_processor = RequestProcessor(
            batch_size=request_processor_batch_size,
            batch_timeout_seconds=request_processor_batch_timeout_seconds,
            session=tool_session,
            concurrency=request_processor_concurrency,
            batch_submit_func=run_jupyter_tool_calls_on_server_async,
        )
```

直接来看`run_tool_calls_on_server_async(…)`函数，该函数首先调用generate_tool_call_code(…)添加拼接的code_template_setup与code，调用generate_tool_call_input(…)获取input，随后获取code judge所在的端口，通过call_long_batch调用post “http:localhost:8080/run/long-batch”，阻塞等待返回结果，并根据返回的结果添加对应的Success/Failure与执行时间。

随后来看看judge_server内部具体是如何执行该段代码的，以及如何处理所谓的并行

## Judge_server内部逻辑

### Code judge初始化

在启动命令中可以看到，code judge的初始化包含三部分

- 启动redis-server
- 在master node上启动唯一一次app.main
- 可选：在多个node上启动run_workers.py

首先来看master node启动的app.main

app.main做的事情很简单，本质就是通过调用connect_queue(…)启动了RedisQueue

```python
def connect_queue(is_async: bool = False) -> RedisQueue:
    return RedisQueue(
        redis_uri=app_config.REDIS_URI,
        socket_timeout=app_config.REDIS_SOCKET_TIMEOUT,
        is_async=is_async,
    )
```

Redis_server中主要定义两个子类，分别为QueueOp(queue)和PriorityQueueOp(pqueue)，其中pqueue用来存放进入的请求，queue用来存放请求执行的结果

随后看run_workers.py函数，其中，则在每个执行node上创建WorkerManager()

```python
class WorkerManager:
    def __init__(self):
        max_workers = app_config.MAX_WORKERS
        self.workers: list[Worker] = []
        logger.info(f'Starting {max_workers} workers...')
        for _ in range(max_workers):
            worker = Worker()
            worker.start()
            self.workers.append(worker)
        logger.info(f'Started {max_workers} workers')
```

其中创建MAX_WORKERS个Worker，并执行start()函数，其中每个worker即占据一个线程，执行_run_loop(…)函数，里面为你每个worker创建一个Redis_queue，其中rq.redis为同一个，概念上来讲即为论文中所说的transferQueue,，在死循环中，worker不断尝试从pqueue中pop对应的work_item，也即下一节会详细讲的payload_chunk，一旦work_item不为none，则拿到对应的payload_json，将其转变为payload，调用judge(payload.submission)执行代码

judge函数的定义如下，其中，根据sub.type获取对应额executor，如pythonExecutor/CppExecutor，

其中，设置临时目录，通过function_call方式，创建临时目录，调用ProcessExecutor.execute(…)执行代码，并将结果返回

```python
def judge(sub: Submission):
    try:
        executor = executor_factory(sub.type)
        result = executor.execute_script(sub.solution, sub.input)

        success = result.success
        run_success = result.success
        if sub.expected_output is not None:
            success = success and result.stdout.strip() == sub.expected_output.strip()
        if not success:
            save_error_case(sub, result)
        sub_result = SubmissionResult(
            sub_id=sub.sub_id, success=success, cost=result.cost,
            run_success=run_success,
            # only save stdout and stderr if expected_output is None
            stdout=result.stdout[:app_config.MAX_STDOUT_ERROR_LENGTH]
                if result.stdout is not None else None,
            stderr=result.stderr[:app_config.MAX_STDOUT_ERROR_LENGTH]
                if result.stderr is not None else None,
            reason=ResultReason.WORKER_TIMEOUT
                if result.exit_code == TIMEOUT_EXIT_CODE
                else ResultReason.UNSPECIFIED
        )
    except Exception as e:
        logger.exception(f'Worker failed to judge submission {sub.sub_id}')
        save_error_case(sub, None, e)
        sub_result = SubmissionResult(
            sub_id=sub.sub_id, run_success=False, success=False, cost=0, reason=ResultReason.INTERNAL_ERROR
        )
    return sub_result
```

当judge函数返回结果后，则将对应的结果push进redis_queue.queue中，因此可以看到，所谓的MAX_WORKER也即论文中提到的1024个EXECUTION WORKER。

### Code judge执行

其中，调用/run/long-batch后，实际执行定义在code-judge/app/main.py中的`run_long_batch`函数，而该函数则调用judge_batch(…)，接着调用_judge_batch_impl(…)函数

接下来详细看下`_judge_batch_impl(…)`函数

```python
async def _judge_batch_impl(redis_queue: RedisQueue, subs: list[Submission], long_batch=False):
    start_time = time()
    max_wait_time = app_config.LONG_BATCH_MAX_QUEUE_WAIT_TIME \\
        if long_batch else app_config.MAX_QUEUE_WAIT_TIME
    batch_chunk_size = app_config.MAX_LONG_BATCH_CHUNK_SIZE \\
        if long_batch else app_config.MAX_BATCH_CHUNK_SIZE
    # use a hash tag to make sure all payloads are in the same slot in redis cluster
    hash_tag = '{' + str(uuid.uuid4()) + '}'
    sub_chunks = chunkify(subs, batch_chunk_size)

   # ... ...

    # submit all submissions to the queue
    payload_chunks = []
    for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
        payload_chunk = [
            WorkPayload(work_id=f'{hash_tag}:{sub_chunk_id}-{idx}', submission=sub, long_running=long_batch)
            for idx, sub in enumerate(sub_chunk)
        ]
        payload_chunks.append(payload_chunk)
        await _submit(payload_chunk)

    results = []
    wait_start_time = time()
    for chunk in payload_chunks:
        # get all results from the queue
        left_time = max_wait_time - int(time() - wait_start_time)
        chunk_results = await _get_result(chunk, left_time)
        results.extend(chunk_results)
    return results
```

其中主要执行如下操作

- 根据batch_chunk_size将提交的batch大小任务切分为多个chunk，default为2
- 针对每个chunk，将其分装进 LIst[ WorkPayload]结构的payload_chunk中，随后调用_submit(…)函数提交任务
- 针对每个chunk，调用`await _get_result(…)`获取对应的执行结果，每个chunk的结果拍平存入results中

`_submit(…)`的定义如下, 其中，将payload的json和timestamp打包放入payload_jsons中，并将其push到redis_queue.pqueue中

```python
async def _submit(payloads: list[WorkPayload]):
    # payload.work_id is different, so we can safely use dict
    payload_jsons = {payload.model_dump_json(): payload.timestamp for payload in payloads}
    await redis_queue.pqueue.push(app_config.REDIS_WORK_QUEUE_NAME, payload_jsons)
```

`_get_result(…)`的定义如下，其中获取需要发送的payload.work_id作为key放入left_result_queue_names中，调用_pop_results(…)从redis_queue.queue中取出所需的payload对应的result，调用_to_result获取结构为SubmissionResult的result，并将已完成payload从剩余队列中移除；同时在最后，将未ready的work均设置为超时

```python
async def _get_result(payloads: list[WorkPayload], max_chunk_wait_time):
        """max_chunk_wait_time <= 0 means no wait (which is different from block_pop)"""
        result_queue_names = {
            f'{app_config.REDIS_RESULT_PREFIX}{payload.work_id}': payload
            for payload in payloads
        }
        results = {}
        result_start_time = time()
        left_time = max_chunk_wait_time
        start_working_time = 0
        left_result_queue_names = list(result_queue_names.keys())

        while left_result_queue_names:
            max_timestamp = max(
                result_queue_names[result_queue_name].timestamp
                for result_queue_name in left_result_queue_names
            )
            name_results = await _pop_results(left_result_queue_names, left_time)
            if not name_results: # if no result, check if timeout
                if start_working_time == 0:
                    # the queue is ordered by timestamp
                    next_payload_info = await redis_queue.pqueue.peak(app_config.REDIS_WORK_QUEUE_NAME)
                    if not next_payload_info:
                        start_working_time = time()
                    else:
                        # before next_work_timestamp, all work is done or processing.
                        # so if it is bigger than max_timestamp, we can assume all work is done or in progress.
                        next_work_timestamp = next_payload_info[1]
                        if next_work_timestamp > max_timestamp:
                            start_working_time = time()
                else:
                    # if start_working_time is set, it means all work is done or in progress.
                    # so we only wait for app_config.MAX_PROCESS_TIME for them to finish.
                    # if it is still not finished, we assume some error happened.
                    # and we can break the loop.
                    if time() - start_working_time > app_config.MAX_PROCESS_TIME:
                        logger.warning(f'No result for {len(left_result_queue_names)} submissions. '
                                       f'Assuming all submissions are timed out.')
                        logger.warning('This is mostly caused by redis OOM or workers killed or potential bug. ')
                        break
            else:
                start_working_time = 0

            for name_result in name_results:
                result_queue_name, _ = name_result
                payload = result_queue_names[result_queue_name]
                results[result_queue_name] = _to_result(payload.submission, start_time, name_result)
                left_result_queue_names.remove(result_queue_name)

            left_time = max_chunk_wait_time - int(time() - result_start_time)
            if left_time <= 0:
                break

        # fill non-ready work as timeout
        for result_queue_name in left_result_queue_names:
            results[result_queue_name] = _to_result(result_queue_names[result_queue_name].submission, start_time, None)

        await redis_queue.delete(*result_queue_names)
        return [results[result_queue_name] for result_queue_name in result_queue_names]
```

至此，工具调用完成。