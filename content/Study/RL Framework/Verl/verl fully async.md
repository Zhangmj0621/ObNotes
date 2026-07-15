本文解析verl是如何实现fully async策略，并思考假设集成了turn-level partial rollout后如何实现

# 主pipeline

核心实现taskRunner类FullyAsyncTaskRunner，在其中的run方法中，首先初始化组件，包括FullyAsyncRollouter与FullyAsyncTrainer，消息队列Message Queue（消费者生产者模式，rollout结束结果放入message queue）
![[Pasted image 20260408135758.png]]
其中message queue为利用ray实现的simple的双端队列，其中维护asyncio.lock与asyncio.condition两个对象控制多线程，主要核心看下FullyAsyncRollouter逻辑
![[Pasted image 20260408135818.png]]
核心维护pending_queue, result_queue与cancel_queue三个asyncio队列，主核心pipeline在函数fit中，由FullyAsyncTaskRunner中调用

```python
async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")
```

其中在`_streaming_generation_main()`函数中，负责维护三个协程，feed_task, process_task, consume_task，三个协程的执行逻辑如下：

- feed_task：不断的根据epoch与batch_dict获取单个sample，并根据rollout.n获取full batch数据，将其存入pending_queue中；
- process_task：不断的获取新的sample，优先从cancel_queue中获取，否则从pending_queue中获取，确保目前正在rollout都sample数量不超过len(server) * 16（假设rollout.n=8，可以理解为每个dp的batchsize为128），如果不超过，则select_worker，将对应的sample发送给对应的server进行推理，将该sample添加到active_task队列中，**底层执行利用PartialSingleTurnAgentLoop的agentloop实现**，如果完成，则根据if_cancel为true还是false决定将该sample放入cancel_queue还是result_queue中；
- consume_task：不断的从result_queue取出已完成sample，将其放入message_queue中；

# 适配fully async sglang multi-turn

- [x] 增添fully async async_sglang_server
- [x] 增加FullyAsyncSGLangReplica
- [x] 适配fsdp_worker/param sync等
    - [x] 解决tensor未能成功创建到GPU报错
    - [x] 解决权重同步报错 Assert: Flush cache failure
    - [x] 修改generate cancel逻辑，要求等待generate被/abort_request打断
    - [x] 删去代码中duplicate部分
- [ ] 适配multi-turn sglang+tool(code/search)
    - [x] 增加partial_multi_turn_tool_agent_loop
    - [x] 修改pending、generation逻辑
    - [x] 增添aborted逻辑
    - [x] 修改response mask逻辑不缓存tool-call response部分
    - [x] 验证code功能
    - [ ] 适配自定义reward，目前定位问题到用ray启动的方法会缺少导入的自定义模块，通过添加options解决（难以解决，暂时不管了。。。）
    - [x] 解决尝试终止sglang所有请求的bug
    - [x] 解决tool-call时无法被正确cancel的bug
    - [x] 解决comsumer_worker奔溃问题

# 假设加入turn-level，如何实现？

slime实现multi-turn partial rollout retool

[https://github.com/THUDM/slime/pull/510/files](https://github.com/THUDM/slime/pull/510/files)