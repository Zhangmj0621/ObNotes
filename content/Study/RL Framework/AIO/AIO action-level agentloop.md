# siirl-agentic swe-agent loop
由于在siirl-agentic的实现中，不同的agent会采用不同的agent_flow，此处，针对swe.agent的SWEAgentFlow来讲解当跑swe-agent训练时，具体完整的pipeline执行；
在每个RolloutWorker的初始化阶段，会创建一个thread来实际负责每个sglang DP的完整数据流，具体创建thread执行的操作在_build_executor(...)中，其中，默认采用NaiveExecutor类；
每个rolloutWorker启动线程，不断执行NaiveExecutor的run(...)函数，其中，在while self.running循环中，首先通过self.\_wait\_dispatch\_resumed(...)函数确保不会在权重同步时仍然尝试generate sample，随后，尝试获取一个semaphore，仅当有可用的semaphore的时，允许从dataloader中获取sample，避免不会超取sample导致其余的rollout worker陷入starvation；一旦获取到某个sample后，则调用self.generate(...)函数，将task放入self.tasks列表组，并设计callback，当某个task完成时，将其从tasks中discard，并且释放信号量；
再看generate函数，在generate函数中，核心首先调用预处理，若是validation，则prompt进行tokenizer，随后，**调用实际的rollout_flow负责该轨迹的完整生命周期**；随后post_proproces进行后处理；其中，如果报RolloutGenerationAborted的exception，代表该请求由于权重更新而被abort了，因此，直接将改样本放入partial的cancel\_queue中，并返回None通知上层；
具体的rollout_flow可以先看NaiveFlow，其中核心维护状态机代表agentloop，动态切换状态，仅当TERMINATED或者ABORTED才退出循环，其中，如果由于aborted而退出，则会调用_save_partial_agent_data(...)保存partial状态，其中，具体保存的状态为sglang_engine调用generate(...)函数时返回，其中，尤其注意，rid必须返回给上层，保证下次该请求被调用时，rid保持固定，sglang Engine能够意识到其为同一条请求的multi-turn交互；
# Action-level Scheduling
为了保证rollout worker不会陷入starvation，因而trajectory-level scheduling并不满足需求，需要实现action-level的scheduling，也即将RolloutController和EnvController分离，分开进行调度；
为了实现其，首先在config文件中，添加新配置，enable_action_level_scheduling，默认false，用来控制是否允许开启action-level scheduling；随后，在BUILTIN_FLOW中新增action_level_agentflow，用于在load_agentflow时能够可选加载action_level_agentflow；
```py title=""
BUILTIN_FLOW = {
    "swe": ".swe:agentflow",
    "action_level_swe": ".swe:action_level_agentflow",
}
```
action_level_agentflow定义如下，其中，返回ActionLevelSWEAgentFlow而非原先的SWEAgentFlow，其中，在BUILD_PROVIDERS中新增action_level_minisweagent指向其对应的ActionLevelMiniSWEAgentBuilder；
```py title="action_level_agentflow"
def action_level_agentflow(config: dict, model: Model) -> AgentFlow:
    """Build an action-level SWE AgentFlow (state-machine variant).

    Same config schema as :func:`agentflow`, but returns an
    :class:`ActionLevelSWEAgentFlow` instance that implements the
    ``ActionLevelFlow`` protocol consumed by ``AgenticExecutor``.
    """
    from .action_level_flow import ActionLevelSWEAgentFlow

    python_path = config.get("python_path")
    instances = []
    for name in ("agent", "environment", "runtime"):
        try:
            subconf = config[name]
            builtin = BUILTIN_PROVIDERS[name]
            cls = import_any(subconf["name"], builtin=builtin, path=python_path)
            if cls is None:
                raise ImportError(f"`{subconf['name']}` not found")
            instances.append(cls(subconf))
        except Exception as e:
            raise ImportError(f"Fail to initialize SWE action-level agentflow {name} builder: {e}") from e
    return ActionLevelSWEAgentFlow(*instances, model=model)
```
首先关注下ActionLevelMiniSWEAgentBuilder的具体实现，其中，核心Build每次run直接返回一个ActionLevelMiniSWEAgent对象，其中，弃用了原先MiniSWEAgent的run方法，转而实现了generation_step(...)与env_step(...)两个核心方法，其中该两个方法可以理解为，将原先的run方法拆分开来，其中，每次query可以理解为一次model的generation操作，parse_action(...)可以理解为从上一轮生成的response中提取环境交互的命令；
```py title="ActionLevelMiniSWEAgent"
class ActionLevelMiniSWEAgent(MiniSWEAgent):
    """Re-entrant version of ``MiniSWEAgent``.

    Reuses ``render_template``, ``parse_action``, ``has_finished``,
    ``query``, ``execute_action``, and ``extra_template_vars`` from the
    parent class. The only divergence is the driver: rather than running
    the full loop, the executor calls ``generation_step`` and ``env_step``
    alternately.
    """

    def __init__(self, config: AgentConfig, sample: SWESample):
        super().__init__(config, sample)
        # False until the system/instance prompt pair has been added.
        self._initialized: bool = False

    async def run(self, env: ContainerEnv):  # pragma: no cover - not used in action-level mode
        raise NotImplementedError(
            "ActionLevelMiniSWEAgent is driven via generation_step/env_step; "
            "do not call run() directly."
        )

    async def generation_step(self) -> tuple[StepStatus, Optional[str]]:
        """Run one LLM call and parse its action.

        Returns ``(status, pending_action)`` where ``pending_action`` is the
        shell command to execute when ``status == NEEDS_ENV_STEP``, else
        ``None``.
        """
        if not self._initialized:
            self.sample.add_message("system", self.render_template(self.config.system_template))
            self.sample.add_message("user", self.render_template(self.config.instance_template))
            self._initialized = True

        try:
            response = await self.query()
        except TerminatingException as e:
            self.sample.add_message("user", str(e))
            self._set_terminal_result(e)
            return StepStatus.TERMINATED, None

        try:
            cmd = self.parse_action(response)
        except FormatError as e:
            self.sample.add_message("user", str(e))
            return StepStatus.NEEDS_GENERATION, None
        except NonTerminatingException as e:
            self.sample.add_message("user", str(e))
            return StepStatus.NEEDS_GENERATION, None
        except TerminatingException as e:
            self.sample.add_message("user", str(e))
            self._set_terminal_result(e)
            return StepStatus.TERMINATED, None

        return StepStatus.NEEDS_ENV_STEP, cmd

    async def env_step(
        self, env: ContainerEnv, pending_action: str
    ) -> tuple[StepStatus, None]:
        """Execute ``pending_action`` in ``env`` and append the observation."""
        try:
            output = await env.execute(pending_action, check=False)
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            raw = (
                e.output.decode("utf-8", errors="replace")
                if isinstance(e, subprocess.TimeoutExpired)
                else e.strerror
            )
            msg = self.render_template(
                self.config.timeout_template, action=pending_action, output=raw
            )
            self.sample.add_message("user", msg)
            return StepStatus.NEEDS_GENERATION, None

        try:
            self.has_finished(output)
        except TerminatingException as e:
            # Matches the original flow: on Submitted/LimitsExceeded,
            # ``get_observation`` never got to append the observation, so we
            # only append the terminating message.
            self.sample.add_message("user", str(e))
            self._set_terminal_result(e)
            return StepStatus.TERMINATED, None

        observation = self.render_template(
            self.config.action_observation_template, output=output
        )
        self.sample.add_message("user", observation)
        return StepStatus.NEEDS_GENERATION, None

    def _set_terminal_result(self, e: TerminatingException) -> None:
        if isinstance(e, Submitted):
            result = SWERolloutResult.SUBMIT
        elif isinstance(e, LimitsExceeded):
            result = SWERolloutResult.EXCEED
        else:
            result = SWERolloutResult.FAILURE
        self.sample.m.rollout.result = result


class ActionLevelMiniSWEAgentBuilder(MiniSWEAgentBuilder):
    """Builder that produces :class:`ActionLevelMiniSWEAgent` instances."""

    def build(self, sample: SWESample) -> Agent:
        return ActionLevelMiniSWEAgent(self.config, sample)
```
其中在看ActionLevelSWEAgentFlow的具体实现，其中，run_generation_step中，如果为首次到来，state为None，则会进行初始化，初始化环境与runtime信息，并未该样本初始化一个对应的ActionLevelMiniSWEAgent，随后调用具体的generation_step(...)进行生成，而对于环境交互来说，则是调用具体的env_step(...)来执行环境交互，其中，每次action结束后，将结果存入PartialRolloutState返回；
```py title=""
class ActionLevelSWEAgentFlow(SWEAgentFlow):
    """State-machine variant of :class:`SWEAgentFlow`.

    The non-segmented ``generate`` method is intentionally *not* overridden;
    callers drive this flow via ``run_generation_step`` / ``run_env_step``
    instead.
    """

    def __init__(
        self,
        agent: AgentBuilder,
        env: ContainerEnvBuilder,
        runtime: RuntimeBuilder,
        model: Model,
    ):
        super().__init__(agent, env, runtime, model)
        # Keyed by id(sample) because SWESample carries no stable uid and
        # samples are mutable in-process objects for the whole rollout.
        self._env_by_uid: dict[int, ContainerEnv] = {}
        self._agents_by_uid: dict[int, ActionLevelMiniSWEAgent] = {}
        self._bootstrap_done: set[int] = set()

    async def preprocess(self, raw_sample: dict) -> SWESample:  # type: ignore[override]
        return super().preprocess(raw_sample)

    async def run_generation_step(
        self, sample: SWESample, state: Optional[PartialRolloutState]
    ) -> PartialRolloutState:
        uid = id(sample)

        if state is None:
            try:
                env = await self.env.start(sample.m.data.container_args)
            except Exception:
                return PartialRolloutState(status=StepStatus.ABORTED)

            try:
                await sample.m.runtime.bootstrap(env)
            except Exception:
                await _safe_cleanup(env)
                return PartialRolloutState(status=StepStatus.ABORTED)

            self._env_by_uid[uid] = env
            self._bootstrap_done.add(uid)

            agent = sample.m.agent
            if not isinstance(agent, ActionLevelMiniSWEAgent):
                agent = self.agent.build(sample)
                sample.m.agent = agent
            self._agents_by_uid[uid] = agent

        agent = self._agents_by_uid[uid]
        status, pending_action = await agent.generation_step()
        return PartialRolloutState(status=status, pending_action=pending_action)

    async def run_env_step(
        self, sample: SWESample, state: PartialRolloutState
    ) -> PartialRolloutState:
        uid = id(sample)
        env = self._env_by_uid[uid]
        agent = self._agents_by_uid[uid]
        status, _ = await agent.env_step(env, state.pending_action)
        return PartialRolloutState(status=status)

    async def finalize(self, sample: SWESample, state: PartialRolloutState) -> None:
        uid = id(sample)
        env = self._env_by_uid.pop(uid, None)
        self._agents_by_uid.pop(uid, None)
        self._bootstrap_done.discard(uid)
        if env is None:
            return
        try:
            await sample.m.runtime.diff(env)
        finally:
            await _safe_cleanup(env)

    async def on_abort(self, sample: SWESample, state: PartialRolloutState) -> None:
        uid = id(sample)
        env = self._env_by_uid.pop(uid, None)
        self._agents_by_uid.pop(uid, None)
        self._bootstrap_done.discard(uid)
        if env is None:
            return
        await _safe_cleanup(env)

    async def reward(self, sample: SWESample) -> None:
        m = sample.m
        async with await self.env.start(m.data.container_args) as env:
            await m.runtime.eval(env)

    async def project_to_dc_sample(self, flow_sample: SWESample, dc_sample: Any) -> None:
        # Mirrors AgentFlowCallable.__call__ field copy: executor uses dc_sample
        # for downstream tensor formatting and put_data.
        dc_sample.responses = np.array(flow_sample.tokens, dtype=np.int64)
        dc_sample.rollout_log_prob = np.array(flow_sample.rollout_log_probs, dtype=np.float32)
        dc_sample.response_mask = np.array(flow_sample.loss_mask, dtype=np.int64)
        dc_sample.rewards = float(flow_sample.reward) if flow_sample.reward is not None else 0.0
```
# Swe-agent Action-level Scheduling
本节概述在siirl-agentic+swe-agent上集成的swe-agent上实现的action-level agentloop以及基于fcfs的priority-based scheduling；
## Priority-aware Scheduling
仔细回顾完整的基于priority-aware的scheduling代码；
### Predictor
在async_run.py启动时候，根据是否指定config.rollout.predictor来加载预测模型，如果设置了config.rollout.predictor.enabled=True，则从LengthPredictorServerActor中进行初始化，其中LengthPreditorActor则主要调用lauch也即LengthPredictorEngine.launch_server将小模型启动，并返回url；
```py title=""
@ray.remote
class LengthPredictorServerActor:
    def __init__(self, engine_kwargs: dict):
        self._engine_kwargs = dict(engine_kwargs)
        self._engine: Optional[LengthPredictorEngine] = None

    def launch(self, ready_timeout_s: int = 300) -> str:
        if self._engine is not None:
            return self._engine.url()
        self._engine = LengthPredictorEngine(**self._engine_kwargs)
        self._engine.launch_server(ready_timeout_s=ready_timeout_s)
        return self._engine.url()

    def url(self) -> Optional[str]:
        return self._engine.url() if self._engine is not None else None

    def shutdown(self) -> None:
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
```
其中，创建的LengthPredictorServer核心提供_predict_sync方法，接受messages作为输入；
### AgenticExecutor
再来看主pipeline，核心仍然通过AgenticExeuctor中的submit_sample将其提交到ActionLevelSWEAgentFLow中，在_submit_sample_async时，首先调用_init_env_and_agent(...)启动agent, env和runtime，其中包括对partial状态的恢复；随后，调用_fire_replica_inflight通知data_coordinator某个inflight样本正在生成了，用于后续预测决策；
由于目前的abort逻辑直接放到了ActionLevelSWEAgentFlow层进行处理，AgenticExecutor相当于只做一个中转层，他的工作主要只有如下几步：
* 初始化时，判断是否允许优先级调度，设置rollout_flow的最大生成concurreny和data_coordinator以及predictor，并设置oversubscribed为max_concurrency_size * async_factor (最大不超过1.5 * max_concurrency_size， 核心用来overlap环境启动时间，但不应该一次性进入太多Sample)；
* 核心提供run方法，其中，首先调用一次data_coordinator.get_priority_snapshot将初始优先级snapshot获取到self.\_pri\_snapshot中，随后，设置最大并发为信号量train_semaphore，随后，如NaiveExecutor一样，启动dispatch_loop，随后启动rollout_status协称，不断打印状态信息日志，随后在while self.running循环中，不断等待dispatch_resumed，确认是否现在允许获取新样本，如果可以，则获取信号量 (oversubscribed)，并尝试获取get_sample，get_sample的顺序为优先从被权重同步打断的sample中拿，随后再拿新sample，然后调用_run_rollout(...)方法；
* 在\_run\_rollout方法中，
  * 首先调用_pre_process方法，对sample进行预处理，预处理操作本身仅复制raw_prompt_ids到sample.prompts中，随后，创建_InFlight对象，其中，设置dc_sample为预处理完的sample，
  * 随后，首先调用_ensure_flow_sample(in_flight)，调用rollout_flow的预处理函数，此处的preprocess则包含agent与runtime的初始化，是厚重的同步操作，因而调用asyncio.to_thread放入单独的线程中执行，其中，会将sample的partial_agent_data恢复到flow_sample.meta的rollout中，随后，由于所有在调用_ensure_flow_sample的样本要么是由于权重同步被打断，要么就是新样本，因此生成一个唯一的新rid，并将rid分配到给flow_sample.model，**并根据fcfs原则，为该sample设置初始优先级**；
  * 随后，调用rollout_flow.submit_sample进行实际的生成，其中，异常处理分别包含两层，首先对于RolloutGenerationAborted，直接put_partial，如果是SglangGenerationAborted，则认为是由于权重同步而产生的打断 (不然会在rollout_flow层直接处理)，将flow_sample中的partial_agent_data进行保存到partial中，如果不存在，则再看flow_sample.m.rollout中是否存在partial_agent_data，如果存在，则赋值给partial；随后将partial赋值到dc_sample的partial_agent_data中，将dc_sample调用put_partial放到partial的data_coordinator队列中；如果不存在partial或者存在别的异常，则调用_create_fallback_sample创建假样本 (注意，fallback的sample需要存raw_prompt，uid和replica_index等等)；
  * 对样本 (成功 or fallback)进行post_process，调用put_data将其放入等待被消费的sample_queue中，等到Trainer获取；
  * 最后，释放信号量；
至此，AgenticExecutor结束；
### ActionLevelSWEAgentFlow
随后查看ActionLevelSWEAgentFlow的运行逻辑，核心关注submit_sample后，对该sample的具体处理；
首先关注_WorkItem class，用于保存单个请求的状态，具体定义如下：
```py title="_WorkItem"
@dataclass
class _WorkItem:
    flow_sample: Any
    dc_sample: Any
    state: Optional["PartialRolloutState"]
    done_future: asyncio.Future
    # HP gets slots first and may preempt an LP mid-gen.
    priority_is_high: bool = False
    # Use it to distinguish preempted from weight-sync abort.
    preempted: bool = False
    # Used to tag predictor calls so stale predictions can be discarded.
    gen_turn_seq: int = 0
```
==随后再来看实际的ActionLevelSWEAgentFlow的完整逻辑==，ActionLevelSWEAgentFlow继承自SWEAgentFlow，用于实现action-level的调度，核心提供如下方法：
* 在初始化时，新增_env_by_uid/\_agents\_by\_uid/\_bootstrap\_done，确保后续的每个Sample能够绑定到同一个ContainerEnv，不用重复创建环境，并且绑定到同一个ActionLevelSiiSWEAgent，同时，提供_pending\_queue, \_hp\_resume\_queue, \_lp\_resume\_queue, \_env\_queue四个队列，分别在获取实际需要进行的生成的样本定义优先级；此外，提供_lp_inflight来作为双向链表提供O(1)的preempt eviction，具体定义的class类为OrderedSet类，定义在/Users/zhangmingjun/code/siirl-agentic/siirl/execution/rollout/priority/ordered_set.py中；
* 在submit_sample函数中，核心通过使用get_shared_flow_worker().submit来让sample的submit跑在不同的asyncio的thread上，其中，看_submit_sample_async：
  * 其中，首先调用_ensure_scheduler(...)函数，确保_pending_queue/\_hp\_resume\_queue等等被成功初始化，并启动_processor_worker和_interaction_worker协程；
  * 随后调用_init_env_and_agent(...)对环境和agent进行初始化，在初始化结束后，调用self.\_fire\_replica\_inflight(...)通知data_coordinator部分样本已经准备好了进入了pending_queue，随后，按照如下，将对应的Sample包装至_WorkItem中；
  * 随后，将item放入_pending_queue中，随后设置_dispatch_signal(...)，通知processor_worker有新的可用的item可获取了，await等待item中的done_future完成；
==随后来看processor_worker协程以及interaction_worker协程具体做的事情==；
process_worker的代码逻辑如下，核心维护while循环，首先通过调用_next_dispatchable获取是否目前有可用的样本可以用来生成，按照高优等待环境交互、低优等待环境交互以及pending_queue的顺序查看是否有可用sample，如果都没有，则等待\_dispatch\_signal被人set；在获取到了可用于生成的样本后，判断其对应的priority_idx，如果为1，则证明为高优请求，则可以尝试打断正在生成的低优先级请求，调用_maybe_preempt_for_hp()，调用\_pick\_lp\_victim获取最新的可被打断的低优先请求，打断其，随后调用abort_rid向sglang engine发送请求进行打断；随后在主pipeline中，等待目前生成的任务个数小于最大生成上限，则按照顺序从队列中取出一个可用的样本 (\_WorkItem)；随后调用_reclassify_priority为改样本设置优先级，根据最新的\_priority\_snapshot中的优先级，以及policy，对该样本进行优先级设置，存入item.priority_is_high中，随后对改item调用\_run\_gen\_task，将其塞入self.\_active\_gen\_tasks中；
```py title=""
async def _processor_worker(self) -> None:
        while True:
            try:
                priority_idx = await self._next_dispatchable()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[ActionLevelSWEAgentFlow._processor_worker] _next_dispatchable failed: {exc}")
                continue

            if priority_idx == 1:
                await self._maybe_preempt_for_hp()

            # Throttle: wait until a slot frees up.
            while len(self._active_gen_tasks) >= self._gen_concurrency:
                done, _ = await asyncio.wait(
                    self._active_gen_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    self._active_gen_tasks.discard(t)
                    # Swallow exceptions — they were handled inside
                    # ``_run_gen_task`` and resolved onto done_future.
                    if not t.cancelled():
                        try:
                            t.result()
                        except BaseException as exc:
                            logger.error(
                                f"[ActionLevelSWEAgentFlow] gen task raised uncaught: {exc}"
                            )

            item: Optional[_WorkItem] = None
            if not self._hp_resume_queue.empty():
                item = self._hp_resume_queue.get_nowait()
            elif not self._lp_resume_queue.empty():
                item = self._lp_resume_queue.get_nowait()
            elif not self._pending_queue.empty():
                item = self._pending_queue.get_nowait()
            if item is None:
                # The queue we'd identified is now empty — restart the
                # outer loop to wait for fresh signal.
                continue

            self._reclassify_priority(item)
            task = asyncio.create_task(self._run_gen_task(item))
            self._track_gen_dispatch(task, item)
```
==随后再看_run_gen_task的具体实现==，具体调用_run_generation_step_async，其中，当检测到有SglangGeneratinAborted异常时，根据item的preempted是否为True，来判断是由于priority-aware scheduling还是权重同步被打断，如果权重同步被打断，则调用_on_abort_async函数保存partial状态到sample.partial_agent_data中，并将异常进一步上送给done_future，让其能够被_submit_sample_async捕捉随后被AgenticExecutor的_run_rollout捕捉到；**若正常返回，则调用_route_after_step**，根据具体的状态进行切换；
* StepStatus.TERMINATED：正常结束的样本，调用_finalize_async(...)保存状态，清理环境等等，随后调用_reward_async计算reward，其中，如果返回的state.status为submitted，则允许进行评估；随后调用project_to_dc_sample(...)将flow_sample的状态保存到dc_sample中；
* StepStatus.ABORTED：调用_on_abort_async，将SglangGenerationAborted上送，并detach env；
* StepStatus.NEEDS_ENV_STEP：如果允许predict，则获取agent对应的messages，将其调用异步task \_predict\_and\_report(...)进行预测，随后直接将该item放入self.\_env\_queue中，等待环境交互；
* StepStatus.NEEDS_GENERATION：证明环境交互结束，则调用_reclassfiy_priority根据最新的priority_snapshot进行优先级更新，并放入不同resume_queue中；
==再看 \_predict\_and\_report(...)函数==，其中，若未初始化，则调用data_coordinator.get_predictor_url(...)获取predictor url，若获取失败，则fallback到fcfs policy；若成功，则调用PredictorClient.predict方法，进行预测，返回predicted，随后将自己所在的uid, replica_idx以及预测的predicted长度，turn_seq一并发送给data_coordinator，使其能够异步的进行决策；
再看==\_interaction\_worker(...)==函数，其中维护while死循环，不断从+\_env\_queue中获取item，创建\_run\_env\_task(...)task即可；
## DataCoodinator
