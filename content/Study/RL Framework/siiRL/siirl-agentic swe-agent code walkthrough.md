# Code Walkthrough
前文讲mini-sweagent的实现在 AIO action-level agentloop中； 
本节概述siirl-agentic使能swe-agent时，具体的调用链；
同样的，在每个RolloutWorker的初始化阶段，会创建一个thread来实际负责每个sglang DP的完整数据流，具体创建thread执行的操作在_build_executor(...)中，其中，默认采用NaiveExecutor类；
具体来看，swe-agent的agentflow定义，其中，build_agentflow中首先需要获取对应的flow，其中，仍然采用原先的.swe.agentflow，具体的定义在siirl/execution/rollout/agentflow/swe/__init__.py中，其中根据BUILTIN_PROVIDERS以及对应的config中的name获取对应的subconf，其中对于sweagent，采用定义的RLTokenAgentBuilder来初始化agent，其中，runtime暂时skip不管，environment中采用k8s_adapter.py中的K8sEnvAdapterBuilder;
```py title="BUILTIN_PROVIDERS"
# Use original SWE-ReX/SWE-agent implementations via adapters
BUILTIN_PROVIDERS = {
    "agent": {
        "minisweagent": ".swe.agent.minisweagent:MiniSWEAgentBuilder",
        # 推荐：使用 RLTokenAgent (有完整的 token tracking 支持)
        "sweagent": ".swe.agent.sii_sweagent:RLTokenAgentBuilder",
        # 实验性：使用原始 SWE-agent + 适配器 (缺少 token tracking，仅用于非 RL 场景)
        "sweagent_original": ".swe.agent.swe_adapter:SWEAgentAdapterBuilder",
        # 旧名称（兼容）
        "sweagent_legacy": ".swe.agent.sii_sweagent:RLTokenAgentBuilder",
        # 使用 DefaultAgent + LiteLLMModel (通过 sglang API，用于 validate 阶段)
        "litellm_agent": ".swe.agent.sii_sweagent:LiteLLMAgentBuilder",
    },
    "environment": {
        "docker": ".swe.environment.docker:DockerEnvBuilder",
        # Use original SWE-ReX K8sDeployment via adapter (recommended)
        "k8s": ".swe.environment.k8s_adapter:K8sEnvAdapterBuilder",
        # Legacy K8sEnvBuilder (custom implementation, 1895 lines)
        "k8s_legacy": ".swe.environment.k8s:K8sEnvBuilder",
        "kr8s": ".swe.environment.kr8s:Kr8sEnvBuilder",
    },
    "runtime": {
        "swefactory": ".swe.runtime.swefactory:SWEFactoryBuiler",
        "swebench": ".swe.runtime.swebench:SWEBenchBuiler",
        "swebench_sii": ".swe.runtime.swebench_sii:SWEBenchBuiler",
        "swebench_agent": ".swe.runtime.swebench_k8s_eval:SWEBenchK8sEvalBuilder",
    },
}
```
首先分析RLTokenAgentBuilder和K8sEnvAdapterBuilder的具体定义，其中，先看RLTokenAgentBuilder，核心提供build方法，返回RLTokenAgentWrapper类；
```py title=""
class RLTokenAgentBuilder(AgentBuilder):
    def __init__(self, config: dict):
        config.pop("name")
        # convert config to DefaultAgentConfig
        # Note: model field is required by DefaultAgentConfig but not used in training (sample.model is used instead)

        # Get model config from agent configuration
        model_config = config.get("model", {})

        # Filter out fields not supported by GenericAPIModelConfig
        # max_workers and tool_parser are SGLang-specific fields that GenericAPIModelConfig doesn't support
        # These fields are still used by build_agentflow() for SGLangModelConfig, just not here
        unsupported_fields = {"max_workers", "tool_parser"}
        clean_model_config = {k: v for k, v in model_config.items() if k not in unsupported_fields}

        # Ensure name field exists (required by GenericAPIModelConfig)
        # The actual model used in training comes from sample.model (SGLang), not this config
        if "name" not in clean_model_config:
            clean_model_config["name"] = "dummy"  # Placeholder, not used in training

        self.config = DefaultAgentConfig(
            name=config.get("name", "main"),
            templates=TemplateConfig.model_validate(config.get("templates", {})),
            tools=ToolConfig.model_validate(config.get("tools", {})),
            history_processors=config.get("history_processors", [DefaultHistoryProcessor()]),
            max_requeries=config.get("max_requeries", 3),
            action_sampler=config.get("action_sampler"),
            model=clean_model_config,
        )

    def build(self, sample: SWESample) -> Agent:
        # Pop problem_statement from instance to avoid passing it twice
        instance = sample.m.data.runtime_meta.instance.copy()
        if "problem_statement" in instance:
            instance.pop("problem_statement")
        # Pop repo to avoid duplicate keyword error in _get_format_dict
        # The original SWE-agent code computes repo from self._env.repo
        if "repo" in instance:
            instance.pop("repo")

        # Handle multimodal (with images) vs text-only
        if sample.m.data.issue_images:
            instance.pop("issue_images", None)
            problem_statement = SWEBenchMultimodalProblemStatement(
                text=sample.m.data.problem_statement,
                issue_images=sample.m.data.issue_images,
                id=instance["instance_id"],
                extra_fields=instance,
            )
        else:
            problem_statement = TextProblemStatement(
                text=sample.m.data.problem_statement,
                id=instance["instance_id"],
                extra_fields=instance,
            )

        # Use original RLTokenAgent via wrapper
        return RLTokenAgentWrapper(
            templates=self.config.templates,
            tools=ToolHandler(self.config.tools),
            history_processors=self.config.history_processors,
            model=sample.model,
            max_requeries=self.config.max_requeries,
            name=self.config.name,
            action_sampler_config=self.config.action_sampler,
            problem_statement=problem_statement,
            sample=sample,
        )
```
其中再看RLTokenAgentWrapper的定义，其中负责wrapper实际调用swe-agent 三方库中的RLTokenAgent对象，其中核心提供run(...)与resume(...)方法，resume用于partial rollout时提供，而run用于正常生成流程，其中调用swe-agent库中的RLTokenAgent进行生成，并提供异常捕捉SglangGenerationAborted，用于判断该请求被abort，调用_snapshot将历史状态保存在partial_state中，随后raise异常给上层捕获；
```py title=""
class RLTokenAgentWrapper(AbstractAgent):

    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: Model,
        max_requeries: int = 3,
        name: str = "main",
        problem_statement: ProblemStatement | ProblemStatementConfig,
        sample: SWESample,
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        """Initialize the wrapper."""
        self.sample = sample
        self.problem_statement = problem_statement
        self.model = model

        # Create state with tokenizer for original RLTokenAgent
        if hasattr(model, "tokenizer"):
            state = StateWithTokenizer(model.tokenizer)
        else:
            raise ValueError(f"Model {type(model).__name__} must have 'tokenizer' attribute for RLTokenAgent")

        # Create original RLTokenAgent with state
        # Note: original RLTokenAgent uses ToolHandler, not SiiToolHandler
        # We need to convert ToolHandler config to ToolHandler
        from sweagent.tools.tools import ToolHandler

        tool_handler = ToolHandler(tools.config)

        # ``model`` is a ``SweSglangModel`` — a faithful async port of the
        # pre-6564c63 ``query_for_swe`` wrapper. Upstream ``RLTokenAgent``
        # reads ``output.get("new_prompt_token_ids", [])`` etc. with defaults,
        # so missing keys (we don't set them) degrade cleanly to empty
        # ``sample.tokens / loss_mask``. Enabling proper RL token tracking is
        # a separate workstream — do NOT bolt it onto this wrapper, the
        # incremental-tokenise attempt is what caused the reward regression.
        self._agent = OriginalRLTokenAgent(
            templates=templates,
            tools=tool_handler,
            history_processors=history_processors,
            model=model,
            max_requeries=max_requeries,
            name=name,
            _catch_errors=_catch_errors,
            _always_require_zero_exit_code=_always_require_zero_exit_code,
            action_sampler_config=action_sampler_config,
            state=state,  # Critical: provide state with tokenizer - must be last
        )

        # Expose key attributes
        self.name = self._agent.name
        self.tools = tools
        self.history = self._agent.history
        self.trajectory = self._agent.trajectory
        self.info = self._agent.info

        # Populated by ``run`` / ``resume`` when SglangGenerationAborted
        # propagates — SWEAgentFlow reads this in the abort branch to build
        # ``sample.partial_agent_data`` and delay env cleanup.
        self.partial_state: dict[str, Any] | None = None

    @property
    def templates(self):
        """Access to agent's template configuration."""
        return self._agent.templates

    def add_hook(self, hook):
        """Add hook to the underlying agent."""
        self._agent.add_hook(hook)

    async def setup(
        self,
        env: ContainerEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance."""
        swe_env = env._env if hasattr(env, "_env") else env
        await self._agent.setup(env=swe_env, problem_statement=problem_statement, output_dir=output_dir)

    async def run(
        self,
        env: ContainerEnv,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        swe_env = env._env if hasattr(env, "_env") else env

        try:
            result = await self._agent.run(
                env=swe_env, problem_statement=self.problem_statement, output_dir=output_dir
            )
        except SglangGenerationAborted:
            self.partial_state = self._snapshot()
            raise

        self._backfill_sample(result)
        return result

    async def resume(
        self,
        env: ContainerEnv,
        partial: dict[str, Any],
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Partial-rollout entry point. Pairs with ``run``'s snapshot path.

        Call order: this method ``setup``s the upstream agent, restores
        model-side TokenManager state from ``partial``, restores upstream
        agent state, then calls ``upstream_agent.resume`` which skips
        ``setup``/``on_run_start`` and jumps straight into the step loop.
        """
        swe_env = env._env if hasattr(env, "_env") else env

        # 1) setup is still required — it binds env/problem_statement/tools on
        #    the upstream agent, installs tools on the pod, and seeds
        #    info/history. restore_state below overwrites the bits that
        #    setup would otherwise discard (history, trajectory, info, ...).
        await self._agent.setup(
            env=swe_env, problem_statement=self.problem_statement, output_dir=output_dir
        )

        # 2) model side first: TokenManager / _rid / stats / _processed_message_count
        self.model.restore_state(partial)

        # 3) upstream agent state
        self._agent.restore_state(partial["swe_agent_state"])

        try:
            result = await self._agent.resume(
                env=swe_env, problem_statement=self.problem_statement, output_dir=output_dir
            )
        except SglangGenerationAborted:
            self.partial_state = self._snapshot()
            raise

        self._backfill_sample(result)
        return result

    def _backfill_sample(self, result: AgentRunResult) -> None:
        """Write rollout outputs back onto the siirl Sample.

        Shared by both normal-completion paths in ``run`` and ``resume``.
        """
        # CRITICAL: Assign token tracking data to sample for RL training
        self.sample.prompts = self._agent.init_input_ids
        self.sample.tokens = self._agent.input_ids[len(self._agent.init_input_ids) :]
        self.sample.loss_mask = self._agent.loss_mask[len(self._agent.init_input_ids) :]
        self.sample.rollout_log_probs = getattr(self._agent, "rollout_log_probs", [])
        # Upstream now also surfaces base64-encoded routed experts; forward it
        # untouched so the training end can reshape it.
        self.sample.rollout_routed_experts = getattr(self._agent, "routed_experts_raw", "") or ""

        # Store patch / exit_status in rollout for evaluation.
        # 优先读 self._agent.info（写 traj 的同一个 dict），落回 result.info 兜底。
        # Train 阶段只对 "submitted" 做 eval；validate 阶段对 "submitted*" 都做 eval。
        agent_info = getattr(self._agent, "info", {}) or {}
        info = agent_info if agent_info else (result.info or {})
        self.sample.m.rollout.patch = info.get("submission", None)
        self.sample.m.rollout.exit_status = info.get("exit_status", None)

    def _snapshot(self) -> dict[str, Any]:
        """Build the partial_agent_data dict (naive_flow-aligned shape).

        naive_flow key convention (see naive_flow._save_partial_agent_data):
            rid / messages / prompts_ids / response_ids / response_mask /
            rollout_log_prob / assistant_turns / env_turns /
            env_rewards / env_kwargs / routed_experts
        SWE-specific additions: swe_agent_state, swe_model_stats,
        swe_processed_message_count. ``swe_env_handle`` and
        ``swe_runtime_bootstrapped`` are filled in by SWEAgentFlow.
        """
        model_state = self.model.dump_state()
        upstream_trajectory = getattr(self._agent, "_trajectory", None) or list(self._agent.trajectory)
        env_turns = sum(
            1 for step in upstream_trajectory if step.get("tool_calls")
        )
        return {
            # naive_flow-aligned
            "rid": model_state["rid"],
            "messages": copy.deepcopy(self._agent.history),
            "prompts_ids": model_state["prompts_ids"],
            "response_ids": model_state["response_ids"],
            "response_mask": model_state["response_mask"],
            "rollout_log_prob": model_state["rollout_log_prob"],
            "assistant_turns": len(upstream_trajectory),
            "env_turns": env_turns,
            "env_rewards": [],
            "env_kwargs": {},
            "routed_experts": getattr(self._agent, "routed_experts_raw", "") or "",
            # SWE-specific
            "swe_agent_state": self._agent.dump_state(),
            "swe_model_stats": model_state["swe_model_stats"],
            "swe_processed_message_count": model_state["swe_processed_message_count"],
        }
```
再看K8sEnvAdapterBuilder的定义，具体定义如下，核心提供start方法，
```py title=""
class K8sEnvAdapterBuilder(ContainerEnvBuilder):
    """Parses a SWE-bench sample into a ``K8sEnvAdapter``."""

    def __init__(self, config: dict):
        self.config = config

    async def build(self, args: ContainerBuildArgs):
        raise NotImplementedError("K8s uses pre-built images. Use 'start' instead.")

    def _make_adapter(self, runtime_meta: Any) -> K8sEnvAdapter:
        sample, is_eval_pod = _extract_sample_and_eval_flag(runtime_meta)
        resume_handle = _extract_resume_handle(runtime_meta)
        image_name = _resolve_image_name(sample)
        if is_eval_pod:
            logger.info("[K8sEnvAdapterBuilder] Starting eval pod (skip_reset=True)")
        if resume_handle is not None:
            logger.info(
                f"[K8sEnvAdapterBuilder] Partial-rollout resume: attaching to "
                f"pod {resume_handle.get('pod_name')} in namespace "
                f"{resume_handle.get('namespace')}"
            )

        container = ContainerStartArgs(
            image=image_name,
            cwd="/testbed",
            startup_timeout=self.config.get("startup_timeout", 1800.0),
        )
        return K8sEnvAdapter(
            args=container,
            instance=sample,
            repo_name=self.config.get("repo_name", "testbed"),
            namespace=self.config.get("namespace", "swe"),
            config=self.config,
            skip_reset=is_eval_pod,
            resume_handle=resume_handle,
        )

    async def start(self, args: ContainerStartArgs, runtime_meta: Any = None) -> K8sEnvAdapter:
        env = self._make_adapter(runtime_meta)
        await env._ensure_initialized()
        return env
```
随后在此处分析SWEAgentFlow的具体定义，完成类定义如下，可以看到，SWEAgentFlow核心分为preprocess, generate和reward三个函数，其中，在preprocess函数中，会初始化对应的runtime以及agent，注意，这里的runtime和 agent实际上都是per-sample概念的，所有sample信息实际上会被绑定到对应的某个Flow里，注意，如果该preprocess的sample含有partial_agent_data字段，代表其为partial样本，此时，需要将对应的历史状态meta.rollout.partial_state中，方便在generate时进行恢复；
在generate函数中，实则是调用generate_async(...)函数，在generate_async函数中，首先将partial_state通过调用_inject_resume_handle(...)函数，将swe_env_handle恢复到runtime_meta中，用于保证该sample能够连接上同一个环境 (pod)；
随后调用self.env.start(...)，也即K8sEnvAdapterBuilder的start(...)函数；随后调用await m.runtime.bootstrap(env)进行runtime连接，然后调用m.agent.run(env)，也即RLTokenAgentWrapper的run方法，也即调用定义在swe-agent中的RLTokenAgent.run；根据返回结果，如果出现SglangGenerationAborted，则把对应的结果保存到partial_state中，并把错误异常raise上送；上送的异常最终在NaiveExecutor中被捕获，将对应的sample通过put_partial存入partial队列中；
```py title=""
class SWEAgentFlow(AgentFlow):
    def __init__(
        self,
        env: ContainerEnvBuilder,
        runtime: RuntimeBuilder,
        agent: AgentBuilder = None,  # 可选，validate 模式下可以为 None
        validate_agent: AgentBuilder = None,
    ):
        # 如果 agent 为 None，使用 validate_agent 作为主要的 agent
        self.agent = validate_agent if agent is None else agent
        self.validate_agent = validate_agent
        self.env = env
        self.runtime = runtime

    async def preprocess(self, sample: dict, model: Model, is_validate: bool = False) -> SWESample:
        """把数据集的一条数据 (dict) 处理成 Sample 对象；可能会 raise exception

        Args:
            sample (dict): 数据集的一条数据
            model (Model): 模型实例（每个样本独立）
            is_validate (bool): 是否为 validate 阶段

        Returns:
            Sample: rollout 并 evaluate 的算例
        """
        data = self.runtime.parse_sampledata(sample)
        meta = SWEAgentMeta(data)
        s = SWESample(meta, model)
        runtime = self.runtime.build(s)

        # 根据是否 validate 选择不同的 agent
        agent_builder = self.validate_agent if is_validate and self.validate_agent else self.agent
        agent = agent_builder.build(s)

        meta.runtime = runtime
        meta.agent = agent
        # 存下 is_validate，reward 依此决定 exit_status 过滤规则
        meta.rollout.is_validate = is_validate

        # Store weight_version (training step) from sample dict for eval to use
        if "weight_version" in sample:
            meta.weight_version = sample["weight_version"]
            logger.debug(f"[SWEAgentFlow.preprocess] Set weight_version={sample['weight_version']} for sample")

        # Partial-rollout inbound: if AgentFlowCallable forwarded a
        # ``partial_agent_data`` from the siirl Sample into this dict,
        # record it on the rollout meta. ``_generate_async`` picks it up
        # and branches into the resume path.
        partial = sample.get("partial_agent_data") if isinstance(sample, dict) else None
        if partial:
            meta.rollout.partial_state = partial
            # 置空出口字段，防止和本轮 _generate_async 新写入的 partial 互相污染
            meta.rollout.partial_agent_data = None
            logger.info(
                f"[SWEAgentFlow.preprocess] Resuming from partial "
                f"(rid={partial.get('rid')}, assistant_turns={partial.get('assistant_turns')})"
            )

        logger.debug(f"[SWEAgentFlow.preprocess] is_validate={is_validate}, agent type: {type(meta.agent)}")
        return s

    async def generate(self, sample: SWESample):
        """Dispatch the full rollout to a worker-thread event loop.

        The actual rollout body (``_generate_async``) runs in isolation — its
        SWEEnv/httpx/kubectl timers can't be starved by Sglang or Ray activity
        on the rollout main loop.  See the module header comment for background.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _get_flow_executor(), _run_async_in_thread, self._generate_async, sample
        )

    async def reward(self, sample: SWESample):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _get_flow_executor(), _run_async_in_thread, self._reward_async, sample
        )

    async def _generate_async(self, sample: SWESample):
        m = sample.m
        partial = getattr(m.rollout, "partial_state", None)
        logger.warning(
            "[SWEAgentFlow.generate] Running agent.%s", "resume" if partial else "run"
        )
        logger.debug(f"[SWEAgentFlow.generate] m.agent type: {type(m.agent)}")

        # Inject the resume handle into runtime_meta so K8sEnvAdapter's
        # builder attaches to the existing pod instead of creating a new one.
        runtime_meta = m.data.runtime_meta
        if partial and partial.get("swe_env_handle"):
            runtime_meta = _inject_resume_handle(runtime_meta, partial["swe_env_handle"])

        env = None
        try:
            env = await self.env.start(m.data.container_args, runtime_meta)
        except Exception as start_exc:
            logger.error(f"[SWEAgentFlow.generate] Failed to start environment: {start_exc}", exc_info=True)
            raise EnvCreateError(f"env start failed: {start_exc}") from start_exc

        aborted = False
        try:
            await m.runtime.bootstrap(env)  # idempotent — resume path is a no-op
            if partial:
                await m.agent.resume(env, partial)
            else:
                await m.agent.run(env)
            await m.runtime.diff(env)
        except SglangGenerationAborted:
            # Build the outbound partial_agent_data. wrapper.partial_state is
            # populated by RLTokenAgentWrapper.run/resume in its own except.
            wrapper_partial = getattr(m.agent, "partial_state", None) or {}
            env_handle = None
            try:
                env_handle = env.get_handle() if env is not None else None
            except Exception as e:
                logger.error(
                    f"[SWEAgentFlow.generate] get_handle failed: {e}; "
                    "will fall through to cleanup and drop partial",
                    exc_info=True,
                )
            if env_handle is None or not wrapper_partial:
                # Can't resume without env handle or agent snapshot — let the
                # finally cleanup the pod and don't publish partial data.
                logger.warning(
                    f"[SWEAgentFlow.generate] Abort without usable partial "
                    f"(env_handle={bool(env_handle)}, wrapper_partial={bool(wrapper_partial)}); "
                    "releasing pod"
                )
                m.rollout.partial_agent_data = None
            else:
                m.rollout.partial_agent_data = {
                    **wrapper_partial,
                    "swe_env_handle": env_handle,
                    "swe_runtime_bootstrapped": True,
                }
                aborted = True
            raise
        finally:
            if env is not None:
                if aborted:
                    # Keep pod alive; close Python client only. Fallback to
                    # full cleanup if detach itself fails — we'd rather lose
                    # the partial than leak a pod into the cluster.
                    try:
                        await env.detach()
                    except Exception as e:
                        logger.error(
                            f"[SWEAgentFlow.generate] detach failed: {e}; "
                            "falling back to cleanup",
                            exc_info=True,
                        )
                        with contextlib.suppress(Exception):
                            await env.cleanup()
                        m.rollout.partial_agent_data = None
                else:
                    try:
                        await env.cleanup()
                    except Exception as e:
                        logger.warning(f"[SWEAgentFlow.generate] cleanup failed: {e}")

        # 按 exit_status 决定是否让该样本进入 eval 阶段（与 Agentic_RL 对齐）
        # 注意：TRUNCATED 样本依然会跑 eval（仅预置 status=TRUNCATED，训练时由上层按 status 排除）。
        # Abort 路径不会到这里（上面 raise 已经离开了函数）。
        do_eval, status_override = _should_eval(m.rollout.exit_status, m.rollout.is_validate)
        if status_override is not None:
            sample.status = status_override
            tag = "Pre-set status" if do_eval else "Skip eval"
            sample.errors.append(f"{tag}: exit_status={m.rollout.exit_status!r}, is_validate={m.rollout.is_validate}")
            logger.info(
                f"[SWEAgentFlow.generate] {tag} (status={status_override.value}, do_eval={do_eval}, "
                f"exit_status={m.rollout.exit_status!r}, is_validate={m.rollout.is_validate})"
            )

    async def _reward_async(self, sample: SWESample):
        """reward body — runs inside the worker thread's event loop."""
        m = sample.m

        # 与 generate 对齐的 exit_status 过滤：不合格样本跳过 eval，但设置 reward=0
        # 这样样本仍然进入 databuffer，保证 GRPO batch size 一致性
        do_eval, _ = _should_eval(m.rollout.exit_status, m.rollout.is_validate)
        if not do_eval:
            logger.info(
                f"[SWEAgentFlow.reward] Skipping eval, setting reward=0 (exit_status={m.rollout.exit_status!r}, "
                f"is_validate={m.rollout.is_validate}, sample.status={sample.status.value})"
            )
            sample.reward = 0.0
            return

        # Check if runtime needs external env (e.g., run_instance_k8s_for_rl creates its own pod)
        needs_external_env = getattr(m.runtime, "needs_external_env", True)
        logger.debug(f"[SWEAgentFlow.reward] needs_external_env={needs_external_env}")

        env = None
        if needs_external_env:
            try:
                # 标记为 eval pod，跳过 reset（eval pod 应该保持镜像中的原始状态）
                runtime_meta = m.data.runtime_meta
                logger.debug(f"[SWEAgentFlow.reward] runtime_meta type: {type(runtime_meta)}")
                if isinstance(runtime_meta, dict):
                    runtime_meta = {**runtime_meta, "_is_eval_pod": True}
                    logger.debug("[SWEAgentFlow.reward] Set _is_eval_pod=True in dict")
                else:
                    # runtime_meta 是 SBSample 对象（dataclass，可能 frozen），绕过 frozen 检查
                    object.__setattr__(runtime_meta, "_is_eval_pod", True)
                    logger.debug("[SWEAgentFlow.reward] Set _is_eval_pod=True on SBSample object")
                env = await self.env.start(m.data.container_args, runtime_meta)
            except Exception as start_exc:
                logger.error(f"[SWEAgentFlow.reward] Failed to start environment: {start_exc}", exc_info=True)
                raise EnvCreateError(f"env start failed: {start_exc}") from start_exc
        else:
            logger.info("[SWEAgentFlow.reward] Skipping external env creation (runtime handles its own pod)")

        try:
            if env is not None:
                await m.runtime.bootstrap(env)
            await m.runtime.eval(env)
        finally:
            if env is not None:
                try:
                    await env.cleanup()
                except Exception as e:
                    logger.warning(f"[SWEAgentFlow.reward] cleanup failed: {e}")

```
随后来看swe-agent中定义的RLTokenAgent的具体实现，其中核心看run方法，其中run方法在改请求正式完成前 (step_output.done为True前)，不断的在while循环内，调用step_output = await self.step()函数进行每轮的生成与交互，在step函数中，核心调用forward_with_handling方法，该方法也是一步step的核心，其中核心分为两步，首先调用self.forward(history)进行生成，其中核心调用self.model.query(history)进行生成，其中self.model即为SWESglangModel，随后，调用self.tools.parse_actions解析output，获取其中的tool_calls，拆分到step.throught和step.action中，随后调用self.handle_action(step)进行实际的环境交互操作；
```py title=""
class RLTokenAgent(DefaultAgent):
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
        state: Any | None = None,
    ):
        super().__init__(
            templates=templates,
            tools=tools,
            history_processors=history_processors,
            model=model,
            max_requeries=max_requeries,
            name=name,
            _catch_errors=_catch_errors,
            _always_require_zero_exit_code=_always_require_zero_exit_code,
            action_sampler_config=action_sampler_config,
        )
        self.state = state
        self.token_manager = getattr(model, "token_manager", TokenManager())
        self.input_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.rollout_log_probs: list[float] = []
        self.init_input_ids: list[int] = []
        self.system_prompt_prefix: list[int] = []
        self.generate_prompt_suffix: list[int] = []
        self.user_input_flag: bool = True
        self._routed_experts_raw: str = ""
        self._error_logs: list[dict[str, Any]] = []

    def _require_tokenizer(self):
        if self.state is None or not hasattr(self.state, "tokenizer"):
            raise ValueError("RLTokenAgent requires state.tokenizer for token tracking.")
        return self.state.tokenizer

    @property
    def routed_experts_raw(self) -> str:
        """Raw base64-encoded routed_experts from the last SGLang query.

        Each query sends the full accumulated input_ids, so the last query's
        routed_experts covers the entire token sequence. slime's
        fill_routing_replay expects shape (seq_len-1, num_layers, top_k).
        """
        return self._routed_experts_raw

    def _normalize_logprobs(self, token_ids: list[int], logprobs: list[float] | None) -> list[float]:
        logprobs = list(logprobs or [])
        if len(logprobs) < len(token_ids):
            logprobs.extend([1.0] * (len(token_ids) - len(logprobs)))
        elif len(logprobs) > len(token_ids):
            logprobs = logprobs[: len(token_ids)]
        return logprobs

    def _add_prompt_tokens(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        if not token_ids:
            return
        token_ids = list(token_ids)
        logprobs = self._normalize_logprobs(token_ids, logprobs)
        self.input_ids.extend(token_ids)
        self.loss_mask.extend([0] * len(token_ids))
        self.rollout_log_probs.extend(logprobs)
        self.token_manager.add_prompt(token_ids, logprobs)

    def _add_response_tokens(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        if not token_ids:
            return
        token_ids = list(token_ids)
        logprobs = self._normalize_logprobs(token_ids, logprobs)
        self.input_ids.extend(token_ids)
        self.loss_mask.extend([1] * len(token_ids))
        self.rollout_log_probs.extend(logprobs)
        self.token_manager.add_response(list(token_ids), logprobs)

    def _sync_token_manager_from_lists(self) -> None:
        self.token_manager.reset()
        if not self.input_ids:
            return
        if not (len(self.input_ids) == len(self.loss_mask) == len(self.rollout_log_probs)):
            raise ValueError(
                "RLTokenAgent token buffers length mismatch: "
                f"input_ids={len(self.input_ids)}, loss_mask={len(self.loss_mask)}, "
                f"rollout_log_probs={len(self.rollout_log_probs)}"
            )

        cur_mask = int(self.loss_mask[0])
        buf_ids: list[int] = []
        buf_logprobs: list[float] = []

        def flush() -> None:
            if not buf_ids:
                return
            if cur_mask:
                self.token_manager.add_response(buf_ids, buf_logprobs)
            else:
                self.token_manager.add_prompt(buf_ids, buf_logprobs)

        for token_id, mask, logprob in zip(self.input_ids, self.loss_mask, self.rollout_log_probs):
            mask = int(mask)
            if mask != cur_mask:
                flush()
                buf_ids = []
                buf_logprobs = []
                cur_mask = mask
            buf_ids.append(int(token_id))
            buf_logprobs.append(float(logprob))
        flush()

    def _append_history(self, item: dict[str, Any]) -> None:
        self._chook.on_query_message_added(**item)
        self.history.append(item)  # type: ignore[arg-type]

        # Token tracking is optional but required for token-in/token-out mode.
        if self.state is None or not hasattr(self.state, "tokenizer"):
            return

        tokenizer = self.state.tokenizer
        tools_schema = getattr(self.tools.config, "tools", None)
        role = item["role"]

        if role == "system":
            try:
                content_token_ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": item["content"]}],
                    tools=tools_schema,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=False,
                )
            except TypeError:
                content_token_ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": item["content"]}],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=False,
                )

            self.init_input_ids = list(content_token_ids)
            self._add_prompt_tokens(content_token_ids)

        elif role == "assistant":
            output_tokens = item.get("output_tokens")
            rollout_log_probs = item.get("rollout_log_probs")

            if output_tokens is None or not isinstance(output_tokens, list):
                output_tokens = []
            if rollout_log_probs is None or not isinstance(rollout_log_probs, list):
                rollout_log_probs = []

            tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()

            if "qwen" in tokenizer_name:
                # 和第一版对齐：Qwen 额外补一个换行 token 198，换行不参与 loss。
                self._add_response_tokens(output_tokens, rollout_log_probs)
                self._add_prompt_tokens([198], [1.0])
            else:
                self._add_response_tokens(output_tokens, rollout_log_probs)

        elif role == "user":
            content_token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["content"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            content_token_ids = content_token_ids[len(self.system_prompt_prefix):]

            if self.user_input_flag:
                self.init_input_ids.extend(content_token_ids)
                self.user_input_flag = False
                self.logger.debug("user instance template matched")

            self._add_prompt_tokens(content_token_ids)

        elif role == "tool":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": item["content"],
            }
            content_token_ids = tokenizer.apply_chat_template(
                [tool_msg],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            content_token_ids = content_token_ids[len(self.system_prompt_prefix):]

            tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
            if "glm" in tokenizer_name and "qwen" not in tokenizer_name and content_token_ids:
                content_token_ids = content_token_ids[1:]

            self._add_prompt_tokens(content_token_ids)

    def save_trajectory(self) -> None:
        """No-op in RL mode — trajectory data is returned via AgentRunResult,
        persistence is handled by slime's rollout data saver."""
        pass

    async def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        tokenizer = self._require_tokenizer()

        model = self.model
        if hasattr(model, "reset_rollout_state"):
            model.reset_rollout_state()

        self.token_manager = getattr(model, "token_manager", self.token_manager)
        self.token_manager.reset()

        self.input_ids = []
        self.loss_mask = []
        self.rollout_log_probs = []
        self.init_input_ids = []
        self.system_prompt_prefix = []
        self.generate_prompt_suffix = []
        self.user_input_flag = True
        self._routed_experts_raw = ""
        self._error_logs = []

        try:
            self.system_prompt_prefix = tokenizer.apply_chat_template(
                [{}],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
            )
        except Exception:
            self.system_prompt_prefix = []

        self.generate_prompt_suffix = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )[len(tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
        )):]

        await super().setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

    def add_step_to_history(self, step: StepOutput) -> None:
        content = step.output if not step.tool_calls else (step.thought or step.output)
        if content is None:
            content = ""

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "thought": step.thought,
            "action": step.action,
            "agent": self.name,
            "message_type": "action",
            "thinking_blocks": step.thinking_blocks,
            "reasoning_content": step.reasoning_content,
            "output_tokens": step.output_tokens,
            "rollout_log_probs": step.rollout_log_probs,
            "rollout_routed_experts": step.rollout_routed_experts,
        }
        if step.tool_calls:
            assistant_message["tool_calls"] = step.tool_calls

        self._append_history(assistant_message)

        elided_chars = 0
        if step.observation.strip() == "":
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:
            templates = [self.templates.next_step_template]

        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            tool_calls=step.tool_calls,
            **step.state,
        )

    def get_model_requery_history(
        self,
        error_template: str,
        *,
        output: str,
        **kwargs: str | int | float | bool | None,
    ) -> list[int]:
        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)
        self.logger.warning(f"{error_template}")

        input_ids = copy.deepcopy(self.input_ids)
        return input_ids

    async def forward(self, history: list[int]) -> StepOutput:
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        step = StepOutput()
        step.query = copy.deepcopy(history)

        try:
            # Hooks / viewers 仍然看 message history；真正喂给 model 的是 input_ids。
            self._chook.on_model_query(messages=self.messages, agent=self.name)

            if self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=self.messages,
                )
                output = best.completion
                step.extra_info.update(best.extra_info)
            else:
                output = await self.model.query(history)  # type: ignore[arg-type]

            step.output = output["message"]

            if not self.init_input_ids:
                new_prompt_token_ids = output.get("new_prompt_token_ids", []) or []
                if isinstance(new_prompt_token_ids, list):
                    self.init_input_ids = list(new_prompt_token_ids)

            step.output_tokens = output.get("output_tokens", []) or []
            step.thinking_blocks = output.get("thinking_blocks", [])
            step.reasoning_content = output.get("reasoning_content", None)
            step.rollout_log_probs = output.get("rollout_log_probs", []) or []

            raw_experts = output.get("rollout_routed_experts", "") or ""
            if raw_experts:
                self._routed_experts_raw = raw_experts

            step.thought, step.action = self.tools.parse_actions(output)

            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]

            self._chook.on_actions_generated(step=step)
            return await self.handle_action(step)

        except Exception as error:
            if step.action == step.thought == "":
                step.thought = step.output
            error.step = step  # type: ignore[attr-defined]
            raise

    async def forward_with_handling(self, history: list[int]) -> StepOutput:
        async def handle_error_with_autosubmission(
            exception: Exception,
            exit_status: str,
            message: str,
        ) -> StepOutput:
            full_traceback = traceback.format_exc()
            current_step = len(self.trajectory) + 1
            self._error_logs.append(
                {
                    "step": current_step,
                    "error_type": type(exception).__name__,
                    "exit_status": exit_status,
                    "message": str(exception),
                    "n_requeries": None,
                    "traceback": full_traceback,
                }
            )
            self.logger.warning(message)
            step = getattr(exception, "step", StepOutput())
            step.thought = message
            step.exit_status = exit_status
            step.output = message
            step.done = True
            return await self.attempt_autosubmission_after_error(step)

        def handle_error_with_retry(
            exception: Exception,
            template: str,
            n_requeries: int,
        ) -> list[int]:
            full_traceback = traceback.format_exc()
            current_step = len(self.trajectory) + 1
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                try:
                    exception_message = exception.args[0]
                except (IndexError, AttributeError):
                    pass

            self._error_logs.append(
                {
                    "step": current_step,
                    "error_type": "retry",
                    "exception_type": type(exception).__name__,
                    "message": str(exception_message),
                    "n_requeries": n_requeries,
                    "traceback": full_traceback,
                }
            )

            self.logger.warning(
                "Requerying model after %s (%dth requery)",
                type(exception).__name__,
                n_requeries,
            )

            step: StepOutput = getattr(exception, "step", StepOutput())
            self.add_step_to_trajectory(step)

            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        last_requery_exception: Exception | None = None
        n_format_fails = 0

        while n_format_fails < self.max_requeries:
            try:
                return await self.forward(history)

            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

            except FormatError as error:
                last_requery_exception = error
                n_format_fails += 1
                history = handle_error_with_retry(
                    error,
                    self.tools.config.format_error_template,
                    n_format_fails,
                )
            except _BlockedActionError as error:
                last_requery_exception = error
                n_format_fails += 1
                history = handle_error_with_retry(
                    error,
                    self.tools.config.filter.blocklist_error_template,
                    n_format_fails,
                )
            except ContentPolicyViolationError as error:
                last_requery_exception = error
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
            except BashIncorrectSyntaxError as error:
                last_requery_exception = error
                n_format_fails += 1
                history = handle_error_with_retry(
                    error,
                    self.templates.shell_check_error_template,
                    n_format_fails,
                )
            except _RetryWithOutput as error:
                last_requery_exception = error
                n_format_fails += 1
                history = handle_error_with_retry(
                    error,
                    self.templates.next_step_template,
                    n_format_fails,
                )
            except _RetryWithoutOutput as error:
                last_requery_exception = error
                self.logger.warning("Retry without output, trying to resample")
                n_format_fails += 1

            except _ExitForfeit as error:
                return await handle_error_with_autosubmission(
                    error,
                    "exit_forfeit",
                    "Exiting due to forfeit",
                )
            except _TotalExecutionTimeExceeded as error:
                self.logger.exception("Exiting due to total execution time exceeded", exc_info=True)
                return await handle_error_with_autosubmission(
                    error,
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                )
            except CommandTimeoutError as error:
                self.logger.exception("Exiting due to multiple consecutive command timeouts", exc_info=True)
                return await handle_error_with_autosubmission(
                    error,
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                )
            except ContextWindowExceededError as error:
                return await handle_error_with_autosubmission(
                    error,
                    "exit_context",
                    "Exit due to context window",
                )
            except InstanceCallLimitExceededError as error:
                return await handle_error_with_autosubmission(
                    error,
                    "exit_max_iter_out",
                    "Exit due to max iteration out of limit",
                )
            except TotalCostLimitExceededError:
                raise
            except CostLimitExceededError as error:
                return await handle_error_with_autosubmission(
                    error,
                    "exit_cost",
                    "Exit due to cost limit",
                )
            except RetryError as error:
                self.logger.exception("Exiting due to retry error: %s", error, exc_info=True)
                return await handle_error_with_autosubmission(
                    error,
                    "exit_api",
                    f"Exit due to retry error: {error}",
                )
            except SwerexException as error:
                self.logger.exception("Exiting due to environment error: %s", error, exc_info=True)
                return await handle_error_with_autosubmission(
                    error,
                    "exit_environment_error",
                    f"Exit due to environment error: {error}",
                )
            except RuntimeError as error:
                self.logger.exception("Exiting due to runtime error: %s", error, exc_info=True)
                return await handle_error_with_autosubmission(
                    error,
                    "exit_error",
                    f"Exit due to runtime error: {error}",
                )
            except Exception as error:
                self.logger.exception("Exiting due to unknown error: %s", error, exc_info=True)
                return await handle_error_with_autosubmission(
                    error,
                    "exit_error",
                    f"Exit due to unknown error: {error}",
                )

        self.logger.exception("Exit due to repeated format/blocklist/bash syntax errors", exc_info=True)

        exception_to_use = last_requery_exception
        if exception_to_use is None:
            exception_to_use = RuntimeError("Exit due to repeated format/blocklist/bash syntax errors")
            exception_to_use.step = StepOutput()  # type: ignore[attr-defined]

        return await handle_error_with_autosubmission(
            exception_to_use,
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep(
            {
                "action": step.action,
                "observation": step.observation,
                "response": step.output,
                "thought": step.thought,
                "execution_time": step.execution_time,
                "state": step.state,
                "query": step.query,
                "extra_info": step.extra_info,
                "reasoning_content": step.reasoning_content,
                "output_tokens": step.output_tokens,
                "rollout_log_probs": step.rollout_log_probs,
                "rollout_routed_experts": step.rollout_routed_experts,
            }
        )
        self.trajectory.append(trajectory_step)

    async def step(self) -> StepOutput:
        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)

        # 关键改动：这里传 input_ids，不再传 self.messages。
        step_output = await self.forward_with_handling(self.input_ids)

        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore[index]
        self.info.update(
            await self._get_edited_files_with_context(patch=step_output.submission or "")
        )  # type: ignore[arg-type]
        self.info["model_stats"] = self.model.stats.model_dump()

        self.add_step_to_trajectory(step_output)

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

    async def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        await self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        self._chook.on_run_start()
        step_output = StepOutput()
        while not step_output.done:
            step_output = await self.step()
            self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        data = self.get_trajectory_data()
        data["info"]["error_logs"] = self._error_logs
        data["info"]["response_turn"] = len(self.trajectory)
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])

    def dump_state(self) -> dict[str, Any]:
        """Snapshot mutable state accumulated across steps.

        Fields here are the ones that ``setup()`` clears or that the step
        loop mutates. External refs (env/model/tools/hooks) are rebound
        per-run by ``setup``, so they're intentionally left out.
        """
        return {
            "history": copy.deepcopy(self.history),
            "trajectory": copy.deepcopy(self._trajectory),
            "info": copy.deepcopy(dict(self.info)),
            "input_ids": list(self.input_ids or []),
            "loss_mask": list(self.loss_mask or []),
            "rollout_log_probs": list(self.rollout_log_probs or []),
            "init_input_ids": list(self.init_input_ids or []),
            "system_prompt_prefix": list(self.system_prompt_prefix or []),
            "generate_prompt_suffix": list(self.generate_prompt_suffix or []),
            "user_input_flag": bool(self.user_input_flag),
            "error_logs": copy.deepcopy(self._error_logs),
            "total_execution_time": float(self._total_execution_time),
            "n_consecutive_timeouts": int(self._n_consecutive_timeouts),
            "routed_experts_raw": self._routed_experts_raw,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Inverse of ``dump_state``.

        Call order on the resuming worker: construct agent → ``await
        self.setup(env, problem_statement, output_dir)`` → ``restore_state(state)``.
        ``setup`` would otherwise clear ``init_input_ids / _error_logs /
        info / _routed_experts_raw``; we overwrite its fresh values with
        the snapshot.
        """
        self.history = list(state["history"])
        self._trajectory = list(state["trajectory"])

        self.info.clear()
        self.info.update(state["info"])

        self.input_ids = list(state.get("input_ids", self.token_manager.token_ids))
        self.loss_mask = list(state.get("loss_mask", self.token_manager.loss_mask))
        self.rollout_log_probs = list(state.get("rollout_log_probs", self.token_manager.logprobs))
        self.init_input_ids = list(state.get("init_input_ids", []))
        self.system_prompt_prefix = list(state.get("system_prompt_prefix", []))
        self.generate_prompt_suffix = list(state.get("generate_prompt_suffix", []))
        self.user_input_flag = bool(state.get("user_input_flag", False))
        self._sync_token_manager_from_lists()

        self._error_logs = list(state["error_logs"])
        self._total_execution_time = float(state["total_execution_time"])
        self._n_consecutive_timeouts = int(state["n_consecutive_timeouts"])
        self._routed_experts_raw = state["routed_experts_raw"]

    async def resume(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Partial-rollout entry point. Mirrors ``run`` minus setup / on_run_start.

        The caller must have called ``setup`` AND ``restore_state`` before
        this. We skip ``on_run_start`` so hooks don't double-initialise
        their own per-run state (e.g. timers, log files) — this is a
        continuation, not a new run.
        """
        step_output = StepOutput()
        while not step_output.done:
            step_output = await self.step()
            self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        data = self.get_trajectory_data()
        data["info"]["error_logs"] = self._error_logs
        data["info"]["response_turn"] = len(self.trajectory)
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])

```
首先看具体的SWESglangModel的query函数，其中，强制要求history为list[dict]，不然直接退出报error，随后调用self.\_query函数进行生成，而\_query函数则调用single\_query函数进行生成，其中，