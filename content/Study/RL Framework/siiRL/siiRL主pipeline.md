siiRL初始化流程

## 主干pipeline

siiRL初始化在文件/SIIRL/siirl/client/main_dag.py

该文件作为launcher，主要类为MainRunner，初始化ray分布式环境，调用MainRunner.run()函数来进行初始化，所做的主要pipeline有

- 若配置，从environment_path中加载环境配置
    
- 初始化DataBuffer，databuffer数量取决于trainer节点数量与每个节点batchsize的较小值
    
- 加载DAG workflow文件，通过DAGConfigLoader进行初始化获取taskGraph
    
    - 主要讲nodes与depencies依赖添加至taskGraph中
- taskGraph为每个任务进行调度，分配具体的资源
    
    - 根据trainer.nnodes数量和n_gpus_per_nodes启动TaskScheduler
    - 调用TaskScheduler.schedule_and_assign_tasks(…)根据taskgraph进行mapping，每个DAG Worker有一个irreducible subgraphs
    - 分类irreducible subgraphs核心函数为TaskLoader.discover_and_split_parallel_paths(…)，其中subgraph的核心切分策略采用如下几种（对于给定的ppo而言，已经是irreduciable subgraph)
        - fan-out：当一个node有多个children时，可以进行fork
        - reconverge：从初始节点到终点的不同路径进行划分
    - 从大到小排序tasks，使得大的任务有更高的优先级
    - 调用_apportion_workers_to_tasks(…)获取每个task分到的workers数量（**此处未考虑workers的异构性，应对的仍然是同构性worker**）
    - 根据评分机制给每个worker分配对应的tasks
- 构建分布式进程组
    
    - 创建ProcessGroupManager对象，主要管理进程组的创建和分配，主要内容为分析分配给不同workers的task graph的拓扑，决定哪些rank需要进行通信，并根据通信模式决定进程组，并提供query这些配置的方法
- 初始化RayTrainer(main Trainer)
    
    - 分布式训练中使用ray实现的主要编排元件
    - 该类主要有如下作用
        - 验证任务图中所有组件的配置。
        - 管理跨节点的硬件资源（GPU）。
        - 初始化并管理 Ray actor（DAGWorkers）的生命周期。
        - 启动训练过程并监控其执行，直到完成或失败为止。
- trainer.start_workers()
    
    在初始化后，我们准备好了所有的DAGWorker(GPU)与其对应的taskgraph中所有的node_worker的初始化，随后调用每个DAGWorker的`execute_task_graph(…)`函数，该函数定义在siirl/workers/dag_worker/mixins/execution_mixin.py中
    
    ```python
    def execute_task_graph(self):
            """Main entry point to start the DAG execution pipeline."""
            logger.info(f"Rank {self._rank}: Starting DAG execution pipeline...")
            logger.success(f"Rank {self._rank}: All components initialized. Starting training loop from step {self.global_steps + 1}.")
    
            if self.val_reward_fn and self.config.trainer.val_before_train:
                # _validate handles multi-rank logic internally
                val_metrics = self._validate()
                if self._rank == 0 and val_metrics and self.logger:
                    logger.info(f"Initial validation metrics:\\n{pformat(val_metrics)}")
                    self.logger.log(data=val_metrics, step=self.global_steps)
    
                if self.config.trainer.val_only:
                    logger.info("`val_only` is true. Halting after initial validation.")
                    return
            self._run_training_loop()
    
            if self.progress_bar:
                self.progress_bar.close()
            self.taskgraph_execute_finished = True
            logger.success(f"Rank {self._rank}: DAG execution finished.")
    ```
    
    self._run_training_loop()负责实际训练流程，其中训练过程分为如下几步：
    
    - 利用dataLoader加载当次batch数据（如果非第一node则从dataBuffer获取）
    - 执行节点，调用Node.run()
    - 将下个节点所需的数据写入dataBuffer
    - 统计metrixs信息
    
    其中在node.run()中
    
    ```python
    def run(self, **kwargs: Any) -> Any:
            """
            Execute the task of the node.
            Args:
                **kwargs: Parameters passed to the executable function, usually the outputs of its dependent nodes.
            Returns:
                Any: The result of the node execution.
            """
            logger.debug(f"Starting to execute node: {self.node_id} (Type: {self.node_type.value}, Role: {self.node_role.value})")
            self.update_status(NodeStatus.RUNNING)
    
            if not self.executable:
                # For nodes without an executable reference, they may be handled by an external system,
                # or they are purely structural nodes (e.g., BARRIER_SYNC, whose logic is in the scheduler).
                # one implement for barrier...
                if self.node_type == NodeType.BARRIER_SYNC and kwargs.get("do_barrier", False):
                    import torch.distributed as dist
    
                    logger.debug(f"Node {self.node_id} block before barrier ...")
                    dist.barrier(group=kwargs.get("barrier_group", None))
    
                logger.debug(f"Node {self.node_id} has no executable function, skipping execution.")
                self.output = None  # Or set a specific output based on the node type
                return self.output
    
            try:
                # Prepare the parameters to be passed to the executable function
                # Here, the actual parameters can be constructed based on kwargs and node configuration
                # For example, only pass the outputs of dependencies
                # resolved_args = {dep_id: kwargs.get(dep_id) for dep_id in self.dependencies if dep_id in kwargs}
                # resolved_args.update(self.config) # Or pass the node configuration as well
    
                # Simplification: Pass all kwargs directly, and the user function handles them
                node_output = self._executable(**kwargs, node_config=self.config)
                self.output = node_output
                self.update_status(NodeStatus.COMPLETED)
                logger.debug(f"Node {self.node_id} execution completed.")
                return self.output
            except Exception as e:
                error_message = f"An error occurred while executing node {self.node_id}: {e}"
                self.update_status(NodeStatus.FAILED, error_message)
                # An exception can be raised here, or the scheduler can handle the FAILED status
                raise RuntimeError(error_message) from e
    ```
    
    其中通过self._executable获取对应的类进行执行
    
- trainer.init_workers()
    
    初始化DAGWorker在内的所有资源，代码如下：
    
    ```python
    def init_workers(self):
            """Initializes the resources and Ray actors required for training."""
            # Step 1: Create the resource pool based on the spec defined in __init__.
            self.resource_manager.create_resource_pool()
            # Step 2: Create the RayActorManager, which will be responsible for creating and managing the actual DAGWorker actors.
            ray_actor_manager_kwargs = {"ray_wait_register_center_timeout": self.base_config.trainer.ray_wait_register_center_timeout}
    
            self.ray_actor_manager = RayActorManager(
                resource_pool=self.resource_manager.get_resource_pool(self.global_pool_id),
                base_config=self.base_config,
                process_manager=self.process_group_manager,
                rank_taskgraph_mapping=self.rank_taskgraph_mapping,
                data_buffer_handles=self.data_buffer_handles,
                device_name=self.device_name,
                **ray_actor_manager_kwargs,
            )
    ```
    
    其中，resource_manager为ResourcePoolManager对象，其中保存了nnodes与num_gpus_per_node等值，所调用的create_resource_pool()函数如下
    
    ```python
    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool
    
        self._check_resource_available()
    ```
    
    其中resource_pool_spec的格式为 [self.base_config.trainer.n_gpus_per_node] * self.base_config.trainer.nnodes，大概为[8, 8 , 8, … ,8]，创建resource_pool，并将key指定为self.global_pool_id
    
    初始化RayActorManager对象，其中，resource_pool即为创建的管理资源池，base_config为具体的配置，process_manager为构建的分布式进程组，rank_taskgraph_mapping为得到的taskgraph和对应所用gpu的mapping数组，data_buffer_handles为data_buffer，device_name为cuda
    
    其中，指定的ray_actor_class为DAGWorker，随后调用`_initialize_workers(…)`函数
    
    ```python
    def _initialize_workers(self, resource_pool: RayResourcePool, bin_pack: bool) -> None:
            """
            Creates and configures all worker actors based on the resource pool.
    
            This method orchestrates the creation of placement groups, iterates through
            them to launch actors with the correct rank and environment variables,
            and establishes the master address for distributed coordination.
            """
            strategy = "STRICT_PACK" if bin_pack else "PACK"
            placement_groups = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
            sorted_pgs = sort_placement_group_by_node_ip(placement_groups)
    
            num_gpus_per_worker = 1 / resource_pool.max_colocate_count
            local_world_size = resource_pool.store[0]
            rank = -1
    
            for pg_index, placement_group in enumerate(sorted_pgs):
                if local_world_size > placement_group.bundle_count:
                    raise ValueError(f"Placement group for '{self.name_prefix}' has too few bundles ({placement_group.bundle_count}) to support the required local world size ({local_world_size}).")
    
                for local_rank in range(local_world_size):
                    rank += 1
                    worker = self._create_worker_actor(
                        rank=rank,
                        local_rank=local_rank,
                        local_world_size=local_world_size,
                        pg_index=pg_index,
                        placement_group=placement_group,
                        num_gpus_per_worker=num_gpus_per_worker,
                        use_gpu=resource_pool.use_gpu,
                        device_name=self.device_name,
                    )
                    self._workers.append(worker)
    
                    if rank == 0:
                        # Rank 0 worker is special: it establishes the master
                        # address and port for the entire worker group.
                        self._master_addr, self._master_port = self._get_register_center_and_master_info()
            work_futures = self.map_async(method_name="init_graph")
            ray.get(work_futures)
            # only support single agent
            if self.base_config.actor_rollout_ref.rollout.mode == 'async':
                tp_size = self.base_config.actor_rollout_ref.rollout.tensor_model_parallel_size
                world_size = len(self._workers)
                dp_size = world_size // tp_size
                self.async_rollout_manager = []
                futures = self._init_async_rollout_manaager(dp_size)
                
                loop = asyncio.get_event_loop()
                self.async_rollout_manager = loop.run_until_complete(futures)
                for manager in self.async_rollout_manager:
                    manager.sleep()
                for i in range(len(self._workers)):
                    if i % tp_size == 0:
                        ray.get(self._workers[i].set_async_rollout_manager.remote(self.async_rollout_manager[i // tp_size]))
    ```
    
    主要该函数做的主要操作如下：
    
    - 获取ray中的placement_groups，并按照node_id进行排序
        
    - 对于同一个placement_groups中的所有8个rank，调用`_create_worker_actor(…)`函数获取worker，并将其添加到_workers列表中，对于每个placement group的rank0，获取其master_addr和master_port，**注意，此处的workers list并不区分是否为同一个PG**
        
        获取的worker即为DAGWorker，同时由于DAGWorker为InitializationMixin, ExecutionMixin, NodeExecutorsMixin, ValidationMixin, UtilitiesMixin的子类，因而如上几类都会进行初始化，其中，在DAGWorker.init()中调用initializationMixin._initialize_worker()进行初始化
        
        ```python
        def _initialize_worker(self):
                """Orchestrates the ordered initialization of all worker components."""
                self._rank = self._get_and_validate_rank()
                self.taskgraph = self._get_taskgraph_for_rank(self.taskgraph_mapping)
                self._setup_distributed_environment()
                self._initialize_core_components()
                self._initialize_node_workers()
                self._profiler = DistProfiler(rank=self._rank, config=self.config.profiler)
        
                if self._rank == 0:
                    logger.info("Rank 0: Initializing tracking logger...")
                    from siirl.utils.logger.tracking import Tracking
        
                    self.logger = Tracking(
                        project_name=self.config.trainer.project_name,
                        experiment_name=self.config.trainer.experiment_name,
                        default_backend=self.config.trainer.logger,
                        config=self.config.to_dict(),
                    )
        ```
        
        在InitializationMixin中，根据FSDP与Megatron后端将具体需要import的类导入self.role_worker_mapping中，例如对于megatron，则为如下几个worker类
        
        ![image.png](attachment:ee41b853-5824-4736-8a97-b47dbbf08120:image.png)
        
        在_initialize_node_workers(…)函数中，实际根据node.node_role类型，进行初始化，随后，对所有worker调用`init_graph(…)`函数，也即InitializationMixin.init_graph(…)
        
        ```python
        def init_graph(self):
            # this is needed by async rollout manager
            self._set_node_executables()
            self.init_model()
            self._load_checkpoint()
            # Ensure all models are initialized and checkpoints are loaded before starting.
            dist.barrier(self._gather_group)
        ```
        
        init_model(…)，初始化每个DAGWorker中对应的taskgraph中的所有node，注意，每个node有属于自己的workers，这个worker为ActorWorker/RolloutWorker等等，并非DAGWorker，以RolloutWorker为例，会调用其node_worker的init_model(…)方法，具体定义在siirl/workers/megatron_workers中，其中，主要包括初始化hf_config与tf_config，获取generation_config，主要调用_build_rollout(…)函数获取rollout和sharding_manager
        
        ```python
        def _build_rollout(self, trust_remote_code=False):
                from torch.distributed.device_mesh import init_device_mesh
                infer_tp = self.config.rollout.tensor_model_parallel_size
                dp = self.world_size // infer_tp
                assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        
                if self.config.rollout.name == "vllm":
                    from siirl.workers.rollout.vllm_rollout import vLLMRollout
                    # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
                    # we will reorganize their weight format when resharding from actor to rollout.
                    rollout_device_mesh = init_device_mesh(get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
                    self.device_mesh = rollout_device_mesh
                    log_gpu_memory_usage("Before building vllm rollout", logger=None)
                    # local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.use_shm)
                    rollout = vLLMRollout(
                        model_path=self.local_path,
                        config=self.config.rollout,
                        tokenizer=self.tokenizer,
                        model_hf_config=self.hf_config,
                        device_mesh=rollout_device_mesh,
                        trust_remote_code=trust_remote_code,
                    )
                    log_gpu_memory_usage("After building vllm rollout", logger=logger)
        
                elif self.config.rollout.name in ["sglang", "sglang_async"]:
                    from siirl.workers.rollout.sglang_rollout import SGLangRollout
                    if self.config.rollout.name == "sglang_async":
                        warnings.warn(
                            "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                    rollout_device_mesh = init_device_mesh("cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp"))
                    self.device_mesh = rollout_device_mesh
                    # local_path = copy_to_local(self.config.model.path)
                    log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
                    rollout = SGLangRollout(
                        actor_module=self.local_path,
                        config=self.config.rollout,
                        tokenizer=self.tokenizer,
                        model_hf_config=self.hf_config,
                        trust_remote_code=trust_remote_code,
                        processing_class=self.processor if self.processor is not None else self.tokenizer,
                        device_mesh=rollout_device_mesh,
                    )
                    log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)
                else:
                    raise NotImplementedError("Only vllmRollout and SGLangRollout are supported with Megatron now")
                
                return rollout, None
        ```
        
        在SGLangRollout初始化中，主要有 1）初始化环境 2）初始化分布式环境 3）初始化推理引擎 4）初始化采样策略；
        
        在初始化完每个node_worker后，对于每个agent_group，如果agent_group同时拥有actor node和rollout node，则需要sharding_manager来做权重更新，其中主要调用`_setup_sharding_manager(…)`函数实现；
        
        其中，根据对应的backend将各个不同的sharding_manager存入sharding_manager_map中，并根据实际调用时的strategy和rollout_name初始化sharding manager，对应的sharding_manager即MultiAgentMegatronSGLangShardingManager
        
    - 调用每个DAGWorker的init_graph(…)，也即