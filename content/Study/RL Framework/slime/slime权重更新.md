本文关注megatron backend下，异步模式的megatron权重更新

核心update_weights函数在MegatronTrainerRayActor类中

```python
@timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            reload_process_groups()

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    print("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    for name in self.weights["old_actor"]:
                        self.weights["old_actor"][name].copy_(self.weights["rollout_actor"][name])
                    # Then copy current actor to rollout_actor
                    self.update_cpu_params_dict(self.weights["rollout_actor"])
                else:
                    self.update_cpu_params_dict(self.weights["old_actor"])

        if self.args.offload_train:
            destroy_process_groups()
```

其中有个非常有意思的设计，那就是reload通信组与destory通信组，核心思路是将torch.distribute.process_group包装成了ReloadableProcessGroup类，reload本质上就是根据group_info重建一个通信组，destroy就是删除原先通信组；据悉改原因是为了节省显存，可以尝试开启nccl_lazy_connection观察显存占用情况；

weights_updater为UpdateWeightFromDistributed类，其中connect_rollout_engines函数，获取目前rank是否为pp_src_rank，即每个pp stage对应的第一个rank，可以抽象理解为tp=0, ep=0时的所有rank，在其中建立每个pp_srck_rank与所有sglang_engine的通信组，确保建立通信组后在所有rank间做一次gloo的barrier强同步

随后核心调用weight_updater.update_weights()函数更新权重，对于每个engine，调用/pause_generation和/flush_cache接口，释放所有显存和停止所有正在生成的请求，随后调用_update_weight_from_distributed函数，在其中，分别更新非MoE权重和MoE权重；

```python
@torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_parameters(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_parameters(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
```

其中，不断累积参数，直到达到了self.args.update_weight_buffer_size后，则进行一次桶更新，即调用_update_expert_bucket_weights_from_distributed(…)更新一次参数

**注意：分桶是为了减少调用sglang的update_weights_from_distributed api的次数**

```python
def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self.args,
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)
```

其中核心调用update_weights_from_distributed(…)函数进行权重更新，其中self._group_name即为”slime-pp_{pp_rank}”，self._model_update_groups即为pp_src_rank和所有的rollout_engine（每个engine一个rank，真实的发送应该是specific训练GPU到specific推理GPU）

```python
def update_weights_from_distributed(
    args: Namespace,
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
```

其中refs本质上是调用每个sglangEngine的update_weights_from_distributed方法，让每个engine接受包括names, dtypes, shapes和group_name的元数据，随后等待group_name的processGroup来接收数据；随后实际由pp_src_rank进行到每个sglangEngine的参数的broadcast

目前slime中的fp8量化在convert_to_hf函数中调用quantize_params(…)实现