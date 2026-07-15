## 概述

主要关注siiRL中，每个step中的执行流程，其中，主要关注perf的log信息是如何被收集与统计的

## 代码

在siiRL中，之前也有提过，在创建完distributed dataLoader、为Sub taskGraph分配完资源启动sub，会在主流水线main_dag.py中调用`trainer.start_workers()`并行的执行每个worker自己的任务；
![[Pasted image 20260408135913.png]]
`start_workers()`的逻辑比较简单，核心为调用先前初始化的ray_actor_manager中的DAGWorker中的`execute_task_graph(…)`函数，随后，根据调用回的结果构建futures列表，代表每个worker的执行任务状态(sub taskgraph)，调用ray.wait(…)不断更新remaining_futures列表，当列表为空时，视作所有的任务均已完成，其中`execute_task_graph(…)`函数定义如下：

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

其中，self._run_training_loop()则是遍历epoch与batch的主函数，其中主函数执行代码如下：

```python
def _run_training_loop(self):
        """
        The main loop that iterates through training steps and epochs.
        """
        
        # ... ...
        
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # with profiler_config as profiler:
            for batch_idx in range(self.dataloader.num_train_batches):
                # If resuming, skip batches that have already been completed in the starting epoch.
                if epoch == start_epoch and batch_idx < batches_to_skip:
                    continue

                if self.global_steps >= self.total_training_steps:
                    logger.info(f"Rank {self._rank}: Reached total training steps. Exiting loop.")
                    if self._rank == 0 and last_val_metrics:
                        logger.info(f"Final validation metrics:\\n{pformat(last_val_metrics)}")
                    return

                ordered_metrics = self._run_training_step(epoch, batch_idx)
                # profiler.step()

                self.global_steps += 1

                if ordered_metrics is not None:
                    is_last_step = self.global_steps >= self.total_training_steps

                    # Save checkpoint at the configured frequency.
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        self._save_checkpoint()

                    # (Logging and validation logic remains unchanged)
                    metrics_dict = dict(ordered_metrics)
                    if self.val_reward_fn and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        val_metrics = self._validate()
                        if self._rank == 0 and val_metrics:
                            metrics_dict.update(val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics

                    if self.enable_perf:
                        self._aggregate_and_write_performance_metrics(metrics_dict)

                    ordered_metric_dict = self.format_metrics_by_group(metrics_dict, DAGConstants.METRIC_GROUP_ORDER)
                    self._log_core_performance_metrics(ordered_metric_dict, self.global_steps)
                    if self._rank == 0:
                        if self.logger:
                            self.logger.log(data=ordered_metric_dict, step=self.global_steps)
                        else:
                            self._log_metrics_to_console(ordered_metric_dict, self.global_steps)

                if self.progress_bar and not (epoch == start_epoch and batch_idx < batches_to_skip):
                    self.progress_bar.update(1)

        if self._rank == 0 and last_val_metrics:
            logger.info(f"Final validation metrics:\\n{pformat(last_val_metrics)}")
```

其中，ordered_metrics是具体该epoch内的对应batch执行单步算法后返回的log矩阵，具体为调用`self._run_training_step(epoch, batch_idx)`获得；

`_run_training_step(…)`中主要做如下操作：

- 加载数据：调用`DataProto.from_single_dict(…)`函数获得，其中self.dataloader中根据epoch维护一个具体的iterator，每次根据next获取下一个位置的batch
    
- 图遍历：获取该DAGWorker对应的subTaskGraph的节点队列，获取剩余待执行节点，存放在node_queue中，知道node_queue为空前不断循环，每次取出第一个node，获取其3d并行的size与rank号
    
- 获取input数据：如果不是node_queue首个node，则调用get_data_from_buffers获取batch，并删去dataProto中的prefix（为了索引取数据唯一）
    
- 执行节点：如果是计算reward或者advantage，直接调用对应函数计算，不然，则调用节点对应的run()函数执行对应model功能
    
    ```python
    node_output = cur_node.run(batch=batch, worker_group_index=cur_node.agent_group)
    ```