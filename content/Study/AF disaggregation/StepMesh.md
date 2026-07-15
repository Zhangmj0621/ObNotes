每个GPU都是一个worker或者server

即public.hpp中的fworker/fserver

ps是node的概念，初始化的时候又worker_rank, server_rank，长度为GPU的数量

其中instance_id指的是同号的GPU

num_workers_与num_servers_指节点数量

Postoffice::GetWorker(instance_id)，其中postOffice一个node一个，根据local gpu id获取instance

每个fworker有自己的pushpull_queue_(gpu粒度）

push_tensors.size()：push tensor的数量

pull_batch_size：pull tensor的数量

发送时候，每个server都发送msg，msg里包含全量put_tensors

```cpp
// broadcast
    for (int i = 0; i < server_count; i++) {
      kv_.SendMsg(msg, i);
    }
```

SendMsg发送时候，貌似发送只存在同号GPU内

```cpp
  void SendMsg(Message& msg, int dst) {
    int group_server_rank = dst;
    int instance_server_id = postoffice_->GroupServerRankToInstanceID(
        group_server_rank, instance_idx_);

    msg.meta.app_id = obj_->app_id();
    msg.meta.customer_id = obj_->customer_id();
    msg.meta.recver = instance_server_id;
    postoffice_->van()->Send(msg);
  }
```

随后调用到rdma_van.h中的SendMsg(…)函数