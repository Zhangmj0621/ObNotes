## Background

在pd分离场景，kvCache需要从prefill node发送到decoding node来避免重复计算，通过阅读sglang & Mooncake论文与源码，可能的KVCache发送主要有两部分组成：

- prefill node在计算完每个chunk时，会并行的将每layer的KVCache发送给decoding node，并在最后一个chunk计算完后，额外发送一个aux data
- 在Mooncake提出的KVCache-centric Scheduling Algorithm中，从prefill pool中选取prefill node的策略并不再仅仅根据prefix len或node load，此时若选取非prefix len最长的prefill node，则存在从best prefix len的prefill node到实际选取的prefill node的差值KVCache的发送（这一部分在mooncake开源的代码中似乎并没有得到体现，sglang也暂时并没有集成，不确定这一部分的通信能否被计算overlap掉，**在论文中，这一部分KVCache的发送对于the automatic replication of hot-spot caches非常重要**）

下面从sglang的视角看mooncake中通信的整个过程。

## Overall Transfer in SGLang

在sglang中，PD分离实际上是一个Producer-Consumer的模式，使用多个transfer_queue来实现non-blocking的kv transfer，在SGLang中，有**Mooncake和NIXL两种不同的backend**实现方式，其中Connector有四种角色

|KVBootstrap|用于prefill，记录prefill与decode交互信息，decode会请求该信息来建联|
|---|---|
|KVManager|prefill与decode均有，用于控制|
|KVSender|用于prefill，发送KVCache|
|KVReceiver|用于Decode，与prefill建联并接收KVcache|

### Prefill与decoding node建联与收发

SGLang中prefill与decode的建联大致示意图如下主要过程由KVManager实现，在此不加赘述，其中，在KVManager初始化的时候，会启动**TransferEngine来做实际通信（即MooncakeTransferEngine），**下面描述在prefill与decode建联后的通信过程：

![image.png](attachment:9ad93bda-4f63-4955-8a71-902de16daed1:image.png)

- 在Scheduler中，对于每个req，prefill与decode会生成一对KVSender和KVReceiver来与req绑定
- 在event_loop中，prefill与decode会分别作如下操作：
    - prefill会从bootstrapQueue中取出req，获取status，对于status特定等于WaitingForInput的req，初始化KVSender，将该req移动到WaitingQueue，等待Decode notify
    - decode会调用preallocated预分配KVCache，随后向prefill的manager发送自己的ip与port等信息，将自己的req移动到TransferQueue
    - prefill收到了notify后，同样将req移动到TransferQueue
- 当prefill计算完chunk后，会调用`send_kv_chunk(…)`再到KVSender中的send函数来发送，send的实现为调用`add_transfer_request(…)`将待发送的信息添加到KVManager的队列中，在KVManager中初始化的transfer_worker线程中，会从队列中取出request，并调用`send_kvcache`（use for KVCache)与`send_aux`(use for aux data)来实现，随后由初始化的executor来执行MooncakeTransferEngine中的`transfer_sync(…)`来实际发送每一layer的KVCache
- 在prefill与decode的event_loop中，会不断poll每个req，通过KVPoll的结果来判断inflight的KVCache是否成功收发

主要关注`transfer_sync(…)`在Mooncake backend的具体实现

## Overall Transfer in Mooncake

本节仅关注RDMA通信模式，Mooncake支持的通信模式还有cxl、nvlink、nvmeof与tcp。

### transferSync

`transfer_sync(…)`实际上是调用Mooncake中的`tranfer_sync_write(…)`函数实现，而`transfer_sync_write(…)`在Mooncake中的具体实现为`transferSyncWrite`

```cpp
auto adaptor_cls =
        py::class_<TransferEnginePy>(m, "TransferEngine")
        // ... ...
        .def("transfer_sync_write", &TransferEnginePy::transferSyncWrite)
        .def("transfer_sync_read", &TransferEnginePy::transferSyncRead)
        // ... ...
        ;
```

`transferSyncWrite(…)`直接调用`transferSync(…)`实现，在`transferSync(…)`中，首先通过目标主机的name获取handle，随后发送分为如下几步：

- 分配batch_size为1的batch，填充entry信息（包括发送长度，地址等）
- 调用`submitTransfer(…)`函数，该函数**实际上做的是将request分为多个slice，添加到slice_queue中**，实际的发送在WorkerPool中的transferWorker线程调用`performPostSend(…)`执行（详细步骤在下面介绍）
- **阻塞该线程（Mooncake所谓的non-blocking似乎并不是完全意义上的non-blocking)**，不断poll该batch的完成情况（ 该batch的slice success与fail数由transferWorker线程轮询调用`performPollCq(…)`更新），直到确认该batch的状态为Success或Fail或TImeout
- 若成功，直接返回，若失败，则在不超过MAXRETRY_Cnt时，会尝试选择其他网卡重新进行通信

### submitTransfer

`submitTransfer(…)`根据request，封装对应的task，并调用`selectTransport(…)`函数选取通信类型(RDMA、TCP等），通过调用`submitTransferTask(…)`，将待发送数据，根据slice_size切成多个slice，将其添加到所选网卡的slices_to_post队列中，调用`submitPostSend(…)`进行进一步处理

```cpp
// src/transport/rdma_transport/rdma_transport.cpp
Status RdmaTransport::submitTransferTask(
    const std::vector<TransferRequest *> &request_list,
    const std::vector<TransferTask *> &task_list) {
    // ... ...
    for (size_t index = 0; index < request_list.size(); ++index) {
        assert(request_list[index] && task_list[index]);
        auto &request = *request_list[index];
        auto &task = *task_list[index];
        for (uint64_t offset = 0; offset < request.length;
             offset += kBlockSize) {
             // ... ...
             while (retry_cnt < kMaxRetryCount) {
                if (selectDevice(local_segment_desc.get(),
                                 (uint64_t)slice->source_addr, slice->length,
                                 buffer_id, device_id, retry_cnt++))
                    continue;
                assert(device_id >= 0 && device_id < context_list_.size());
                auto &context = context_list_[device_id];
                assert(context.get());
                if (!context->active()) continue;
                assert(buffer_id >= 0 && buffer_id < local_segment_desc->buffers.size());
                assert(local_segment_desc->buffers[buffer_id].lkey.size() == context_list_.size());
                slice->rdma.source_lkey =
                    local_segment_desc->buffers[buffer_id].lkey[device_id];
                slices_to_post[context].push_back(slice);
                task.total_bytes += slice->length;
                // task.slices.push_back(slice);
                __sync_fetch_and_add(&task.slice_count, 1);
                found_device = true;
                break;
            }
            // ... ...
```

在submitPostSend(…)中，将slice添加到对应的slice_queue_中，在此不加赘述

### transferWorker

每个RDMA context(per nic)会有globalConfig().workers_per_ctx(在moonCake中为2）个transferWorker thread来实际执行收发，多个thread均分slice_queue_中的tasks

```cpp
void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    const static uint64_t kWaitPeriodInNano = 100000000;  // 100ms
    uint64_t last_wait_ts = getCurrentTimeInNano();
    while (workers_running_.load(std::memory_order_relaxed)) {
        auto processed_slice_count =
            processed_slice_count_.load(std::memory_order_relaxed);
        auto submitted_slice_count =
            submitted_slice_count_.load(std::memory_order_relaxed);
        if (processed_slice_count == submitted_slice_count) {
            uint64_t curr_wait_ts = getCurrentTimeInNano();
            if (curr_wait_ts - last_wait_ts > kWaitPeriodInNano) {
                std::unique_lock<std::mutex> lock(cond_mutex_);
                suspended_flag_.fetch_add(1);
                cond_var_.wait_for(lock, std::chrono::seconds(1));
                suspended_flag_.fetch_sub(1);
                last_wait_ts = curr_wait_ts;
            }
            continue;
        }
        performPostSend(thread_id);
#ifndef USE_FAKE_POST_SEND
        performPollCq(thread_id);
#endif
    }
}
```

### performPostSend

performPostSend执行实际收发，每个transferWorker thread执行一半的slice_queue_中的任务，在实际收发时，根据slice_count与可用的wr_count尽可能多的发送slice，具体发送调用`ibv_post_send(…)`实现。

### performPollCq

可理解为从cq中poll获得wc的过程，来判断某个req是否完成，具体调用`ibv_poll_cq(…)`实现。

### Fault Tolerance

在Mooncake中，同样有数据链路级别的容错，具体的思路为：

- 在尝试发送时，根据实际的Topo随机Preferred NIC list中的某个网卡进行发送
- 若失败，则将失败网卡的失败计数+1，尝试使用次优网卡重发任务，发送前，若未建联，会尝试建联
- 若所有网卡均发送失败，则tasks失败
- 若某张网卡失败计数超过阈值，则将其暂时从可访问网卡列表中删去，等待后续尝试恢复

其中，由于Mooncake的transfer是同步的，因此线程在处理该task的时候，并不涉及其他task的处理，**Mooncake的P2p transfer可理解为尽一切方法尝试发送成功**。

## Conclusion & Sth to do

纯通信侧优化（如果要利用ibgda实现p2p）

1. Mooncake目前实现的transfer是纯同步的，在layer层数较大且thread数较少场景，对于后续的request，TTFT会非常差，且这点在长文本在前，短文本在后的case，用户体验可能更不佳（短文本user可能更期望更好的TTFT？我们可否考虑根据长文本和短文本的长度关系，调度收发顺序？）
2. 在Mooncake中，KVCache的收发为纯粹的单边P2p操作，receiver侧通过在send侧的request完全完成后，通过KVSender向KVReceiver发送一个TCP消息来告知Receiver，考虑能否将该次TCP通知改为类似nvshmem中atomic signal，同样由RDMA来完成
3. 按照Mooncake论文中所讲，prefill侧的通信可以完全被compute overlap，但是对于decoding node，First token的generate仍然很大受限于KVCache的发送，纯优化通信部分带宽，预期TTFT依然还是会有很大收益
4. Mooncake中的容错，思路与VCCL中的容错大差不差，但是从实现上来看，单边通信上容错的实现成本确实比VCCL要低很多，照搬到IBGDA上似乎并不是难事
5. 由于当次计算的KVCache实际上数据量并不大，因此很难打满带宽，对于Mooncake，一种很重要的概念是the automatic replication of hot-spot caches**， 能否存在一种预测host-spot caches的算法，**在发送当次KVCache时，将该prefill node上预测hot-spot的cache一并发送至decode node，由于数据量原先较小，多发部分data，似乎并不会影响time(datasize增大时，发送速率可能也上升）

KVCache-centric schedule优化

1. 目前的schedule根据prefix length和load做了一个balance，目的是为了**the automatic replication of hot-spot caches，**但在特定case，会不会存在mislead？由于特定的load场景，导致目前仍然为贪心的算法找到的非最优解