## SGLang的transfer调用时机

调用/v1/chat/completions接口时，会走到generate接口 --> `_global_state.tokenizer_manager.generate_request()` --> `self._send_one_request()` --> `self.send_to_scheduler.send_pyobj()`，将request发送到两个server的scheduler。

KVPoll的status

![image.png](attachment:b5bf3fd5-47cc-4e60-a54e-bb347a412288:image.png)

KVManager为每个prefill与decode最早便创建的对象，当有一个request到来时，再为每个request建立一个KVSender和一个KVReceiver，KVSender和KVRecevier标识着一对request的结束，用来在应用层监听

在KVSender收到了对应的notify信息后，会开始prefill的计算，当计算完后，调用到batchTransferSync(…)下发发送任务

### 调用链

主thread

每个transferEngine有一个multiTransport，multiTransport有多个对应的transport（包含RdmaTransport），RdmaTransport有多个RdmaContext（网卡），每个RdmaContext有多个RdmaContext，每个RdmaContext有多个WorkerPool

```bash
transferEngine→batchTransferSync
# add task within max_retry_time
transferEngine→submitTransfer
# add transfer to multiTransport(do nothing)
multiTransport→submitTransfer
# add transfer to different transfer type(rdma/hccl)
RdmaTransport→submitTransferTask
# add transfer task to related rdma context(device)
# split task into serveral slices with kBlockSize(65536, 2^18)
RdmaContext->submitPostSend
# add slices of device to its worker_pool(each device have one worker pool)
WorkerPool->submitPostSend
# add to slice_list_map[shared_id]
```

worker thread

```bash
workerPool->transferWorker
# use transferWorker thread to Monitor if there are datas to transfer
workerPool->performPostSend
# send slices in local_slice_queue
RdmaEndPoint->submitPostSend
# issue ibv_post_send
```

worker thread poll

## Mooncake发送的buffer注册与qp建联

### qp建联

粒度为RdmaEndpoints

`int RdmaEndPoint::setupConnectionsByActive()`函数建联

主要关注如何交换qp信息

![image.png](attachment:3478f02e-4f68-4d75-a739-758ee3c17d92:image.png)

主要通过调用MooncakeEngine中的sendHandShake来交换双方信息，需要交换的信息有本地的nic_path，对端的nic_path以及本地的qp_num(标识对应的qp，而非qp数量），调用的对象为endpoints，而endpoints可以从context→endpoints[peer_server_name]获取，每对端点之间使用不同的qp，**每个endpoints唯一标识一个对端结点，可以通过peer_nic_path_指定**

```cpp
// get endpoint by name
auto endpoint = context_.endpoint(entry.first);
```

```cpp
int TransferMetadata::sendHandshake(const std::string &peer_server_name,
                                    const HandShakeDesc &local_desc,
                                    HandShakeDesc &peer_desc) {
    RpcMetaDesc peer_location;
    if (getRpcMetaEntry(peer_server_name, peer_location)) {
        return ERR_METADATA;
    }
    auto local = TransferHandshakeUtil::encode(local_desc);
    Json::Value peer;
    int ret = handshake_plugin_->send(peer_location.ip_or_host_name,
                                      peer_location.rpc_port, local, peer);
    if (ret) return ret;
    TransferHandshakeUtil::decode(peer, peer_desc);
    if (!peer_desc.reply_msg.empty()) {
        LOG(ERROR) << "Handshake rejected by " << peer_server_name << ": "
                   << peer_desc.reply_msg;
        return ERR_METADATA;
    }
    return 0;
}
```

handshake_plugin的send为同步阻塞操作，因而可以保证建联操作两端有一个影式同步，handshake_plugin为SocketHandShakePlugin结构，实际调用的send函数为doSend，其中涉及将local发送至对端，并接收对端的peer，此处本质是阻塞的同步

```cpp
int doSend(struct addrinfo *addr, const Json::Value &local,
               Json::Value &peer) {
        int conn_fd = -1;
        int ret = doConnect(addr, conn_fd);
        if (ret) {
            return ret;
        }

        ret = writeString(conn_fd, HandShakeRequestType::Connection,
                          Json::FastWriter{}.write(local));
        if (ret) {
            LOG(ERROR)
                << "SocketHandShakePlugin: failed to send handshake message: "
                   "malformed json format, check tcp connection";
            close(conn_fd);
            return ret;
        }

        Json::Reader reader;
        auto [type, json_str] = readString(conn_fd);
        if (type != HandShakeRequestType::Connection) {
            LOG(ERROR)
                << "SocketHandShakePlugin: unexpected handshake message type";
            close(conn_fd);
            return ERR_SOCKET;
        }

        if (!reader.parse(json_str, peer)) {
            LOG(ERROR) << "SocketHandShakePlugin: failed to receive handshake "
                          "message: "
                          "malformed json format, check tcp connection";
            close(conn_fd);
            return ERR_MALFORMED_JSON;
        }

        close(conn_fd);
        return 0;
    }
```

### Buffer注册

粒度为RdmaContext，设备粒度，不同的endpoints可以共用同一个rkey和lkey，通过调用`RdmaContext::registerMemoryRegion(…)`可以注册一块内存，将注册后的内存句柄放入memory_region_list中，在需要发送时，根据对应的add找到对应的mr

```cpp
int RdmaContext::registerMemoryRegion(void *addr, size_t length, int access) {
    if (length > (size_t)globalConfig().max_mr_size) {
        PLOG(WARNING) << "The buffer length exceeds device max_mr_size, "
                      << "shrink it to " << globalConfig().max_mr_size;
        length = (size_t)globalConfig().max_mr_size;
    }
    ibv_mr *mr = ibv_reg_mr(pd_, addr, length, access);
    if (!mr) {
        PLOG(ERROR) << "Failed to register memory " << addr;
        return ERR_CONTEXT;
    }

    RWSpinlock::WriteGuard guard(memory_regions_lock_);
    memory_region_list_.push_back(mr);
    return 0;
}
```

获取lkey和rkey的函数如下，仅展示lkey部分函数，rkey同理

```cpp
uint32_t RdmaContext::lkey(void *addr) {
    RWSpinlock::ReadGuard guard(memory_regions_lock_);
    for (auto iter = memory_region_list_.begin();
         iter != memory_region_list_.end(); ++iter)
        if ((*iter)->addr <= addr &&
            addr < (char *)((*iter)->addr) + (*iter)->length)
            return (*iter)->lkey;

    LOG(ERROR) << "Address " << addr << " lkey not found for " << deviceName();
    return 0;
}
```

调用registerMemoryRegion(…)的时机为在RdmaTransport::registerLocalMemory(…)函数中调用，在函数中，会把每个context(device)均对buff注册内存，并将lkey和rkey存到buffer_desc的lkey和rkey的队列中

context_list_的初始化在RdmaTransport::initializeRdmaResources(…)

## Mooncake ibgda集成

### 通信comm添加

p2pibgda通信的核心结构为p2pComm，每对p2p维护一个该结构，考虑在RdmaEndPoints中添加该结构（每对连接粒度，RdmaContext为网卡粒度，针对不同的对端ip与网卡，有不同的RdmaEndPoints，其中存放实际建联的qp与cq等）

新增ibgdaEndPoints