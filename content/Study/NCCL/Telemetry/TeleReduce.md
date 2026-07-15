### 背景和目的

为了更好的监测GPU集合通信时的网络情况，我们期望能够在NCCL里添加细粒度的GPU间点对点流量测量。在nccl中，GPU之间的通信可以理解为多次send/recv操作，每次操作绑定到具体的网卡（NIC）上执行，具体的数据传输流程由多对QP完成。在其中，每次QP任务的下发会由app生成WR（work request）告知具体的网卡，因此，我们期望能够统计端口上的WR数量，通过WR的变化趋势得知工作队列的长度变化，进而推测出当前网络的拥塞情况。此外，我们同时统计了端口上工作队列中剩余未发送的数据大小，并根据wc(work complete)的产生精确计算出了目前的GPU间点对点流量带宽。

### 设计与实现

### wr端口数量及剩余未发送数据大小统计

对每个proxy线程，设置`thread_local`变量数组sendWrCounter与remainWrDataSize，`NCCL_IB_MAX_DEVS_PER_NIC`大小为网卡上最大端口数量

```
// count the number of send wrs and remain data sizes in wrs
thread_local int sendWrCounter[NCCL_IB_MAX_DEVS_PER_NIC] = {0};
thread_local int remainWrDataSize[NCCL_IB_MAX_DEVS_PER_NIC] = {0};
```

在数据send过程中，每个req对应一个wr，nccl将每个wr以链式结构连接，并在最后添加一个lastwr以获得wc（只有lastwr完成才会产生wc，wc的产生意味着所有串接的wr均已完成）。

在发送过程中，每个req会用多个qp进行分割，每个qp对应着固定的网卡端口，因此我们可以准确得知每个req对应的需要在不同端口上发送的数据大小。同时，为了便于统计以及观察wr数量变化，我们假设同一个wr经过n个qp划分后会产生（n+1）个wr，因此，我们依据该逻辑在数据send时增加`wrCounter`以及`wrDataSize`。

```
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
    ......

    for (int i = 0; i < nqps; i++) {
        int qpIndex = comm->base.qpIndex;
        ncclIbQp* qp = comm->base.qps + qpIndex;
        int devIndex = qp->devIndex;

        for (int r=0; r<nreqs; r++) {
            ......

            // update sendWrCounter
            sendWrCounter[devIndex]++;

            ......
            int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
            int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);

            //update remainWrDataSize
            if (length > 0) {
                remainWrDataSize[devIndex] += length;
            }

            ......
        }
    }

}
```

在`ncclIbItest`中，会检测对应req是否已经获取wc，我们可以在收到对应的wc以后，同步减少端口上的`wrCounter`以及`wrDataSize`。

```
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  ......
  while (1) {
    if (r->events[0] == 0 && r->events[1] == 0) {
      ......
    }

    int totalWrDone = 0;
    int wrDone = 0;
    struct ibv_wc wcs[4];

    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      // If we expect any completions from this device's CQ
      if (r->events[i]) {
        NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
        totalWrDone += wrDone;
        if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
        if (wrDone == 0) continue;
        for (int w=0; w<wrDone; w++) {
          struct ibv_wc *wc = wcs+w;
          if (wc->status != IBV_WC_SUCCESS) {
            ......
          }

          ......

          if (req->type == NCCL_NET_IB_REQ_SEND) {
            // update sendWrCounter
            sendWrCounter[i] -= req->nreqs;

            for (int j = 0; j < req->nreqs; j++) {
              struct ncclIbRequest* sendReq = r->base->reqs+((wc->wr_id >> (j*8)) & 0xff);
              if ((sendReq->events[i] <= 0)) {
                WARN("NET/IB: sendReq(%p)->events={%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], i, j);
                ret = ncclInternalError;
                goto ret;
              }
              sendReq->events[i]--;
              if(global_timer_log.collect && sendReq->log[i].loged_start == NCCL_LOG_TELEMETRY && !sendReq->events[i]){
                clock_gettime(CLOCK_REALTIME, &sendReq->log[i].send_end);
                sendReq->log[i].diff = 1000000000L * (sendReq->log[i].send_end.tv_sec) + sendReq->log[i].send_end.tv_nsec;
                sendReq->log[i].sendWrCounter = sendWrCounter[i];
                sendReq->log[i].devIndex = i;

                //sendReq has completed transport, update remainWrDataSize
                remainWrDataSize[i] -= sendReq->log[i].size;
                sendReq->log[i].remainWrDataSize = remainWrDataSize[i];

                if((sendReq->log[i].func == 2 || sendReq->log[i].func == 3) && sendReq->log[i].size > 16){
                  // save log
                  pthread_mutex_lock(&global_timer_log.lock);
                  __sync_synchronize();
                  global_timer_log.push(sendReq->log[i]);
                  pthread_mutex_unlock(&global_timer_log.lock);
                }
              }
            }
          } else {
            ......
          }
        }
      }
    }

}
```

随后利用`timerLogService`进行异步输出到log文件中，输出的格式如下：

```
"%s: Group:%lu, from rank %d to rank %d, devIndex: %d, channel_id:%d, func:%u,FuncTimes:%lld, %d.%d.%d.%d->%d.%d.%d.%d, bandWidths: %d Gbps, sendWrcounter: %d, remainWrDataSize: %d Bytes, timestamp: %lld nsec"
```

### 流量带宽计算

在每次调用`ncclIbItest`时，若req->events[i]清零，则意味着req对应数据已经传送完成，我们记录当前时间戳以及发送完成的数据，将其存入log存入队列中，在`timerLogService`中，我们取出队列中的元素，并维护一个滑动窗口来统计实际的带宽（**由于每次log间的时间间隔内获得的size，只能意味着该req在该时间间隔内完成，并不意味着整个req传输只花费了该时间间隔，同时也不意味着该时间间隔只用来发送该req对应数据**）。我们设置滑动窗口大小（`MAXWINDOWSIZE`，通常设置为50），该滑动窗口为一组log的集合，我们认为该滑动窗口内所发送完成的数据，以及该窗口的时间间隔之比，是较为精确的流量带宽计算，同时，每次由新log到来时，我们将窗口右移，能够在常数时间内计算出每一时刻的流量带宽。

```
int getBandWidths() {
    if (slideWindow.size() <= 1) {
      return 0;
    }

    // Gbps
    unsigned long long sendTime = sendEndTime - slideWindow.front().diff;
    unsigned long long sendDataSizes = windowDataSizes - slideWindow.front().size;
    // 0.93 = 1e9 / 1024 * 1024 * 1024
    return sendDataSizes * 0.93 * 8 / sendTime;
  }
```

### 测试结果

### 测试环境

西云87、西云94两台单机八卡服务器，需要利用VPN连接

```
跳板机：<http://100.64.27.220>
用户名：westhpc
密码：Tywl@123
```

```
100.64.24.87
100.64.24.94
用户名：root
密码：Tywl@123
```

### 测试流程

- 通过ssh连接服务器
- 进入docker环境，可能的指令为：`docker exec -it test bash`
- 在**/work/infrawaves/nccl-test**目录下，执行以下命令，进行4打4的all_gather测试，数据大小8G

```
mpirun -np 8 --host 100.64.24.87:4,100.64.24.94:4 --allow-run-as-root -mca btl ^openib -x NCCL_IB_GID_INDEX=3  -x NCCL_IB_TC=160  -x NCCL_NET_GDR_LEVEL=4 -x LD_LIBRARY_PATH=/workspace/infrawaves/zhangmj/vccl_2.21.51x/build/lib/ -x NCCL_TELEMETRY_ENABLE=1 -x NCCL_TELEMETRY_LOG_PATH=/workspace/infrawaves/Train/logs/ ./build/all_gather_perf -b 8G -e 8G  -f  2 -g 1 -w 0
```

LD_LIBRARY_PATH可替换为你自己的编译lib路径

- 测试结果类似如下：

![](https://cdn.nlark.com/yuque/0/2024/jpeg/42492869/1730973957316-1b344ef8-9cbb-4f66-a769-76546a5c474d.jpeg)

其中包含网卡名称、rank、端口号、channel Id、带宽等等信息。