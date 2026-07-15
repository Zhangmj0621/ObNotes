# 设计背景

在一次通信操作中，需要理解对齐粒度。在NCCL中，节点被划分为不同的`group`，每个节点会有一个唯一的`rank`来标识身份。NCCL的通信主体是节点之间的通信，即GPU与GPU之间的通信（具体概念可参考MPI）。

在通信过程中，操作可以理解为多次`send/recv`操作（此处仅考虑不同主机上的GPU通信）。每个节点通过绑定的网卡（NIC）与其他节点通信。每个节点绑定了固定的NIC，通信通过实际的网络进行。在此过程中，一次数据的发送（`send`）会被划分为多个`channel`（通常是对数据进行均匀分割），在每个`channel`中，会生成多个`req`，并通过多个`QP`（Queue Pair）指示RDMA NIC进行数据的传输。发送请求**可以并行处理，因为数据的实际传输是异步的**，只需要将相应的`req`分配给指定的`QP`即可。

当`req`被分发到具体的`QP`时，会产生`wr`（Work Request），用于指示网卡硬件进行实际的收发操作。因此，可以通过**统计当前每个端口上的`wr`数量来判断工作队列的长度，并根据其变化推测当前网络的拥塞情况**。

# 设计思路

## 设定统计变量

由于多个channel是串行的，共享一个线程，因而可以设定一个全局的thread_local变量，来存放目前的wr数量

```
// count sendWr
thread_local int sendWrCounter[NCCL_IB_MAX_DEVS_PER_NIC] = {0};

```

需要注意的是，我们的wr统计是**针对于端口**而言的，一张网卡会有多个端口

## 新增wr count

在ncclIbIsend中，我们在QP分割req发送的时候，添加具体的端口上wr数量，原理是每个QP绑定固定的devIndex，每个req通过QP分割发送，此时将wrCount[devIndex]++

```
for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    ncclIbQp* qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;

    for (int r=0; r<nreqs; r++) {
      ......

      // update sendWrCounter
      sendWrCounter[devIndex] ++;

      ......
    }

    ......
}

```

## 减少wr count

在ncclIbTest中，我们可以检测具体的req是否发送成功，如果发送成功，我们会生成对应的cq，通过cq，我们可以得知对应的wr已经成功被消耗，此时将wrCount[devIndex]--

```
for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      // If we expect any completions from this device's CQ
      if (r->events[i]) {
        ......

        for (int w=0; w<wrDone; w++) {

          ......

          if (req->type == NCCL_NET_IB_REQ_SEND) {
            // update sendWr
            sendWrCounter[i] -= req->nreqs;

            ......
          }

        ......
        }

      }

  }

```

## 监测wr count

在`struct time_log`结构体中，新增`devIndex`和`wrCount`

```
struct timer_log{
  ......
  int devIndex;
  int sendWrCounter;
};

```

在time_log.cc中，修改打印到文件中的log信息，新增端口号以及wr数量信息

```
sprintf(buffer, "%s: Group:%lu, from rank %d to rank %d, channel_id:%d, func:%u,FuncTimes:%lld, %d.%d.%d.%d->%d.%d.%d.%d send %d Bytes used %lld nsec, devIndex:%d, sendWrCounter:%d",
                strTime.c_str(),log.groupHash,log.rank,log.peerRank,log.channel_id,log.func,log.ncclFuncTimes,
                log.srcIp[0],log.srcIp[1],log.srcIp[2],log.srcIp[3],
                log.dscIp[0],log.dscIp[1],log.dscIp[2],log.dscIp[3],
                log.size,log.diff,
                log.devIndex,
                log.sendWrCounter);

```

## 统计端口剩余未发送数据

难点

- 每个req对应一个wr，wr之前练成链式结构，仅由最终的lastwr生成wc告知发送成功
- 然而发送过程由qp进行分割，每次携带req一定量的数据，最终会生成nqp个wc
- 每次生成wc时，如何判断该次wc对应的qp发送了多少数据，若非最后一次分割发送的qp，则每个req分割的size应该为`DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align`
- 难点在于判断获得的wc对应的是第几次的发送（目前来看只能近似？）

设计方案

- 在每个req中新增`remainRecv[devIndex]`以及`chunkSize`
- 在itest中，对对应的wc的req进行判断，如果`remainRecv[devIndex] > 0`，则认为我这一次成功发出了`min(chunksize, remainRecv[devIndex])`的数据
- `remainWrDatasize`对齐粒度为req中的event清零，此时将`data--`。
- 在`timer_log_service`线程中存入buffer打印进log。

# 测试

```
mpirun -np 4 --host 100.64.24.87:2,100.64.24.94:2 --allow-run-as-root -mca btl ^openib -x NCCL_IB_GID_INDEX=3  -x NCCL_IB_TC=160  -x NCCL_NET_GDR_LEVEL=4 -x LD_LIBRARY_PATH=/workspace/infrawaves/zhangmj/vccl_2.21.51x/build/lib/ -x NCCL_TELEMETRY_ENABLE=1 -x NCCL_TELEMETRY_LOG_PATH=/workspace/infrawaves/Train/logs/j_NT_${TIMESTAMP}/ ./build/sendrecv_perf -b 8G -e 8G  -f  2 -g 1 -w 0

```