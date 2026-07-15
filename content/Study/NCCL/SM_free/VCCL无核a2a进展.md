无核a2a开发推荐在分支nokernel_alltoall上进行

## 目前问题

在无核a2a时，发现机间带宽被浪费，疑似在长时间执行机内P2p时，机间带宽未被利用，下面展示了利用telemetry看到的无核和有核时的机内带宽情况。

无核

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MDhjMTBmNGU1MjEwYTk5YjIxODIxN2M5MzZkMTkxOGZfQk92R0wwWmRnbUxZNWZXOWRQOXlxNWNvcTE3UkQ2R1lfVG9rZW46SGNHQmJJcFNGb2tUcVB4VzE1YmM4TlRobjNjXzE3NDk4MjAyNTI6MTc0OTgyMzg1Ml9WNA)

有核

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MzE3OWM1NjQxNzM3M2FkYmMwNjdkYTkwZDc5ZTc3NzlfOHF1dUhUMHp6U0h1VUl5VlFTVGpUa3VjTHZ1SEtUU29fVG9rZW46TWZvOGJ6SFBCb1laem94WVhUeGMwMXJzbkNlXzE3NDk4MjAyNTI6MTc0OTgyMzg1Ml9WNA)

## 已做尝试

### 增加eventRecord与多cpStream

- 在原先设计中，虽然copy的动作异步执行，但是在check一次拷贝动作是否完成时，采用cudaStreamQuery(...)，而该函数返回success代表着stream上没有任何未完成操作，无法精确得知单个拷贝动作何时完成， 导致传输时延增大，因此，在每个cudaMemcpy后增加一个event，通过cudaEventQuery来精确得知单个拷贝动作的完成
- 将机内的P2p拷贝放到多个不同的cpStream上，期望减少单个Stream上的排队（收益不明显，所以机间没有实现这个功能）

commit：nokernel_alltoall最新一次commit

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OGEwZmIxZDhkOWFjMDFlNzcyYjhmZWU5ZGM3OTIzNjhfR2FCbjNld2hMc3JsaUl1U0JDa0tKbkF3TUNBNnNtNXdfVG9rZW46Rkg1dGJEZ3pIbzJDNEd4VjJvSGNKa0ZTbnZmXzE3NDk4MjAyNTI6MTc0OTgyMzg1Ml9WNA)

### buffSize与ChunkSize调参

在nccl原生设计中，每一次chunkSize严格固定为128k或512k(nvls)，而在无核设计中，chunkSize = buffSize / 16(stepCnt)。由于request的数量同样严格受限，因而我们最多同时分配16个request，尝试调整了chunkSize与buffSize的值，均没有得到太好的结果。

### 增大psmProxyGetPostedOps频率

由于在无核中，单个op一旦到达，立刻上送执行，我们期望能尽可能的多拿到不同的op加入args链表，而非因为一直执行机内args而未能利用带宽，在psmP2pProxySend/Recv与psmNetProxySend/Recv中，当成功完成某次操作时候，例如，成功发送一次，成功test一次时，立刻return ncclSuccess，以加快一次progressOps的执行时间。

经过测试，通过此方法的无核，相比原先无核，在两机alltoall下，**性能能从10GB/s提升至60GB/s（基本对齐有核）**，然而，此方法，**测试send/recv时，性能直接飙升至130GB/s，不符合常理，需要排查原因**。

实验结果

在临港23与201上进行测试，容器为zx116

配套代码路径：`/workspace/infrawaves/share/zhangmj/sm_free/vccl_2.21.51x_alltoall`

执行nccl-tests：`root@node023:/workspace/infrawaves/share/zhangmj/sm_free/vccl_2.21.51x_alltoall# bash run_yunmai.sh`

Sendrecv

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmVhNGNlOTFlODdiMDBmOWNiODQ0YzJkNDRlNjljYTlfTjFaNjI3d0VZQndva1FmTFhaZjg1R3h4cExQanlkRTBfVG9rZW46Q1JwZ2JQd002b1hzeEF4eUxsTmNTbTJkbnRkXzE3NDk4MjAyNTI6MTc0OTgyMzg1Ml9WNA)

Alltoall

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OThkZjM3OGNiODJmMjQ1MTYyYTRhZTYwZWEzYWFjZGNfVDZ0eDVrdU9GZWJnUEcyQXFYeVRnZ3dFWXQxeE5Sd2VfVG9rZW46Rml1YmJmaFU2b2tmQll4alR1dWN6cGlDblVlXzE3NDk4MjAyNTI6MTc0OTgyMzg1Ml9WNA)

### 聚合op一次性提交

在原生实现中，一个group内的op会暂存，在整个group结束时，一次性提交，降低getPostedOps调用次数，尝试在无核中加入该点设计。

在目前无核实现中，主进程saveOp时，将op存至生产者队列proProgChannelHead中，在proxy线程中，调用getPostedOps将op从生产者队列转移至消费者对列comProgChannelHead中，而progressOPs则直接从消费者对列中取op进行执行，为了暂存op，**尝试增加等待者队列waitingProgChannelHead**，在saveOp时，将op暂存至等待者对列中，直到在groupEnd过程中，在调用doLaunches(...)时，一次性将等待者队列中暂存op提交至生产者队列中

下为src/group.cc:doLaunches修改的部分

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=NjZmYTFiNThiZTY1OWY1YjEyZTkzZDI2MmY3ZmVkN2RfdjdqaldWSGk3S09MaDNFTWhaWXVKdHBmUlZ5VFpLWllfVG9rZW46VG11bmJDbUtUb1FGbnF4RkhPQ2Nvd3NPbkl1XzE3NDk4MjAyNTI6MTc0OTgyMzg1Ml9WNA)

# 实验结果

在临港23与201上进行测试，容器为zx116

配套代码路径：`/workspace/infrawaves/share/zhangmj/sm_free/vccl_2.21.51x`

执行nccl-tests：`root@node023:/workspace/infrawaves/share/zhangmj/sm_free/vccl_2.21.51x# bash run_yunmai.sh`

目前存在hang，需要定位排查下

1. 一些问题
2. 关于a2a
3. 机内和机件的cpStream是否是同一个？
4. 为什么在2.3.中send/recv性能会到130？
5. alltoall对齐的数据真的可信吗？
6. 2.4 hang的原因需要排查，是op没能成功上送到生产者队列吗？
7. 关于pxn
8. 启用pxn时，假设localRank0 -> 对端localRank1，实际发送为localRank1 -> localRank1，localRank0是如何发送数据到localRank1呢？
9. fProxyState是否照抄Proxystate就行呢？