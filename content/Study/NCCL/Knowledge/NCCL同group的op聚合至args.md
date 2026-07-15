## 概述

在alltoall时候，以两机16GPU为例，对于单个rank来说，与其走net通信的rank有对端服务器上的8个peer，我们会调用8次对应的`addP2pToPlan(...)`，生成八个对应的op(struct ncclProxyOp*)，该op也即在`send/recvProxyProgress`中的args中的sub，在实际通信时，NCCL为了最大化利用带宽，实现了类似pipeline的机制，八个sub放在一个args中，当sub[0]数据未准备好或已经发送还未等到应答ACK时，可以直接发送sub[1]的数据，本文以alltoall为例，梳理多个op是如何合并至同一个args中，交给proxy线程进行收发，并尝试在无核中实现相同设计。

## NCCL connect时建立proxyOps

由于在开PXN场景下，会发生所谓的进程转移，rank0与对端的rank1通信会利用本地的rank1进行同轨通信，因而在后续progressOps的时候需要通知的是rank1处的proxy线程。其中，proxyOps与sock等信息存放在comm->proxyState中，proxyState为**每一个communicator组共享的shared proxyState**，因而proxyOps与pool等信息能被不同的进程共享读写，其中，通过tpLocalRank来标识到具体的proxyOps队列（可以理解为若本端有8个rank，则相应的有8个proxyOps队列，该队列能被所有rank读写），且能通知到唯一个Proxy线程。

## NCCL

### 进程

NCC中，op的添加包含了进程到proxy线程的控制权转移，先通过`ncclLaunchPrepare(...)`等函数，在进程中，填充op对应内容，包含dataSize, peer, **opCount**等信息(opCount很重要，下文会着重介绍），并通过调用`ncclProxyPost(...)`函数，更改pool的标志位，将控制权转给Proxy线程(proxy线程会轮询该状态位），我们主要关注op在post至proxy的过程中，是如何区分合并不同的op至同一个args中。

在addP2pPlan中，计算opCount的方式如下：

![image.png](attachment:d3918457-fc8e-4f36-aacd-73e56c9b466e:image.png)

填充完op对应信息后，调用`addProxyOpIfNeeded`将op post至proxy线程，其中，`addProxyOpIfNeeded`调用`saveProxy`实现，而`saveProxy`则调用`ncclLocalOpAppend`实现，我们主要关注`ncclLocalOpAppend`函数。

在`ncclLocalOpAppend(...)`函数中，主要完成，在pool中，找到空余可用的op（若无可用op会直接使用sched_yield(...)释放控制权），将op加入到proxyOps(proxyOps为每个comm中的proxyState中维护的链表）的链表尾部，该链表头为nextOps，尾为nextOpsEnd，并在，链表长度到达最大峰值**MAX_OPS_PER_PEER**后，尝试post op。

在post前，首先保留了链表尾部的op，**并且只post链表中，与最后一个op的opCount不同的所有op**，目的是为了保证，**相同opCount的op，能一次性的完成提交，保证了args中同opCount的sub的聚合**。

![image.png](attachment:10bdda02-b402-4b16-81d4-1b2bc99c99a7:image.png)

在`ncclProxyPost`中，则直接修改pool的nextOps与nextOpsEnd，告知proxy线程提交的op链表。

```c
ncclResult_t ncclProxyPost(struct ncclProxyOpsPool* pool, int nextOps, int nextOpsEnd) {
  pthread_mutex_lock(&pool->mutex);
  if (pool->nextOps == -1) {
    pool->nextOps = nextOps;
    pthread_cond_signal(&pool->cond);
  } else {
    pool->ops[pool->nextOpsEnd].next = nextOps;
  }
  pool->nextOpsEnd = nextOpsEnd;
  pthread_mutex_unlock(&pool->mutex);
  return ncclSuccess;
}

```

### Proxy线程

在ncclProxyGestPostedOps(...)中，若不存在args且pool链表头为空，则等待cond信号

```c
if (state->active == NULL) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->nextOps == -1 && !state->stop) {
      struct ncclProxyArgs profArgs; // Only used for profiling purposes
      ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileSleep);
      pthread_cond_wait(&pool->cond, &pool->mutex);
      ncclProfilingRecord(&profArgs, 0, 0, ncclProxyProfileWakeup);
    }
    if (state->stop) { // We might have been woken up to stop.
      pthread_mutex_unlock(&pool->mutex);
      return ncclSuccess;
    }
  }

```

将op转成sub，则主要在`ProxyAppend(...)`中，将收到的op加入args中，**若opCount与最后一个args相同，则直接将op添加至args链表尾**，不然，则新建args。

![image.png](attachment:c2bc0d3d-7374-4a63-afec-e8a81acc86e2:image.png)

### VCCL

目前VCCL无核并不存在args聚合概念，主要关注VCCL无核中，op如何转变为args，并在progressOps中是如何执行的。