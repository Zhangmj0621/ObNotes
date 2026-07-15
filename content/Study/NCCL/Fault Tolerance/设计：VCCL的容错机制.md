### 需求背景

大模型训练中，因为软硬件故障的重启带来的时间开销导致高昂的成本。在VCCL的代码中，GPU之间的集合通信通过初始化过程中建立的RDMA QP连接完成。VCCL初始化建立的QP连接是静态的，不可变的，因此，如果某个QP连接由于网络物理链路的故障而断掉，集合通信失败，大模型训练应用挂住，经过设定的timeout时间后，由watchdog强行退出。

我们期望引入一种集合通信的容错机制，在网络物理链路出现故障时，集合通信的流量暂时使用备份的QP连接，保证训练任务不中断；当链路修复后，流量立即切回原来的QP连接，保证训练的性能不损失。

VCCL的容错机制不针对端口瞬时FLAP的场景，这种情况由IB协议本身的容错机制处理，如下面的截屏的引用所示，我们通过设置NCCL的环境变量NCCL_IB_RETRY_CNT和NCCL_IB_TIMEOUT调整这两个参数（timeout & retry_cnt），避免端口FLAP造成的集合通信失败。

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=M2VjZTAyY2UwZWVkYWYxOWZlOGM1MzU0ODRkMjYxZDJfY1prT1FvZXUwS1I4WGRnMTNHbWtWVVpDNmxYTVFBVDhfVG9rZW46RTA4QmI0a1Yxb0VNb3p4cWQ3N2Nsdm9obnhjXzE3NDk4MTk0ODY6MTc0OTgyMzA4Nl9WNA)

总之，端口FLAP的容错由RDMA IB协议自身的机制完成；假如端口出现了故障，在某个chunk的数据传递失败的情况下，VCCL的容错机制使用备份QP，失败的数据，从而，集合通信得以继续完成，模型训练不中断。

### 物理条件

在GPU服务器的标准配备中，多张RNIC用于GPU之间跨服务器节点的通信。一般地，一个GPU使用最“近”的RNIC，即通信效率最高的RNIC。理论上，如果一张RNIC出现错误，端口Down掉，相关联的GPU可以使用服务器上其他的RNIC，使得通信不中断。因此，从服务器端侧看，一个GPU除了最“近”的RNIC之外，存在多个可用的较“远”的备份RNIC。然而，将正在通信的流量切换到另一个RNIC，显然并非一件直截了当的事。

### 备份QP的创建

如下图所示，正常的QP建立在蓝色的线和圆点表示的物理链接和RNIC端口上，备份的QP建立在浅蓝色的线和圆点表示的物理链接和RNIC端口上。NCCL初始化时，对于一对正常的QP连接，有两种方案建立备份的QP连接：

- A方案：备份QP建立在本端的备份RNIC端口和对端正常的RNIC端口上。然而。在多轨的网络拓扑中，不同轨道的RNIC不一定可达；
- B方案：备份QP建立在本端的备份RNIC端口和对端备份的RNIC端口上，保证备份QP通信在同样的轨道上。

暂时无法在飞书文档外展示此内容

无论哪种方案，对于备份QP，至少一端的GPU可能无法使用最“近”的RNIC，并且，切换到备份RNIC端口的流量和该RNIC端口正常的流量竞争带宽，造成通信性能的大幅下降。我们采取B方案建立备份QP连接，即两端都使用备份的RNIC。选取备份RNIC的规则简单明了：

- 对于一卡双口的RNIC（dual-port）的情况，RNIC上两个端口互为备份；
- 对于一卡单口的情况，编号相邻的两个GPU相互以对方最“近”RNIC端口作为备份RNIC端口，如GPU0和GPU1结为一对，GPU0的备份RNIC端口选取GPU1最“近”的RNIC端口为备份的RNIC端口，反之亦然。

在VCCL初始化的过程中，按照建立正常QP连接的步骤，完成备份QP连接的建立。主要的代码改动包括如下几个部分，下文的描述是为了具体指出修改的位置，不能当作实际的实现。

1. 首先，[在net.cc](http://xn--net-lp6e.cc):ncclProxyConnection结构中，在sendNetResources和recvNetResources结构中，增加新的成员变量记录备份RNIC端口的ID。同样的，需要增加backup端口对应的属性值。（这一步可能用不到）

```
struct sendNetResources {
  struct connectMap map;
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int netDev;
+ int backupNetDev;
  int useGdr;
  int useDmaBuf;
+ int backupUseDmaBuf;
  int maxRecvs;
+ int backupMaxRecvs;
// ... ... ... ...
  int netDeviceVersion;
+ int backupNetDeviceVersion;
  ncclNetDeviceType netDeviceType;
+ ncclNetDeviceType backupNetDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
};

struct recvNetResources {
  struct connectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
+ int backupNetDev;
  int useGdr;
  int useDmaBuf;
+ int backupUseDmaBuf;
  int needFlush;
  int maxRecvs;
+ int backupMaxRecvs;
// ... ... ... ...
  int netDeviceVersion;
+ int backupNetDeviceVersion;
  ncclNetDeviceType netDeviceType;
+ ncclNetDeviceType backupNetDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
};

```

1. 在net_ib.cc中，仿照ncclIbMergedDevs，增加ncclIbBackupDevs，跟踪每个RNIC端口的备份RNIC端口。

```
#define MAX_IB_DEVS 32
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_DEVS];

+1 int ncclIbBackupDevs[MAX_IB_DEVS];

struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclIbRelaxedOrderingEnabled = 0;

```

1. 在net.c:setupReq中增加备份的RNIC的ID。

```
struct setupReq {
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int shared;
+ int netDev;
+ int backupNetDev;
  int useGdr;
  int needFlush;
  int channelId;
  int connIndex;
};

```

1. [在net.cc](http://xn--net-lp6e.cc):sendSetup(...)[和net.cc](http://xn--net-8i4e.cc):recvSetup(...)中，[调用search.cc](http://xn--search-2w4ox67l.cc):ncclTopoGetNetDev(...)时，需要获取两个RNIC端口的ID，一个用于正常的QP连接，一个用于备份的QP连接，为了叙述的方便，分别称为主RNIC端口/备RNIC端口，主QP/备QP。注意，无论在发送端和接收端，备QP务必和主QP关联同一个Proxy中，这样，流量在主备切换时，更方便发送端和接收端同步。并将备RNIC端口记录在ncclIbBackupDevs数组中。

```
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };

 //... ... ... ...

  int64_t netId;
+ int backupNetDev;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &req.backupNetDev, &proxyRank));
+ ncclIbBackupDevs[req.netDev] = req.backupNetDev;
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, netId, 1, &req.useGdr));
//... ... ... ...
  return ncclSuccess;
}

static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = { 0 };
  // ... ... ... ...
  int64_t netId;
+ int backupNetDev;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, &req.backupNetDev, &proxyRank));
+ ncclIbBackupDevs[req.netDev] = req.backupNetDev;
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, netId, 0, &req.useGdr));

// ... ... ... ...

  return ncclSuccess;
}

```

1. [在net.cc](http://xn--net-lp6e.cc):sendProxySetup(...)[和net.cc](http://xn--net-8i4e.cc):recvProxySetup中设置备份RNIC的ID。并且根据获取到的backupNetDev，获取对应Properties并更新Resources

```
static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
+ resources->backupNetDev = req->backupNetnetDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->channelId = req->channelId;
  // ... ... ... ...
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));

  // backup proportities
+ ncclNetProperties_t backupProps;
+ NCCLCHECK(proxyState->ncclNet->getProperties(req->backupNetDev, &backupProps));
  // ... ... ... ...
  /* backup DMA-BUF support */
+ resources->backupUseDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (backupProps.ptrSupport & NCCL_PTR_DMABUF);
+ resources->backupMaxRecvs= backupProps.maxRecvs;
+ resources->backupNetDeviceVersion = backupProps.netDeviceVersion;
+ resources->backupNetDeviceType = backupProps.netDeviceType;
  // ... ... ... ...
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct recvNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
+ resources->backupNetDev = req->backupNetnetDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->needFlush = req->needFlush;
  //... ... ... ...
  // backup proportities
+ ncclNetProperties_t backupProps;
+ NCCLCHECK(proxyState->ncclNet->getProperties(req->backupNetDev, &backupProps));
  // ... ... ... ...
  /* backup DMA-BUF support */
+ resources->backupUseDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (backupProps.ptrSupport & NCCL_PTR_DMABUF);
+ resources->backupMaxRecvs= backupProps.maxRecvs;
+ resources->backupNetDeviceVersion = backupProps.netDeviceVersion;
+ resources->backupNetDeviceType = backupProps.netDeviceType;
  // ... ... ... ...
  return ncclSuccess;
}

```

1. ~~在实现QP连接的net_ib.cc中，标识QP是否可用，是否为备QP。~~ 最后判断是否设备不可用基于warn信号，故这块未用到

```
struct ncclIbQp {
  struct ibv_qp* qp;
  int devIndex;
  int remDevIdx;
  uint8_t srcIp[4];
  uint8_t dscIp[4];
  int channel_id;
  int rank;
  std::string NetworkCardName="";
+ bool backup;
+ bool available;
};

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;
  int devIndex;
+ bool backup;
+ bool available;
};

```

1. 在ncclIbSendComm以及ncclIbRecvComm中增加backupDevs的对应ncclIbSendCommDev变量，为后续实际创建QP时服务，并在ncclIbNetCommBase中增加backupQps

```
struct ncclIbSendComm {
  // ... ... ...
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
+ struct ncclIbSendCommDev backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // ... ... ...
  int ar; // Use adaptive routing when all merged devices have it enabled
+ int backupAr;
  // ... ... ...
};

struct ncclIbRecvComm {
  // ... ... ...
  struct ncclIbRecvCommDev    devs[NCCL_IB_MAX_DEVS_PER_NIC];
+ struct ncclIbRecvCommDev backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // ... ... ...
};

```

```
struct alignas(32) ncclIbNetCommBase {
  int ndevs;
  bool isSend;
  struct ncclIbRequest reqs[MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
+ struct ncclIbQp backupQps[NCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  int devIndex;
+ int backupDevIndex;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
+ struct ncclIbDevInfo backupRemDevs[NCCL_IB_MAX_DEVS_PER_NIC];
};

```

1. 修改ncclIbConnectionMetadata，在其中增加备份QP和端口相关的信息

```
// Struct containing everything needed to establish connections
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
+ struct ncclIbQpInfo backupQpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
+ struct ncclIbDevInfo backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
+ char backupDevName[MAX_MERGED_DEV_NAME];
  uint64_t fifoAddr;
  int ndevs;
};

```

1. 在net_ib.cc:ncclIbConnect(...)和net_ib.cc:ncclIbAccept()中，对于每个主QP，分别在备份RNIC端口上建立对应的备QP。需要注意，Dev与backupDev的devIndex在一卡两口情况下**并不相同**。（Dev.devIndex = 0则backupDev.devIndex = 1)。下面展示ncclIbConnect修改代码，ncclIbAccept类似。

```
ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  // ... ... ...
ib_connect_check:
  // IB Setup
  struct ncclIbMergedDev* mergedDev;
  mergedDev = ncclIbMergedDevs + dev;
  // backup dev
+ int backupDev = ncclIbBackupDevs[dev];
+ struct ncclIbMergedDev *backupMergedDev;
  //... ... ...
  // init PD, Ctx for each IB device
  for (int i = 0; i < mergedDev->ndevs; i++) {
    // ... ... ...
    // backup dev
  + int backupIbDevN = backupMergedDev->devs[i];
  + NCCLCHECK(ncclIbInitCommDevBase(backupIbDevN, &comm->backupDevs[i].base));
  + comm->backupAr = comm->backupAr && ncclIbDevs[backupDev].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }
  // ... ... ...
  // Alternate QPs between devices
  for (int q = 0; q < comm->base.nqps; q++) {
    // ... ... ...
    //back up QP create
  + ncclIbSendCommDev* backupCommDev = comm->backupDevs + backupDevIndex;
  + ncclIbDev* backupIbDev = ncclIbDevs + backupCommDev->base.ibDevN;
  + NCCLCHECK(ncclIbCreateQp(backupIbDev->portNum, &backupCommDev->base, IBV_ACCESS_REMOTE_WRITE, comm->base.backupQps + q));
  + comm->base.backupQps[q].devIndex = backupDevIndex;
  + comm->base.backupQps[q].backup = true;
  + comm->base.backupQps[q].available = true;
  + meta.backupQpInfo[q].qpn = comm->base.backupQps[q].qp->qp_num;
  + meta.backupQpInfo[q].devIndex = comm->base.backupQps[q].devIndex;
  + meta.backupQpInfo[q].backup = comm->base.backupQps[q].backup;
  + meta.backupQpInfo[q].available = comm->base.backupQps[q].available;
    //... ... ...
  + devIndex = (devIndex + 1) % comm->base.ndevs;
  + backupDevIndex = (backupDevIndex + 1) % comm->base.ndevs;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    // ... ... ...
    // write backup info to the meta struct via this pointer
  + // ... ... ...
    // ... ... ...
    // back up RoCE support
  + // ... ... ...
  }
  // ... ... ...

ib_send:
  // ... ... ...

ib_connect:
  // ... ... ...
  // check remote backup link_layer
+ // ... ... ...
  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    // ... ... ...
  + comm->base.backupRemDevs[i] = remMeta.backupDevs[i];
  + comm->base.backupRemDevs[i].remoteGid.global.interface_id = comm->base.backupRemDevs[i].iid;
  + comm->base.backupRemDevs[i].remoteGid.global.subnet_prefix = comm->base.backupRemDevs[i].spn;
    // ... ... ...
  + comm->remSizesFifo.backupRkeys[i] = remMeta.backupDevs[i].fifoRkey;
  }
  // ... ... ...
  for (int q = 0; q < comm->base.nqps; q++) {
    // ... ... ...
  + struct ncclIbQpInfo* backupRemQpInfo = remMeta.backupQpInfo + q;
  + struct ncclIbDevInfo* backupRemDevInfo = remMeta.backupDevs + backupRemQpInfo->devIndex;
    // ... ... ...
    // Assign per-QP backupRemDev
  + comm->base.backupQps[q].remDevIdx = backupRemQpInfo->devIndex;
  + devIndex = comm->base.backupQps[q].devIndex;
  + ncclIbSendCommDev* backupCommDev = comm->backupDevs + devIndex;
  + gidIndex = backupCommDev->base.gidInfo.localGidIndex;

  + qp = comm->base.backupQps[q].qp;
  + if (backupRemQpInfo->ece_supported && backupRemQpInfo->ece_supported)
    + NCCLCHECK(wrap_ibv_set_ece(qp, &backupRemDevInfo->ece, &backupRemDevInfo->ece_supported));

  + NCCLCHECK(ncclIbRtrQp(qp, gidIndex, backupRemQpInfo->qpn, backupRemDevInfo));
  + NCCLCHECK(ncclIbRtsQp(qp));
  }
  // ... ... ...
+ if (link_layer == IBV_LINK_LAYER_ETHERNET ) { // RoCE
  + for (int q = 0; q < comm->base.nqps; q++) {
  + struct ncclIbQp* qp = comm->base.backupQps + q;
  + int ibDevN = comm->backupDevs[qp->devIndex].base.ibDevN;
  + struct ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  + INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
    + ibDevN, ibDev->portNum, remMeta.backupQpInfo[q].qpn, remMeta.backupQpInfo[q].ece_supported, remMeta.backupQpInfo[q].ece.vendor_id, remMeta.backupQpInfo[q].ece.options, remMeta.backupQpInfo[q].ece.comp_mask);
  + }
+ }
  // ... ... ...

  ib_send_ready:
    // ... ... ...
}

```

同步的，需要增加backup Cq的概念，我们在每个端口上新增backup cq，并修改ncclIbInitCommDevBase函数，保证了在init backup端口时能创建backup的cq

```
ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, bool if_backup) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  pthread_mutex_lock(&ibDev->lock);
  if (0 == ibDev->pdRefs++) {
    ncclResult_t res;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context), res, failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ibDev->lock);
      return res;
    }
  }
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  if(!if_backup) NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), NULL, NULL, 0));
+ else NCCLCHECK(wrap_ibv_create_cq(&base->backupCq, ibDev->context, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), NULL, NULL, 0));

  return ncclSuccess;
}

ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base, bool if_backup) {
  ncclResult_t res;
  if(!if_backup) {
    NCCLCHECK(wrap_ibv_destroy_cq(base->cq));
  }
+ else {
  + NCCLCHECK(wrap_ibv_destroy_cq(base->backupCq));
+ }

  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd), res, returning);
  }
  res = ncclSuccess;
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

```

### 发送和接收失败后两端的同步

[发送端在函数net.cc](http://xn--net-ee2ep7fuuhi93a0j3b9v1b.cc):sendProxyProgress(...)中，有三个发送步骤节点指示发送的状态：

- Posted - 准备要发送步骤位置；
- Submitted - 已发送的步骤位置；
- Done - 发送成功的步骤位置。

如果发送失败，即net_ib.cc:ncclIbTest(...)返回失败的结果，则sub->transmitted回退到done的位置，以保证发送失败的数据利用备份的QP重传。

```
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
     // ... ... ...
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done;
        int size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        int result = proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size);
        if(!result){
          ncclIbTestOutput((struct ncclIbRequest*)sub->requests[buffSlot]);
        }
        NCCLCHECK((ncclResult_t)result);
        if (done) {
         // ... ... ... ...
          if (sub->reg == 0) connFifo[buffSlot].size = -1;
          __sync_synchronize();
          TRACE(NCCL_NET, "sendProxy [%ld/%d] request %p done", sub->done, buffSlot, sub->requests[buffSlot]);
          sub->done += args->sliceSteps;
          // ... ... ... ...
+       } else {
+           sub->transmitted = sub->done
        }
      }
    }
    // ... ... ... ...
}

```

同样地，[在函数net.cc](http://xn--net-ee2e89u12x.cc):recvProxyProgress(...)中，以下的步骤节点指示接收的状态：

- Posted - 准备要接收的步骤位置；
- Received - 已接收的步骤位置；
- Transmitted - flush成功的步骤位置；
- Done - 接收成功步骤位置。

如果接收失败，posted退回args->slicesteps步。

```
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // ... ... ... ...

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->received > subGroup->transmitted) {
        uint64_t step = subGroup->transmitted;
        int done = 1;
        void* request = subGroup->requests[step%NCCL_STEPS];
        if (request){
          int result = proxyState->ncclNet->test(request, &done, NULL);
          if(!result){
            ncclIbTestOutput((struct ncclIbRequest*)request);
          }
          NCCLCHECK((ncclResult_t)result);
        }
        if (done) {
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;

            sub->transmitted += args->sliceSteps;
            for (uint64_t step=sub->transmitted-args->sliceSteps; step<sub->transmitted; step++) ncclProfilingRecord(args, s+i, step, ncclProxyProfileRecvGPUWait);
            if (step < sub->nsteps) {
              __sync_synchronize();
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
              if (sub->reg) {
                // We may have added more net steps, but reg operations only have a single step w.r.t. the GPU.
                if (sub->transmitted == sub->nsteps) *recvTail = sub->base + args->sliceSteps;
              } else
                *recvTail = sub->base + sub->transmitted;
              if (resources->gdcSync) wc_store_fence(); // Flush out WC write
            }
          }
          args->idle = 0;
        }
+        else {
+           struct ncclProxySubArgs* subGroup = args->subs+s;
+           for (int i=0; i<subGroup->groupSize; i++) {
+               struct ncclProxySubArgs* sub = subGroup+i;
+               sub->posted -= args->sliceSteps;
+           }
        }
 // ... ... ... ...
    }

```

### 发送接收失败后QP状态设置和检查

在net_ib.cc:ncclIbTest(...)中，如果发送或接收失败，可以通过ncclIbRequest知道故障RNIC的ID。

```
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;
  while (1) {
// ... ... ... ...
    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
// ... ... ... ...
            std::string Line= ncclSocketToString(&addr, line);
            r->devBases[i]->warn.is_warn=true;
            r->devBases[i]->warn.line=Line;
            r->devBases[i]->warn.status=wc->status;
            r->devBases[i]->warn.opcode=wc->opcode;
            r->devBases[i]->warn.len=wc->byte_len;
            r->devBases[i]->warn.error=wc->vendor_err;
// ... ... ... ...
  return ret;
}

```

在收发数据的函数中，包括：

- net_ib.cc:ncclSend(...)
- net_ib.cc::ncclIbMultiSend(...)
- net_ib.cc:ncclIbIrecv(...)
- net_ib.cc:ncclIbPostFifo(...)
- net_ib.cc:ncclIbIflush(...)

为了实现同步机制中的rkeys和lkeys同步切换，修改ncclIbMrHandle结构体，扩容mrs，存储正常时以及backup时的mr。

```
struct ncclIbMrHandle {
  // mrs[0:1] qp rkeys
  // mrs[2:3] backup qp rkeys
- ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
+ ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC * 2];
};

```

在regmr时，将backup相关的mr一起存入mhandle中

```
struct ncclIbNetCommDevBase* ncclIbGetBackupNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->backupDevs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->backupDevs[devIndex].base;
  }
}

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  assert(size > 0);
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) malloc(sizeof(struct ncclIbMrHandle));
  for (int i = 0; i < base->ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbRegMrDmaBufInternal(devComm, data, size, type, offset, fd, mhandleWrapper->mrs + i));

    // fill backup mhandleWrapper->mrs
  + struct ncclIbNetCommDevBase* backupDevComm = ncclIbGetBackupNetCommDevBase(base, i);
  + NCCLCHECK(ncclIbRegMrDmaBufInternal(backupDevComm, data, size, type, offset, fd, mhandleWrapper->mrs + i + 2));
  }
  *mhandle = (void*) mhandleWrapper;
  return ncclSuccess;
}

```

检查QP的状态，如果不可用，则使用备份QP。

```
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;

// ... ... ... ...

  // Select either all QPs, or one qp per-device
  const int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;

  // Post recvs
  struct ibv_recv_wr* bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
  + bool if_backup = false;
    // check if qp is available
  + if (comm->devs[qp->devIndex].base.warn.is_warn == true) {
    + if_backup = true;
    + qp = comm->base.backupQps + comm->base.qpIndex;
    + ncclIbAddEvent(req, qp->devIndex, &comm->backupDevs[qp->devIndex].base);
  + }
  + else {
    + ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
    }

    // ... ... ... ...

    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex+1)%comm->base.nqps;
  }

 // ... ... ... ...
  return ncclSuccess;
}

```

在net_ib.cc::ncclResult_t ncclIbPostFifo()中，增添使用备份qp发送逻辑，其中包括rkeys的选取

```
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, int* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  // ... ... ...
  ncclIbQp* ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.ndevs;

+ ncclIbQp *backupCtsQp = comm->base.backupQps + comm->base.backupDevIndex;
+ comm->base.backupDevIndex = (comm->base.backupDevIndex + 1) % comm->base.ndevs;

+ bool if_backup = false;
+ if (comm->devs[ctsQp->devIndex].base.warn.is_warn == true) {
  + if_backup = true;
  }
  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];

    // Send all applicable rkeys
    for (int j = 0; j < comm->base.ndevs; j++) {
      if(!if_backup) localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
    + else localElem[i].rkeys[j] = mhandleWrapper->mrs[j + 2]->rkey;
    }
    // ... ... ...
  }
  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct fifoRkey
  if(!if_backup) {
    wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;
  }
  else {
  + wr.wr.rdma.rkey = comm->base.backupRemDevs[backupCtsQp->remDevIdx].fifoRkey;
  }

  // Set the correct sge properties
  if (!if_backup) {
    comm->devs[ctsQp->devIndex].fifoSge.addr = (uint64_t)localElem;
    comm->devs[ctsQp->devIndex].fifoSge.length = n * sizeof(struct ncclIbSendFifo);
    wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;
  }
  else {
  + comm->backupDevs[backupCtsQp->devIndex].fifoSge.addr = (uint64_t)localElem;
  + comm->backupDevs[backupCtsQp->devIndex].fifoSge.length = n * sizeof(struct ncclIbSendFifo);
  + wr.sg_list = &comm->backupDevs[backupCtsQp->devIndex].fifoSge;
  }

  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags; // IBV_SEND_INLINE
  if (!if_backup && slot == ctsQp->devIndex) {
    // ... ... ...
  }
+ else if (if_backup && slot == backupCtsQp->devIndex){
  + wr.send_flags |= IBV_SEND_SIGNALED;
  + wr.wr_id = req - comm->base.reqs;
  + ncclIbAddEvent(req, backupCtsQp->devIndex, &comm->backupDevs[backupCtsQp->devIndex].base);

  + *(u_int *)req->log[backupCtsQp->devIndex].srcIp = *(u_int *)backupCtsQp->srcIp;
  + *(u_int *)req->log[backupCtsQp->devIndex].dscIp = *(u_int *)backupCtsQp->dscIp;
  + req->lTest[backupCtsQp->devIndex].linkPingQp = backupCtsQp->qp;

  + if(global_timer_log.collect){
    + req->log[backupCtsQp->devIndex].loged_start = NCCL_LOG_TELEMETRY;
    + req->lTest[backupCtsQp->devIndex].status = LINK_STATUS_UNUSED;
    + req->log[backupCtsQp->devIndex].size = n*sizeof(struct ncclIbSendFifo);
    + clock_gettime(CLOCK_REALTIME, &req->log[backupCtsQp->devIndex].send_start);
  + }
  + else
    + req->log[backupCtsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
  }
  else if (!if_backup) req->log[ctsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
  else {
  + req->log[backupCtsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
  }

  struct ibv_send_wr* bad_wr;
  if(!if_backup) NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  else NCCLCHECK(wrap_ibv_post_send(backupCtsQp->qp, &wr, &bad_wr));
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

```

在net_ib.cc::ncclResult_t ncclIbIsend()中，做同样检查，其中涉及到req->send.lkeys的获取

```
ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) { WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0"); return ncclInternalError; }
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  // ... ... ...
  for (int r=0; r<nreqs; r++) {
    // ... ... ...
    while (nEvents > 0) {
      ncclIbQp* qp = comm->base.qps + qpIndex;

    + bool if_backup = false;
    + if (comm->devs[qp->devIndex].base.warn.is_warn == true) {
      + qp = comm->base.backupQps + qpIndex;
      + if_backup = true;
    + }
    + int devIndex = qp->devIndex;

      // add event
      if(!if_backup) ncclIbAddEvent(req, devIndex, &comm->devs[devIndex].base);
    + else ncclIbAddEvent(req, devIndex, &comm->backupDevs[devIndex].base);
      // ... ... ...
      // get the right lkey from mrs
      if(!if_backup) req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
    + else req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex + 2]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same set of QPs inside ncclIbMultiSend()
      qpIndex = (qpIndex+1)%comm->base.nqps;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.ndevs; i++) {
      if (!if_backup) {
        req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
      }
      else {
      + req->send.lkeys[i] = mhandleWrapper->mrs[i + 2]->lkey;
      }
    }
    // ... ... ...
  }
}

```

函数net_ib.cc::ncclResult_t ncclIbMultiSend()的实现中，做同样的检查。

```
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  // ... ... ... ...
  for (int i = 0; i < nqps; i++) {
    sendWrCounter ++;
    int qpIndex = comm->base.qpIndex;
    ncclIbQp* qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;

   // check if qp is available
  + bool if_backup = false;
  + if (comm->devs[devIndex].base.warn.is_warn == true) {
    + qp = comm->base.backupQps + qpIndex;
    + if_backup = true;
  + }
  + int devIndex = qp->devIndex;

// ... ... ... ...

  return ncclSuccess;
}

```

在ncclIbIflush中，同样增加backup相关逻辑

```
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;
  // ... ... ...
  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

  + bool if_backup = false;
    if (comm->devs[i].base.warn.is_warn == true) {
    + if_backup = true;
    + wr.wr.rdma.rkey = mhandle->mrs[i + 2]->rkey;
    + wr.sg_list = &comm->backupDevs[i].gpuFlush.sge;
    }
    else {
      // ... ... ...
    }
    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    if (!if_backup) {
      // ... ... ...
    }
    else {
    + *(u_int *)req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].srcIp = *(u_int *)comm->backupDevs[i].gpuFlush.qp.srcIp;
    + *(u_int *)req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].dscIp = *(u_int *)comm->backupDevs[i].gpuFlush.qp.dscIp;
    + if (global_timer_log.collect)
      {
      + req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].loged_start = NCCL_LOG_TELEMETRY;
      + req->lTest[comm->backupDevs[i].gpuFlush.qp.devIndex].status = LINK_STATUS_UNUSED;

      + clock_gettime(CLOCK_REALTIME, &req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].send_start);
      }
      else
      + req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].loged_start = NCCL_LOG_NOT_USE;
    }
    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    if (!if_backup) NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
  + else NCCLCHECK(wrap_ibv_post_send(comm->backupDevs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    if (!if_backup) ncclIbAddEvent(req, i, &comm->devs[i].base);
  + else ncclIbAddEvent(req, i, &comm->backupDevs[i].base);
  }

  *request = req;
  return ncclSuccess;
}

```

### 结束收发时资源释放

在ncclIbCloseSend，增加释放backup qp逻辑

```
ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp != NULL)
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
    + if (comm->base.backupQps[q].qp != NULL)
      + NCCLCHECK(wrap_ibv_destroy_qp(comm->base.backupQps[q].qp))
    }

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct ncclIbSendCommDev* commDev = comm->devs + i;
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i]));
    + if (comm->remSizesFifo.mrs[i + NCCL_IB_MAX_DEVS_PER_NIC] != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i + NCCL_IB_MAX_DEVS_PER_NIC]));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));

    + struct ncclIbSendCommDev *backupCommDev = comm->backupDevs + i;
    + if (backupCommDev->fifoMr!= NULL) NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->fifoMr));
    + NCCLCHECK(ncclIbDestroyBase(&backupCommDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

```

在ncclIbCloseRecv进行同样操作，此处省略代码

1. 发送接收同步的检验
2. 同步fifo

在send端，我们维护了一个类似收发时的Fifo的syncFifo，来做当网卡出现故障时，切换所需同步信息的同步，其中包括，任务sub所需回退的位置，继续进行收发的fifo指针fifohead。

```
// fifo for synchronizing when changing to backup
struct alignas(32) ncclIbSyncFifo{
  uint64_t recvFifoTail;    // get send fifo head, and roll back fifoTail to fifoHead
  uint64_t restartPos;  // update recv sub->posted because recv sub->received may be greater than send sub->done
  uint64_t idx;
  int errPortIdx;
};

struct ncclIbRemSyncFifo
{
  struct ncclIbSyncFifo elems[MAX_REQUESTS];
  uint64_t syncFifoTail;
  uint64_t addr;
};

```

### 同步fifo的建联

在ncclIbConnect以及ncclIbAccept中，新增类似fifo的建联逻辑，同步syncFifo的rkey、addr等信息

ncclIbConnect

```cpp
    // Prepare my sync fifo
    NCCLCHECK(wrap_ibv_reg_mr(&commDev->syncFifoMr, commDev->base.pd, comm->syncFifo, sizeof(struct ncclIbSyncFifo)*MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
    devInfo->syncFifoRkey = commDev->syncFifoMr->rkey;

    // Prepare backup sync fifo
    NCCLCHECK(wrap_ibv_reg_mr(&backupCommDev->syncFifoMr, backupCommDev->base.pd, comm->syncFifo, sizeof(struct ncclIbSyncFifo)*MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
    backupDevInfo->syncFifoRkey = backupCommDev->syncFifoMr->rkey;

```

ncclIbAccept

```cpp
    // Retain remote sync fifo info and prepare my RDMA ops
    rCommDev->syncFifoRkey = remMeta.devs[i].syncFifoRkey;
    rComm->remSyncFifo.addr = remMeta.syncFifoAddr;
    NCCLCHECK(wrap_ibv_reg_mr(&rCommDev->syncFifoMr, rCommDev->base.pd, &rComm->remSyncFifo.elems, sizeof(struct ncclIbSyncFifo)*MAX_REQUESTS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
    rCommDev->syncFifoSge.lkey = rCommDev->syncFifoMr->lkey;

    // backup Retain remote sync fifo info and prepare my RDMA ops
    backupRCommDev->syncFifoRkey = remMeta.backupDevs[i].syncFifoRkey;
    NCCLCHECK(wrap_ibv_reg_mr(&backupRCommDev->syncFifoMr, backupRCommDev->base.pd, &rComm->remSyncFifo.elems, sizeof(struct ncclIbSyncFifo)*MAX_REQUESTS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
    backupRCommDev->syncFifoSge.lkey = backupRCommDev->syncFifoMr->lkey;

```

### 同步fifo过程

类似postFifo的postSyncFifo函数，用于同步addr等信息

```cpp
// recv post to the sync fifo
ncclResult_t ncclIbPostSyncFifo(void *recvComm, uint64_t restartPos, int errPortIdx) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remSyncFifo.syncFifoTail % MAX_REQUESTS;
  struct ncclIbSyncFifo* localElem = &comm->remSyncFifo.elems[slot];

  // when come into this function, you should use backup qp
  ncclIbQp *backupCtsQp = comm->base.backupQps + comm->base.backupDevIndex;
  comm->base.backupDevIndex = (comm->base.backupDevIndex + 1) % comm->base.ndevs;

  comm->remFifo.fifoTail += 1000;

  localElem->recvFifoTail = comm->remFifo.fifoTail;
  localElem->restartPos = restartPos;
  localElem->idx = comm->remSyncFifo.syncFifoTail + 1;
  localElem->errPortIdx = errPortIdx;

  // fill wr
  wr.wr.rdma.remote_addr = comm->remSyncFifo.addr + slot * sizeof(struct ncclIbSyncFifo);
  wr.wr.rdma.rkey = comm->base.backupRemDevs[backupCtsQp->remDevIdx].syncFifoRkey;
  comm->backupDevs[backupCtsQp->devIndex].syncFifoSge.addr = (uint64_t)localElem;
  comm->backupDevs[backupCtsQp->devIndex].syncFifoSge.length = sizeof(struct ncclIbSyncFifo);
  wr.sg_list = &comm->backupDevs[backupCtsQp->devIndex].syncFifoSge;
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;

  // write
  struct ibv_send_wr *bad_wr;
  NCCLCHECK(wrap_ibv_post_send(backupCtsQp->qp, &wr, &bad_wr));

  comm->remSyncFifo.syncFifoTail++;

  return ncclSuccess;
}

```

在net.cc中，当recv端发现错误wc时，需要回退时，将自己对应的回退信息发送至send端syncFifo中，保证了收发两端回退的一致性

```
          if(result != ncclSuccess) {
            int errPortIdx;
            NCCLCHECK(ncclIbGetErrorPortIdx(subGroup->requests[subGroup->received % NCCL_STEPS], errPortIdx));
            for (int s = 0; s < args->nsubs; s += args->subs[s].groupSize) {
              struct ncclProxySubArgs* t_subGroup = args->subs+s;
              for (int t_b = t_subGroup->received; t_b < t_subGroup->posted; t_b += args->sliceSteps) {
                int t_buffSlot = t_b % NCCL_STEPS;
                _ncclIbFreeRequest(t_subGroup->requests[t_buffSlot]);
              }

              for (int i = 0; i < t_subGroup->groupSize; i++) {
                struct ncclProxySubArgs* t_sub = t_subGroup + i;
                struct recvNetResources* resources = (struct recvNetResources*) (t_sub->connection->transportResources);

                INFO(NCCL_INIT, "recvProxy/iflush (channelId %d: myRank %d -> peerRank %d) failed, the step of posted (%ld) is rolled back to the step of received (%ld)",
                                 t_sub->channelId,
                                 args->peerRank,
                                 proxyState->tpRank,
                                 t_sub->posted,
                                 t_sub->received);

                t_sub->posted = t_sub->received;
              + NCCLCHECK(ncclIbPostSyncFifo(resources->netRecvComm, t_sub->received, errPortIdx));
              }
            }
            args->idle = 0;
            return ncclSuccess;
          }

```

### 问题

1. 当发送失败时，所有聚合在ncclIbMultiSend中的req全部失败吗？

### 实验验证

- 验证backup QP收发能力

设置所有的if_backup默认为true，默认使用backup qp进行发送，在8打8 8G all_gather场景下带宽如下

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MjJkYmFlOGFiN2MwMzAwOWFiZDYxYmEwNjcxZDQzMmNfWllrS3JlRzBZV3h0QjQ1ZEMxSjFNRkoxcFJwRjQxb0ZfVG9rZW46VmZ0VGI0TEo1b0k0RWd4dHhGU2NJTnY2bkhlXzE3NDk4MTk0ODY6MTc0OTgyMzA4Nl9WNA)

在1打1场景下，全部同步backup qp进行收发，并在运行nccl-test中down掉mlx5_0网卡，此时测试不受影响，证实了backup端口的qp收发能力

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGRjOTExOTgyZDM2ZTI3YzNhMzdhY2E1OWVlN2JkNjZfZ0FOcGVoSmlwak5NRDRZRkZZWW82bjlXZUhGYklTeERfVG9rZW46TEx1TGJMNUV3b2tOMWl4ZWVDU2MyWDFwblhnXzE3NDk4MTk0ODY6MTc0OTgyMzA4Nl9WNA)

如何手动down掉端口，以宁夏环境为例

[宁夏环境-手动down端口](https://infrawaves.feishu.cn/docx/ODYZd5UYOoK5tTx9krFcnDQSnLe)

- 验证接收同步检验

在主qp发送失败时，是否可以正确切换到backup QP进行发送

reduce_scatter_perf测试，两机8打8测试

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=ODc3ODE2ZjkzODMzNDAxNWJlZDcwMTNiMzMxNzEzYThfSkkzbUFNN1hGWWQwZHppNVBnYzNTbHNDVWVBUTVpY2NfVG9rZW46WnBzMGI1bmFKb0ZlYXl4eTVuQmNEWE1nbnliXzE3NDk4MTk0ODY6MTc0OTgyMzA4Nl9WNA)

在网卡down掉前，reduce_scatter正常执行，当网卡突然down掉，经过收发同步以后，切换到备份qp进行通信，此时虽发送带宽略有下降（符合预期），但集合通信不终止，测试结果通过telemetry工具可验证。