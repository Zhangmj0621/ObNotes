NCCL是英伟达开源的GPU通信库，支持集合通信和点对点通信

官方提供小demo，其中`rank0`调用`ncclGetUniqueId(&id)`获取id，并将其广播给其他所有rank

```
  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  ......
  //generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

```

`ncclGetUniqueId`

```
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}

```

在ncclInit中，首先执行InitEnv设置环境变量，随后执行initNet用来初始化所需要的网络，一个是bootstrap网络，另外一个是数据通信网络，bootstrap网络主要用来初始化时交换一些简单的信息，比如每个机器的ip端口，由于数据量很小，而且主要是在初始化阶段执行一次，因而bootstrap使用的是tcp，而通信网络是用于实际数据的传输，因此会优先使用rdma（支持gdr的话会优先使用gdr）

`initNet`

```
ncclResult_t initNet() {
  // Always initialize bootstrap network
  NCCLCHECK(bootstrapNetInit());

  NCCLCHECK(initNetPlugin(&ncclNet, &ncclCollNet));
  if (ncclNet != NULL) return ncclSuccess;
  if (initNet(&ncclNetIb) == ncclSuccess) {
    ncclNet = &ncclNetIb;
  } else {
    NCCLCHECK(initNet(&ncclNetSocket));
    ncclNet = &ncclNetSocket;
  }
  return ncclSuccess;
}

```

bootstrapNetInit就是bootstrap网络的初始化，**主要就是通过`findInterfaces`遍历机器上所有的网卡信息，通过prefixList匹配选择使用哪些网卡，将可用网卡的信息保存下来**，将ifa_name保存到全局的bootstrapNetIfNames，ip地址保存到全局bootstrapNetIfAddrs，默认除了docker和lo其他的网卡都可以使用，例如在测试机器上有三张网卡，分别是xgbe0，xgbe1，xgbe2，那么就会把这三个ifaname和对应的ip地址保存下来，另外nccl提供了环境变量NCCL_SOCKET_IFNAME可以用来指定想用的网卡名，例如通过export NCCL_SOCKET_IFNAME=xgbe0来指定使用xgbe0，其实就是通过prefixList来匹配做到的。

随后是数据通信网络的初始化。

首先会执行`InitNetPlugin`[检查是否有libnccl-net.so](http://xn--libnccl-net-703su98pirchsedue.so)，若没有则返回（测试环境没有这个动态库）

然后尝试使用IB网络：

首先执行ncclNetIb的init函数，就是`ncclIbInit`，具体执行的步骤可概括为

- 通过`wrap_ibv_symbols`[加载动态库libibverbs.so](http://xn--libibverbs-t12q6cx69qngexz0t.so)，随后获取动态库的各个函数
- 通过`wrap_ibv_fork_init`避免fork引起RDMA网卡读写出错
- 由于IB网络也会通过socket进行带外网络的传输，所以这里也通过**`findInterfaces`**获取可用的网卡保存到ncclIbIfAddr
- 通过`ibv_get_device_list`获取所有rdma设备到devices中，遍历devices的每个device，因为每个HCA可能有多个物理port，所以对每个device遍历每一个物理port，获取每个port的信息，然后将相关信息保存到全局的ncclIbDevs中，比如是哪个device的哪个port，使用的是IB还是ROCE，device的pci路径，maxqp，device的name等**，注意这里也有类似bootstrap网络NCCL_SOCKET_IFNAME的环境变量，叫NCCL_IB_HCA，**可以指定使用哪个IB HCA
- 初始化过程结束，总结来看，为获取了当前机器上的所有可用的IB网卡和普通以太网卡然后保存下来

完整代码如下：

```
ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if (ncclParamIbDisable()) return ncclInternalError;

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      if (findInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        return ncclInternalError;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      char* userIbEnv = getenv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) return ncclInternalError;

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/IB : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          TRACE(NCCL_INIT|NCCL_NET,"NET/IB: [%d] %s:%d/%s ", d, devices[d]->name, port,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          NCCLCHECK(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          ncclNIbDevs++;
          nPorts++;
          pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { return ncclInternalError; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    } else {
      char line[1024];
      line[0] = '\\0';
      for (int d=0; d<ncclNIbDevs; d++) {
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
            ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
      }
      line[1023] = '\\0';
      char addrline[1024];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s ; OOB %s:%s", line, ncclIbIfName, socketToString(&ncclIbIfAddr.sa, addrline));
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

```

完成初始化后即可生成`UniqueId`

```
ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  ncclNetHandle_t* netHandle = (ncclNetHandle_t*) id;
  void* listenComm;
  NCCLCHECK(bootstrapNetListen(idFromEnv ? dontCareIf : 0, netHandle, &listenComm));
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, listenComm);
  return ncclSuccess;
}

```

ncclNetHandle_t也是一个字符数组，然后执行`bootstrapNetListen`

```
static ncclResult_t bootstrapNetListen(int dev, ncclNetHandle_t* netHandle, void** listenComm) {
  union socketAddress* connectAddr = (union socketAddress*) netHandle;
  static_assert(sizeof(union socketAddress) < NCCL_NET_HANDLE_MAXSIZE, "union socketAddress size is too large");
  // if dev >= 0, listen based on dev
  if (dev >= 0) {
    NCCLCHECK(bootstrapNetGetSocketAddr(dev, connectAddr));
  } else if (dev == findSubnetIf) {
    ...
  } // Otherwise, handle stores a local address
  struct bootstrapNetComm* comm;
  NCCLCHECK(bootstrapNetNewComm(&comm));
  NCCLCHECK(createListenSocket(&comm->fd, connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

```

首先会通过调用`bootstrapNetGetSocketAddr`获取一个可用的ip地址

```
static ncclResult_t bootstrapNetGetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= bootstrapNetIfs) return ncclInternalError;
  memcpy(addr, bootstrapNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

```

此时dev是0， bootstrapNetIfs是初始化bootstrap网络的时候一共找到了几个可用的网卡，这里就是获取了第0个可用的ip地址。

然后是通过`bootstrapNetNewComm`创建bootstrapNetComm，bootstrapNetComm其实就是fd，bootstrapNetNewComm其实就是new了一个bootstrapNetComm。

随后通过`createListenSocket`启动socket server

```
static ncclResult_t createListenSocket(int *fd, union socketAddress *localAddr) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return ncclSystemError;
  }

  if (socketToPort(&localAddr->sa)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#else
    SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // localAddr port should be 0 (Any port)
  SYSCHECK(bind(sockfd, &localAddr->sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  SYSCHECK(getsockname(sockfd, &localAddr->sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[1024];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", socketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sockfd, 16384), "listen");
  *fd = sockfd;
  return ncclSuccess;
}

```

创建监听fd，ip由localaddr指定，初始端口为0，bind时随机找一个可用端口，并通过getsockname(sockfd, &localAddr->sa, &size)将ip端口写回到localaddr，这里localaddr就是UniqueId。

到这里UniqueId也就产生了，其实就是当前机器的ip和port。