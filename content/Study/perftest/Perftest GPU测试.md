# 下载perftest

1. 方法一

`https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/`

1. 方法二

`git clone <https://github.com/linux-rdma/perftest.git`>

# 编译

```
>>> cd perftest/
>>> ./autogen.sh && ./configure CUDA_H_PATH=<path to cuda.h> && make -j, e.g.:
>>> ./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && make -j

```

# 运行

以一对一write为例（需要在perftest目录下执行）

```
// server
>>> ./ib_write_bw -d mlx5_0 --use_cuda=<gpu index> -a -p 10002 -F --run_infinitely --report_gbits

// client
>>> ./ib_write_bw -d mlx5_0 --use_cuda=<gpu index> -a -p 10002 -F --run_infinitely --report_gbits <server_ip_address>

```

可能的执行结果如下图

[](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=MzAxMGMyMWY3YzM2ZTJjZDQyYmVhYTI5MTRkZjc0MmNfcXVuZWZna0JlWmlEdkpLeWdoN0poVk1ZZ2RhanZsOUtfVG9rZW46S3pPWWJINnJ3bzFvVnN4OTd6QmNiOWN1bjZnXzE3NDk4MjEzMTg6MTc0OTgyNDkxOF9WNA)

# 可能问题

1. `configure: error: pciutils header files not found, consider installing pciutils-devel`

解决方法：

```
>>> sudo apt-get install libpci-dev

```

# GPU显存监控

```
// 以T（单位秒/s）为周期查看GPU使用情况
>>> watch -n T nvidia-smi

```