# perftest是什么

perftest是一组基于uverbs编写的测试程序，是RDMA性能相关的benchmark。可用于软硬件调优以及功能测试。

perftest测试软件包含的测试集包括

- Send - ib_send_bw and ib_send_lat
- RDMA Read - ib_read_bw and ib_read_lat
- RDMA Write - ib_write_bw and ib_write_lat
- RDMA Atomic - ib_atomic_bw and ib_atomic_lat
- Native Ethernet (when working with MOFED2) - raw_ethernet_bw, raw_ethernet_lat

源码仓库https://github.com/linux-rdma/perftest

# 安装perftest

```
git clone <https://github.com/linux-rdma/perftest.git>
-After cloning, Follow this commands:

-cd perftest/

-./autogen.sh

-./configure    Note:If you want to install in a specific directory use the optional flag --prefix=<Directory path> , e.g: ./configure --prefix=<Directory path>

-make

-make install

-All of the tests will appear in the  perftest directory and in the install directory.

```

# perftest基本测试

一对一

```
// server
ib_send_bw -d mlx5_0 -p 10002 -F --run_infinitely --report_gbits

// client
ib_send_bw -d mlx5_0 -p 10002 -F --run_infinitely --report_gbits ip_address

```

二对二

```
//server
ib_send_bw -d mlx5_0 -p 10002 -F --run_infinitely --report_gbits
ib_send_bw -d mlx5_1 -p 10003 -F --run_infinitely --report_gbits

//client
ib_send_bw -d mlx5_0 -p 10002 -F --run_infinitely --report_gbits ip_address
ib_send_bw -d mlx5_1 -p 10003 -F --run_infinitely --report_gbits ip_address

```

Option

```
Options:
  -a, --all  Run sizes from 2 till 2^23
  -b, --bidirectional  测量双向带宽（默认为单向）
  -c, --connection=  连接类型RC / XRC / UC / DC（默认RC）
  -d, --ib-dev=  使用IB设备（找到第一个默认设备）
  -D, --duration  在自定义的秒数内运行测试。
  -f, --margin  measure results within margins. (default=2sec)
  -F, --CPU-freq  即使已加载cpufreq_ondemand模块，并且cpu-freq不在最大值，也不会显示警告。
  -h, --help  Show this help screen.
  -i, --ib-port=  使用IB设备的端口<端口>（默认1）
  -I, --inline_size=  Max size of message to be sent in inline
  -l, --post_list= Post list of WQEs of size (instead of single post)
  -L, --hop_limit=  设置跳数限制值（对于IPv4 RawEth QP，为ttl）。值0-255（默认64）Set hop limit value (ttl for IPv4 RawEth QP). Values 0-255 (default 64)
  -m, --mtu=  MTU size : 256 - 4096 (default port mtu)
  -n, --iters=  交换次数（至少5次，默认为5000次）
  -N, --noPeak 取消峰值BW计算（默认情况下峰值不超过iters = 20000）
  -O, --dualport  在双端口模式下运行测试。（Run test in dual-port mode.）
  -p, --port=  Listen on/connect to port (default 18515)
  -q, --qp=  qp的数量（默认为1）
  -Q, --cq-mod  仅在<-cq-mod>完成后生成Cqe
  -R, --rdma_cm  Connect QPs with rdma_cm and run test on those QPs
  -s, --size=  （每个QP）交换消息的大小（默认为65536）
  -S, --sl=  SL (default 0)
  -t, --tx-depth=  发送队列（tx）的大小（默认为128）
  -T, --tos=  Set to RDMA-CM QPs. available only with -R flag. values 0-256 (default off)
  -u, --qp-timeout= QP超时，超时值为4 usec * 2 ^（超时），默认值为14
  -V, --version  显示版本号
  -w, --limit_bw=  设置验证器带宽限制(Set verifier limit for bandwidth)
  -x, --gid-index=  Test uses GID with GID index (Default : IB - no gid . ETH - 0)
  -y, --limit_msgrate=  Set verifier limit for Msg Rate
  -z, --com_rdma_cm  Communicate with rdma_cm module to exchange data - use regular QPs

      --cpu_util  在报告中显示CPU使用率，仅在持续时间模式下有效(Show CPU Utilization in report, valid only in Duration mode )
      --dlid  Set a Destination LID instead of getting it from the other side.
      --dont_xchg_versions  Do not exchange versions and MTU with other side
      --force-link=  Force the link(s) to a specific type: IB or Ethernet.
      --ipv6  Use IPv6 GID. Default is IPv4
      --mmap=file  Use an mmap'd file as the buffer for testing P2P transfers.
      --mmap-offset=  Use an mmap'd file as the buffer for testing P2P transfers.
      --mr_per_qp  Create memory region for each qp.
      --odp  Use On Demand Paging instead of Memory Registration.
      --output=   设置详细度输出级别：带宽，message_rate，延迟 Set verbosity output level: bandwidth , message_rate, latency
 Latency measurement is Average calculation
      --perform_warm_up Perform some iterations before start measuring in order to warming-up memory cache, valid in Atomic, Read and Write BW tests
      --pkey_index= PKey index to use for QP
      --report-both  Report RX & TX results separately on Bidirectinal BW tests
      --report_gbits  Report Max/Average BW of test in Gbit/sec (instead of MB/sec)
        Note: MB=2^20 byte, while Gb=10^9 bits. Use these formulas for conversion:
        Factor=10^9/(20^2*8)=119.2; MB=Gb_result * factor; Gb=MB_result / factor
      --report-per-port  Report BW data on both ports when running Dualport and Duration mode
      --reversed  Reverse traffic direction - Server send to client
      --run_infinitely  永久运行测试，每隔秒打印结果
      --retry_count=  Set retry count value in rdma_cm mode
      --tclass=  Set the Traffic Class in GRH (if GRH is in use)
      --use_exp  Use Experimental verbs in data path. Default is OFF.
      --use_hugepages  Use Hugepages instead of contig, memalign allocations.
      --use_res_domain  Use shared resource domain
      --verb_type=  Set verb type: normal, accl. Default is normal.
      --wait_destroy=  Wait before destroying allocated resources (QP/CQ/PD/MR..)

 Rate Limiter:
      --burst_size= Set the amount of messages to send in a burst when using rate limiter
      --typical_pkt_size= Set the size of packet to send in a burst. Only supports PP rate limiter
      --rate_limit= Set the maximum rate of sent packages. default unit is [Gbps]. use --rate_units to change that.
      --rate_units= [Mgp] Set the units for rate limit to MBps (M), Gbps (g) or pps (p). default is Gbps (g).
        Note (1): pps not supported with HW limit.
        Note (2): When using PP rate_units is forced to Kbps.
      --rate_limit_type= [HW/SW/PP] Limit the QP's by HW, PP or by SW. Disabled by default. When rate_limit is not specified HW limit is Default.
        Note: in Latency under load test SW rate limit is forced
      --use_ooo  Use out of order data placement

```

# 注意

若出现要求安装pciutils-devel要求，执行

```
sudo apt-get install libpci-dev

```