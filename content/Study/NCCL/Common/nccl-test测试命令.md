宁夏

mpirun -np 16 --host 10.101.5.101:8,10.101.5.102:8 --allow-run-as-root -mca btl ^openib -mca routed_radix 600 -mca plm_rsh_no_tree_spawn 1 -x OMPI_MCA_btl_tcp_if_include="bond4" --mca oob_tcp_if_include bond4 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_SOCKET_IFNAME=bond4 -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=160 -x NCCL_NET_GDR_LEVEL=4 -x NCCL_DEBUG=version -x LD_LIBRARY_PATH=./build/lib/ -x NCCL_TELEMETRY_ENABLE=1 -x NCCL_TELEMETRY_LOG_PATH=../logs/ ../../nccl-tests/build/all_gather_perf -b 8G -e 8G -f 2 -g 1 -w 0

不断打 -f 0 -i 0

输出topo -x NCCL_TOPO_DUMP_FILE=./topo.xml

mpirun -np 2 --host 10.101.5.101:1,10.101.5.102:1 --allow-run-as-root -mca btl ^openib -mca routed_radix 600 -mca plm_rsh_no_tree_spawn 1 -x OMPI_MCA_btl_tcp_if_include="bond4" --mca oob_tcp_if_include bond4 -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_SOCKET_IFNAME=bond4 -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=160 -x NCCL_NET_GDR_LEVEL=4 -x NCCL_IB_TIMEOUT=1 -x NCCL_MAX_NCHANNELS=1 -x UCX_NET_DEVICES=mlx5_4:1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH=./build/lib/ ../../nccl-tests/build/sendrecv_perf -b 8G -e 8G -f 0 -i 0 -n 1 -w 0

mpirun -np 16 --host 10.101.6.127:8,10.101.6.128:8 --allow-run-as-root -mca btl ^openib -mca routed_radix 600 -mca plm_rsh_no_tree_spawn 1 -x OMPI_MCA_btl_tcp_if_include="bond4" --mca oob_tcp_if_include bond4 -x NCCL_IB_QPS_PER_CONNECTION=1 -x NCCL_SOCKET_IFNAME=bond4 -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=160 -x NCCL_NET_GDR_LEVEL=4 -x NCCL_MAX_NCHANNELS=16 -x UCX_NET_DEVICES=mlx5_4:1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH=./build/lib/ ../../nccl-tests/build/reduce_scatter_perf -b 8G -e 8G -f 0 -i 0 -n 1 -w 0

mpirun --allow-run-as-root -np 16 -H 10.1.3.201:8,10.1.3.202:8 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH=../vccl_2.21.51x/build/lib/ -x NCCL_SOCKET_IFNAME=enp59s0np0 -x NCCL_IB_GID_INDEX=3 ./build/all_gather_perf -b 1G -e 1G -f 2 -g 1

临港

mpirun --allow-run-as-root -np 16 --host 10.1.3.201:8,10.1.3.205:8 -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include="enp86s0f1" -mca btl_tcp_if_include enp86s0f1 --mca oob_tcp_if_include enp86s0f1 -x NCCL_SOCKET_IFNAME=enp86s0f1 -x NCCL_IB_HCA=mlx5_7,mlx5_5 -x UCX_NET_DEVICES=enp86s0f1 -x NCCL_DEBUG=INFO -x NCCL_IB_GID_INDEX=3 -x NCCL_ALGO=RING -x NCCL_PROTO=LL128 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/nccl/build/lib ./build/reduce_scatter_perf -b 128MB -e 8G -f 2 -g 1

mpirun -np 16 --allow-run-as-root --host 10.1.3.102:8,10.1.3.201:8 --mca oob_tcp_if_include enp86s0f1 -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=roce00,roce10,roce20,roce30,roce40,roce50,roce60,roce70 -x NCCL_IB_GID_INDEX=1 -x NCCL_SOCKET_IFNAME=enp86s0f1 -x UCX_NET_DEVICES=enp86s0f1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/share/zhangmj/vccl_2.21.51x/build/lib/ -x NCCL_IB_RETRY_CNT=4 -x NCCL_DEBUG=INFO ./build/reduce_scatter_perf -b 1G -e 8G -f 0 -i 0 -n 1

中兴网卡

mpirun -np 16 --allow-run-as-root --host 10.1.3.24:8,10.1.3.102:8 --mca oob_tcp_if_include enp86s0f1 -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=roce00,roce10,roce20,roce30,roce40,roce50,roce60,roce70,roce01,roce11,roce21,roce31,roce41,roce51,roce61,roce71 -x NCCL_IB_GID_INDEX=1 -x NCCL_SOCKET_IFNAME=enp86s0f1 -x UCX_NET_DEVICES=enp86s0f1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/share/zhangmj/vccl_2.21.51x/build/lib/ -x NCCL_TELEMETRY_ENABLE=1 -x NCCL_TELEMETRY_LOG_PATH=../zhangmj/logs/ -x NCCL_DEBUG=INFO ./build/all_reduce_perf -b 128MB -e 8G -f 2 -g 1

mpirun -np 16 --allow-run-as-root --host 10.1.3.201:8,10.1.3.102:8 --mca oob_tcp_if_include enp86s0f1 -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=roce00,roce10,roce20,roce30,roce40,roce50,roce60,roce70 -x NCCL_IB_GID_INDEX=1 -x NCCL_SOCKET_IFNAME=enp86s0f1 -x UCX_NET_DEVICES=enp86s0f1 -x NCCL_PXN_DISABLE=1 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/share/zhangmj/vccl_2.21.51x/build/lib/ -x NCCL_DEBUG=INFO ./build/all_reduce_perf -b 1G -e 8G -f 0 -i 0 -n 1

平湖

./mpirun -np 16 --host node188:8,node477:8 -genv NCCL_PXN_DISABLE=1 -genv UCX_NET_DEVICES=mlx5_0:1 -genv NCCL_ALGO=NVLSTree -genv NCCL_IB_GID_INDEX 3 -genv NCCL_SOCKET_IFNAME bond0 -genv NCCL_NET_GDR_LEVEL 4 -genv LD_LIBRARY_PATH=/mnt/nfs/zhangmj/vccl_2.21.51x/build/lib -genv NCCL_DEBUG=INFO /mnt/nfs/nccl-tests/build/all_reduce_perf -b 8G -e 8G -f 2 -g 1 -w 0

是石

mpirun -np 16 --allow-run-as-root --host 10.200.88.173:8,10.200.88.174:8 --mca oob_tcp_if_include bond0 -x NCCL_DEBUG=version -x NCCL_IB_HCA=mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7 -x NCCL_SOCKET_IFNAME=bond0 -x UCX_NET_DEVICES=bond0 -x NCCL_NET_GDR_LEVEL=4 -x NCCL_IB_QPS_PER_CONNECTION=2 -x NCCL_MIN_NCHANNELS=32 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/hv/vccl_2.21.51x/build/lib/ ./build/all_reduce_perf -b 128MB -e 8G -f 0 -i 0 -n 1

mpirun -np 16 --allow-run-as-root --host 10.200.89.107:8,10.200.89.108:8 --mca oob_tcp_if_include bond0 -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7 -x NCCL_SOCKET_IFNAME=bond0 -x UCX_NET_DEVICES=bond0 -x NCCL_NET_GDR_LEVEL=4 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/zhangmj/vccl_2.21.51x/build/lib/ ./build/all_reduce_perf -b 8G -e 8G -f 0 -i 0 -n 1

mpirun -np 16 --allow-run-as-root --host 10.200.89.107:8,10.200.89.108:8 --mca oob_tcp_if_include bond0 -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7 -x NCCL_SOCKET_IFNAME=bond0 -x UCX_NET_DEVICES=bond0 -x NCCL_NET_GDR_LEVEL=4 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/zhangmj/vccl_2.21.51x/build/lib/ ./build/alltoall_perf -b 8G -e 8G -f 0 -i 0 -n 1

- x NCCL_IB_QPS_PER_CONNECTION=1

台湾

mpirun -np 128 --hostfile hostfile --allow-run-as-root -x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=ens99f3 -x UCX_NET_DEVICES=ens99f3 -x NCCL_ALGO=RING -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/nccl/build/lib ./build/all_reduce_perf -b 128MB -e 8G -f 2 -g 1