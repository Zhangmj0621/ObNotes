### deepEP编译命令

NVSHMEM_DIR=/usr/local/nvshmem NVSHMEM_LIB_DIR=/usr/local/nvshmem/lib python3 [setup.py](http://setup.py) clean

NVSHMEM_DIR=/usr/local/nvshmem NVSHMEM_LIB_DIR=/usr/local/nvshmem/lib python3 [setup.py](http://setup.py) build

NVSHMEM_DIR=/usr/local/nvshmem NVSHMEM_LIB_DIR=/usr/local/nvshmem**/lib** python3 [setup.py](http://setup.py) install

### nvshmem编译

在一台机器即可

CUDA_HOME=/usr/local/cuda/ && \

NVSHMEM_SHMEM_SUPPORT=0 \

NVSHMEM_UCX_SUPPORT=0 \

NVSHMEM_USE_NCCL=0 \

NVSHMEM_IBGDA_SUPPORT=1 \

NVSHMEM_PMIX_SUPPORT=0 \

NVSHMEM_MPI_SUPPORT=0 \

NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \

cmake -S . -B build/

cd build

make -j$(nproc)

在所有机器 最好分别 或者有先后 同时install 有时候会装不上

make install

### 平湖pod进入方式

nerdctl exec -it ep_test bash