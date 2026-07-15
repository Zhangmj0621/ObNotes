ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

make -j64 src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90”

编译nccl-tests

make -j64 CUDA_HOME=/usr/local/cuda MPI=1 MPI_HOME=/usr/local/mpi/