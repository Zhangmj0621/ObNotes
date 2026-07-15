# Environment

## NVSHMEM

```
# download
git clone <http://183.207.7.174:8081/moon/nvshmem_3.2.5>

# compile
CUDA_HOME=/usr/local/cuda/ && \\
GDRCOPY_HOME=/opt/gdrcopy && \\
NVSHMEM_SHMEM_SUPPORT=0 \\
NVSHMEM_UCX_SUPPORT=0 \\
NVSHMEM_USE_NCCL=0 \\
NVSHMEM_MPI_SUPPORT=0 \\
NVSHMEM_IBGDA_SUPPORT=1 \\
NVSHMEM_PMIX_SUPPORT=0 \\
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \\
NVSHMEM_USE_GDRCOPY=1 \\
cmake -S . -B build/
cd build
make -j$(nproc)
make install

export NVSHMEM_DIR=/usr/local/nvshmem   # Use for DeepEP installation
export  LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export  PATH="${NVSHMEM_DIR}/bin:$PATH"

# examine
nvshmem-info -a

```

## DeepEP

```
# download and apply patch
# we use our patch in commit 7d52ad7248a13bc291ffcf54505c6a328311add9.
# It is theoretically compatible with all the version afterwards.
git clone <https://github.com/deepseek-ai/DeepEP.git>
git clone <http://183.207.7.174:8081/moon/deepep_multiqp_patch.git>
cd DeepEP
git apply ../deepep_multiqp_patch/0001-support-multiQP.patch

# compile
NVSHMEM_DIR=/usr/local/nvshmem NVSHMEM_LIB_DIR=/usr/local/nvshmem/lib python3 setup.py clean
NVSHMEM_DIR=/usr/local/nvshmem NVSHMEM_LIB_DIR=/usr/local/nvshmem/lib python3 setup.py build
NVSHMEM_DIR=/usr/local/nvshmem NVSHMEM_LIB_DIR=/usr/local/nvshmem/lib python3 setup.py install

# examine
# check if deep_ep_cpp.cpython-310-x86_64-linux-gnu.so in the two paths below is same
# if not, you should copy manually
ll /usr/local/lib/python3.10/dist-packages/deep_ep-1.0.0+231e17e-py3.10-linux-x86_64.egg
-rwxr-xr-x 1 root root 69675680 Apr 17 11:52 deep_ep_cpp.cpython-310-x86_64-linux-gnu.so*
-rw-r--r-- 1 root root      437 Mar 14 13:04 deep_ep_cpp.py

root@8538a57f1ee9:/workspace/infra/DeepEP-splitChannel/build/lib.linux-x86_64-cpython-310# ll
drwxr-xr-x 2 root root     4096 Apr 17 11:51 deep_ep/
-rwxr-xr-x 1 root root 69675680 Apr 17 11:51 deep_ep_cpp.cpython-310-x86_64-linux-gnu.so*

```