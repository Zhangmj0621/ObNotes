## 四机测试

### gpu01 - gpu04 同spine deepEP性能

qp = 4

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 32, RDMA chunk 32: 55.25 GB/s (RDMA), 110.94 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 16, RDMA chunk 28: 59.14 GB/s (RDMA), 118.75 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 16: 56.66 GB/s (RDMA), 113.77 GB/s (NVL)

### gpu14 - gpu17 同spine deepEP性能

qp = 4

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 16, RDMA chunk 32: 55.12 GB/s (RDMA), 110.14 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 24, RDMA chunk 28: 59.03 GB/s (RDMA), 117.96 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 16: 57.18 GB/s (RDMA), 114.25 GB/s (NVL)

## 八机测试

### gpu01 - gpu04, gpu14 - gpu17 qp = 1

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 16, RDMA chunk 12: 26.03 GB/s (RDMA), 48.48 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 20, RDMA chunk 8: 25.86 GB/s (RDMA), 48.15 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 3, RDMA chunk 8: 30.30 GB/s (RDMA), 56.42 GB/s (NVL)

### gpu01 - gpu04, gpu14 - gpu17 qp = 2

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 32, RDMA chunk 28: 47.70 GB/s (RDMA), 88.83 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 24, RDMA chunk 32: 50.22 GB/s (RDMA), 93.52 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 16: 48.63 GB/s (RDMA), 90.57 GB/s (NVL)

### gpu01 - gpu04, gpu14 - gpu17 qp = 4

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 24, RDMA chunk 32: 47.68 GB/s (RDMA), 89.67 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 28, RDMA chunk 16: 50.66 GB/s (RDMA), 95.29 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 16: 49.43 GB/s (RDMA), 92.97 GB/s (NVL)

### gpu01 - gpu04, gpu14 - gpu17 qp = 8

[tuning] Best dispatch (FP8): SMs 24, NVL chunk 16, RDMA chunk 28: 46.26 GB/s (RDMA), 86.03 GB/s (NVL)

[tuning] Best dispatch (BF16): SMs 24, NVL chunk 4, RDMA chunk 16: 49.15 GB/s (RDMA), 91.40 GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk 1, RDMA chunk 12: 44.67 GB/s (RDMA), 83.07 GB/s (NVL)