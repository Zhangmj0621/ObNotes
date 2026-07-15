# 1. Abstract

本文章解决的问题是，在模型训练Tensor Parallelism带来的通信与计算overlapping，其中，提出了利用更细粒度的并行，将tp通信组内的reduce_scatter和all_gather实现到一个fuse kernel内，在不compromise GPU利用率的同时，最小化Effective Communication Time (ECT)；

# 2. Motivation

之前采用的通信与计算overlapping策略，