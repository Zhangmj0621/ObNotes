## Mooncake Transfer Engine ibgda支持（更快的transfer）

可以预见，在KVCache这种low latency场景，在TE中实现ibgda，能极大的提高通信效率，具体衡量指标可能有

- Time To First Token(TTFT) general(存疑）
- 在短文本负载场景下，是否存在特定模型架构使得通信无法完全被decoding overlap，体现在TBT上
- 目前Mooncake的TE是CPU侧的同步，从实现上来讲，CPU占用率也能作为一个衡量指

## Explict Hot-spot KVCache transfer（尽可能少的transfer）

实现一种基于预测的hot-spot KVCache pre-transfer，具体思路为，在KVcache-centric Schedule Algorithm中，若选取非best prefix match的节点作为prefill node，会存在从best prefix len的node到选取的prefill node的多余KVCache Transfer，在完成transfer后，prefill node才能开始计算

可能的衡量指标

- Time To First Token(TTFT)
- ……

## Something to do

1. 需要有一个全局的conductor，来实现类似KVCache-centric Scheduler算法，**才会有prefill到prefill的通信需求**（SGLang中的Schedule实际上解决的是，同一个waiting queue中的调度执行顺序），这个conductor的实现包括
    1. A predictive model derived from offline test data for predict computation time.
    2. Calculate the queuing time by aggregating the prefill times of all queued requests.
    3. Predict the network transfer time.
2. 在1的基础上，才能考虑，如何通过修改transfer logic来实现explicit hot-spot KVCache transfer(由于KVCache的通信量并不大，考虑在完成前一部分KVCache的transfer后，先发送amo通知，随后异步发送hot-spot的KVCache，这部分发送不需要amo通知receiver)，具体可能需要实现和设计的包括
    1. A hot-spot predictive Algorithm.
    2. A potential model for perceiving the current network state, by which we can determine not to transfer extra KVCache in overload situation.
3. 在prefill的计算kernel内，直接调用对应实现的send/recv接口