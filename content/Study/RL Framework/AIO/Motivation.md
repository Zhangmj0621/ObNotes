题目可能得重新起：Orchestrating Streaming Rollouts and Elastic Environments for Agentic RL Training (?)

# Better agentic rollout pipeline
## Motivation (补)：
- Turn-level agentic bubble (sample和环境交互的时候导致GPU空转，并没有真正的打满batchsize)；放rollout时GPU利用率的曲线？
- Kvcache eviction (multi-turn的交互在长序列下可能频繁导致正在和环境交互的请求的kvcache被evict，被动的kvcache eviction不再满足需求)；放一个rollout step内，kvcache显存占用和实际cache命中率的两条曲线？
解决这两个问题是non-trivial的：
- 如果简单通过动态插入新请求的方式来最大化GPU利用率，第一可能加剧off-policy，加剧“distribution skew”，第二，GPU利用率满并不直接反映于端到端效率的提升，如RollFlash一样的aggressive overlap可能导致原先和环境交互的请求返回后被迫等待插入的新请求完成，反而阻塞端到端进程，具体的实验结果放在实验章节中，第三可能导致kvcache thrashing的问题被进一步放大，导致端到端效率严重受损；
- 现有工作如Concur通过检测kvcache占用率，动态的降低请求的速率来保证kvcache占用率始终维持在安全水线，thunderAgent通过主动的在kvcache占用率过高时，丢弃短请求来保证长请求的kvcache命中率，然而，他们都没能解决Turn-level agentic bubble的问题；如sglang的hicache支持把kvcache从HBM卸载到DRAM中，能够解决kvcache被trashing的问题，然而，从DRAM load/store kvcache到HBM可能会引入额外的overhead，且同样没能够解决turn-level agentic bubble的问题；
# Better environment interaction
## Motivation (补)
- 每个框架往往需要为异构的环境交互/资源管理额外做出很多重复性的适配，缺乏一个统一的IOLayer，向上屏蔽所有底层细节，提供production-level的IO primitives，使得无论多复杂的环境对于强化框架而言，只是调用一个统一格式的api一样简单；(Environment-as-a-service)
- 现有框架依赖于外部的特定服务如K8S等等，沙箱会占据单个轨迹的完整生命周期，昂贵；同时，沙箱本身其实是task-speicific的，直接依赖外部服务没有办法做到更好的复用沙箱；同时，沙箱往往按照固定的规格进行分配，然而，并不是所有的task都适用于这样的分配方式；
- ……
然而，解决上面的问题同样是non-trivial的：
- 对于问题1，虽然verl-tool，rStar2-Agent等等提供了一定思路，然而，他们简单的将并发数等同于CPU使用核心，没有真正做到在隔离的沙箱内运行，因此，必须把通用的IOLayer扩展到复杂的MicroVM如E2B上
- 在E2B实现动态的弹性扩缩容又不简单，第一需要profile具体适合每个task的资源数，少了可能反而会严重损害interaction效率，第二怎么支持MicroVM的动态弹性缩容也不简单，需要解决页表项的重挂载等等
- ……
# 设计
为此，我们提出AIO (?)，核心具有如下的设计
- Batch Conductor：
    - Design Goal：在不影响原先请求的执行次序等等前提上，non-blocking的执行高优先级请求，best-effort的调度低优先级 (新)请求执行generation和interaction；
         分析单个sample的耗时图，我们大概能够总结每个sample主要的耗时分为如下几步；我们期望对高优先级请求尽可能non-blocking执行 ，而低优先级请求则best-effort执行；$$T_{rollout} = T_{env\_init}+T_{gen}+T_{pre\_gen}+T_{act}+T_{pre\_act}$$
        
        ![](https://infrawaves.feishu.cn/space/api/box/stream/download/asynccode/?code=OWY1MmJkMjY3MjVjZDM1NDI3MTJlNTg5ODgzMTY5NDhfMnZ3MkdQcVBXRml1MTZBWGh2cGlFTFZKRVlNS2xSVTBfVG9rZW46VUZyUGJqVk1Db2dtc014VzVKVmNCbVdhbmRlXzE3NzU2Mjg0ODU6MTc3NTYzMjA4NV9WNA)
        
        - $T_{env\_init}$：拉取镜像，第一次启动沙箱，能够很好的利用前一个样本的生成阶段来overlap拉取镜像的overhead；
        - $T_{pre\_gen}$：两部分开销，假设目前的running请求数等于batchsize (max-running-requests)，则新到达请求需要等待正在运行请求完成，因此带来queueing时间；此外，若kvcache之前从HBM offload到DRAM中，则还有额外load/store与GPU-CPU同步的开销；
        - $T_{gen}$：模型生成时间，可能受到的影响是multi-turn场景下，序列长度较长且不可控时，即使请求个数等同于batchsize，也可能会出现kvcache被thrashing的问题，导致请求被abort，kvcache被evict；todo: pd-disaggregation
        - $T_{pre\_act}$：两部分开销，若资源受限，则需要等待目前正在执行的沙箱被释放；环境从快照恢复也有延迟；
        - $T_{act}$：环境交互时间，主要取决于分配的资源；
    - 优先级区分：将请求分为两类，高优先级请求和低优先级请求，高优先级请求指目前的batchsize个请求；低优先级请求指后续插入的请求；
    - Priority-based rollout scheduling Algorithm：首先定义Effective Kvcache usage，代表高优先级请求在rollout DP中占用的实际kvcache池比例，即 $E_{kv} = U_{priority\_kv} / U_{max}$，~~同时定义active kvcache usage，代表正在生成的所有请求在rollout DP中占用的实际kvcache池比例，即~~分别针对高优先级请求/低优先级请求采用不同的调度策略：
        - 高优先级请求 (bounded preemption scheduling)：
            - First phase：当 $E_{kv} < \lambda$ ( $\lambda$的具体取值可以根据预测来，可以计算出不同的model下，实际还可以存储的token数 $(1-\lambda) * U_{max}$)，通过预测的方法判断是否该turn有概率会导致kvcache eviction)，即认为目前仅仅高优先级请求的并发推理并不会导致kvcache eviction时，则将所有的KVCache全部驻留在HBM中，而不offload到DRAM中，这样的好处是可以完全避免在DRAM与HBM中传输数据带来的PCIe开销，以及完全跳过GPU-CPU的同步，同时，允许高优先级请求随时需要执行时，立刻打断低优先级请求，并主动将其在显存中剩余的kvcache全部offload到DRAM中 (避免kvcache被reactive的evict)，通过这种方式，能够将$T_{pre\_gen}$近乎降为0，且$T_{gen}$严格等于 $T_{llm}$；环境交互部分，则通过调用IO Layer API时的flag来指定，具体的环境资源分配优化在IOLayer中展示；
            - %% 假设还是发生了kvcache eviction，怎么办？有没有什么fallback来严格避免任何可能的reactive kvcache thrashing? %%
            - Second phase：当 $\lambda \leq E_{kv} < 1$，则认为目前的高优先级请求的并发推理，已经可能会导致所谓的kvcache eviction，则借用LayerKV的思路，采用grouped offload 模式，逐步只在HBM中放部分层的KVcache，而不存放全量的kvcache，具体的存放采用二进制指数下降，例如，模型共有64层，则只在HBM中贮存0-31层的kvcache，计算第0层时，load第32层kvcache，在算完第0层后，将第0层store回DRAM，通过这种方法，能保证kvcache压力成倍下降，此时，
                        $$T_{pre\_gen} = \begin{cases} T_{store}, & layer\_index = N-1 \\ T_{load}-T_{compute}*N, & T_{load}>T_{compute} * N \\ 0, & \text{else} \end{cases}$$
                  其中，由于TP/DP等并行策略，N往往不需要降低到非常小的值，因此 $T_{pre\_gen}$可近乎等于 $T_{store}$，同样的，由于采用proactive的kvcache management，$T_{gen}$严格等于 $T_{llm}$；同时，需要注意的是， $\lambda \leftarrow 1- (1-\lambda)/2$，原因在于当保持并发度 (bsz) 固定时，请求对kvcache美轮的压力可近似稳定，而随着驻留在HBM中的kvcache层数指数级下降，则请求对kvcache的压力增长也会下降，更不容易造成kvcache eviction；
        - 低优先级请求 (best-effort scheduling)：
            - 为了降低复杂度，针对低优先级请求，完全不在HBM中驻留任何kvcache，而是全部存放于DRAM中，每次forward前，load第0层的kvcache，计算第N层时，overlap的load第1层的kvcache，这样设计有两个初衷 1）避免低优先级请求可能的将高优先级请求贮存在HBM中的kvcache offload到DRAM中；2）当低优先级请求被高优先级请求打断时，往往需要将低优先级请求的显存全部offload到DRAM中，每次laywise的load/store kvcache能保证被abort时，实际的kvcache offload 到DRAM的时间尽可能小；
                %% 这里其实还有一个可能能优化的点，/abort_request需要等待一轮forward_pass结束后，才能成功打断请求，这个开销大致在10ms，可能可以不考虑优化，但在论文里可以提到这个开销是完全可以接受的 %%
        - 已完成请求：所有已完成的请求会通知sglang engine，将对应请求的kvcache统一先offload到DRAM中；