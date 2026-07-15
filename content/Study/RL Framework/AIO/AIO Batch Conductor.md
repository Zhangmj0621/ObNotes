# Motivation
Better agentic rollout pipeline
- Turn-level agentic bubble (sample和环境交互的时候导致GPU空转，并没有真正的打满batchsize)；放rollout时GPU利用率的曲线？
- Kvcache eviction (multi-turn的交互在长序列下可能频繁导致正在和环境交互的请求的kvcache被evict，被动的kvcache eviction不再满足需求)；放一个rollout step内，kvcache显存占用和实际cache命中率的两条曲线？
解决这两个问题是non-trivial的：
- 如果简单通过动态插入新请求的方式来最大化GPU利用率，第一可能加剧off-policy，加剧“distribution skew”，第二，GPU利用率满并不直接反映于端到端效率的提升，如RollFlash一样的aggressive overlap可能导致原先和环境交互的请求返回后被迫等待插入的新请求完成，反而阻塞端到端进程，具体的实验结果放在实验章节中，第三可能导致kvcache thrashing的问题被进一步放大，导致端到端效率严重受损；
- 现有工作如Concur通过检测kvcache占用率，动态的降低请求的速率来保证kvcache占用率始终维持在安全水线，thunderAgent通过主动的在kvcache占用率过高时，丢弃短请求来保证长请求的kvcache命中率，然而，他们都没能解决Turn-level agentic bubble的问题；如sglang的hicache支持把kvcache从HBM卸载到DRAM中，能够解决kvcache被trashing的问题，然而，从DRAM load/store kvcache到HBM可能会引入额外的overhead，且同样没能够解决turn-level agentic bubble的问题；
# Solution
为此，我们提出AIO Batch Conductor，核心具有如下三个系统组件：
* Admission Controller：严格控制是否允许插入新请求到running queue中进行调度 (主要受staleness影响，严格控制off-policy程度)；
* Rollout Scheduler：负责实际管理请求的完整生命周期，agentloop的运行方式，请求的状态切换等等，是真正batch conductor的核心；
* Proactive KVCache manager：负责主动的管理HBM/DRAM中KVCache，根据Rollout Scheduler的信号动态的调整不同 token KVCache的管理方式；
![[AIO Batch Conductor 2026-04-10 10.24.37.excalidraw.svg]]
%%[[AIO Batch Conductor 2026-04-10 10.24.37.excalidraw.md|🖋 Edit in Excalidraw]]%%
![[Pasted image 20260410103501.png|1043]]
## Admission Controller
* Async_factor控制off-policy程度：设置异步参数async_factor，严格控制，system中对于该step，最多存在async_factor * global_batch_size个staleness请求，即对于第k个step，最多存在 (k + async_factor) * global_batch_size个请求进入system；
* Prefetch data：为了降低从Dataloader中prefetch数据的开销，我们提前从dataloader从预取global_batch_size个数据到队列中，使得rollout Scheduler获取sample并执行调度的同时，能够和dataloader fetch数据的时间做overlap；同时，我们同步修改load/store checkpoint的逻辑，确保不会由于预取而导致恢复时sample丢失；
## Rollout Scheduler
core of batch conductor, decision maker；
* action-level agentloop：就像RollFlash做的一样，原先agentloop从最早的batch-level management到trajectory-level management，然而我们觉得需要更进一步，借鉴了RollFLash的思路，我们将单个agentloop break成不同的action，每个request在不同的action (state)内切换和管理，做到了完全的环境交互与生成解耦；
* priority-based scheduling algorithm：为了解决所提出的turn-level agentic bubble的问题，RollFLash将环境与生成完全分离，确保生成的请求个数始终等同于所需的bsz，优点在于确实始终打满了GPU利用率，然而，他没能考虑到原先运行的请求可能会有额外的生成排队时间，反而劣化每个step的端到端训练时间，同时，也未能解决kvcache被trashing的问题，因此，rollout scheduler提出priority-based scheduling algorithm，design goal为在严格最大化有效请求的推理效率同时，best-effort的oppotunisitic 推理新请求；
  * 有效请求定义：直接影响端到端推理效率的请求，定义为正在running_queue中运行的，按照dataloader顺序靠前的前global_batch_size个请求；维护原先的双链表进行动态优先级调整；
  * 有效请求推理效率：一个请求的完整运行总耗时可建模为如下：$T_{rollout} = T_{env\_init}+n* (T_{gen}+T_{pre\_gen}+T_{act}+T_{pre\_act})$，其中n为multi-turn的轮数，观察真正对于单个请求端到端推理有效的时间为生成和交互的时间，因此定义有效请求推理效率为 $E_{rollout} = n*(T_{gen}+T_{act})/T_{rollout}$，由于环境初始化时间可被上一个请求的生成和交互时间overlap，因此，可近似认为$E_{rollout} = (T_{gen}+T_{act})/ (T_{gen}+T_{pre\_gen}+T_{act}+T_{pre\_act})$，因此最大化有效请求推理效率，也即最小化 $T_{pre\_gen}$与$T_{pre\_act}$，其中 $T_{pre\_gen}$主要分为两部分开销，假设目前的running请求数等于batchsize (max-running-requests)，则新到达请求需要等待正在运行请求完成，因此带来queueing时间；此外，若kvcache之前从HBM offload到DRAM中，则还有额外load/store与GPU-CPU同步的开销；而$T_{pre\_act}$则代表等待环境交互的时间，包括资源调度/环境冷启动，此处不加赘述，在IO layer处在讨论；为了最小化$T_{pre\_gen}$，因此需要保证 1）有效请求的排队时间最小； 2）从DRAM load/store kvcache的时间最小 (尽可能少的需要从多层缓存中加载kvcache，尤其考虑到IO竞争后，load/store时间可能会严重劣化)；
  * 基于上面的分析，提出priority-based scheduling algorithm，算法定义如下：每个rollout DP在正处于generation的请求个数小于global_batch_size / DP_size时候，则按照如下的顺序进行轮训，获取首个可用的sample进行推理：1）由于权重更新而partial rollout的sample，priority=high； 2）刚结束环境交互的sample，priority=high；3）由于权重更新而partial rollout的sample，priority=low；4）刚结束环境交互的sample，priority=low；5）新sample，priority=low；注意，若获取到一个高优先级sample，且系统中正存在低优先级sample推理，则直接abort该低优先级请求，抢占执行；
## KVCache Manager
负责管理底层的KVCache组织形式，需要将无状态被动的kvcache管理改为token-aware 主动的kvcache管理，同时，应当根据实际的kvcache占用情况，动态的调整不同状态KVCache的存放位置；
![[Pasted image 20260414144416.png]]
* Dynamic KVCache buffer management：原先KVCache采用layer-first存放方式，每个token默认享有相同的优先级，当token到来时，统一分配所有层的显存，然而，在Rollout Scheduler这种priority-based的scheduling下，不同token能够占用的kvcache大小不再一致，例如有效请求可能仍然在HBM中享有num_layers层的全量KVCache，尽可能避免任何的由于IO contention而导致的load/store影响有效推理效率，而best-effort请求则可能只占用2层kvcache的显存空间，laywise的进行加载和计算，且best-effort请求可能会随时切换为有效请求，静态的buffer分配不再满足需求；
![[Pasted image 20260410152229.png]] 
%%[[AIO Batch Conductor 2026-04-10 15.15.53.excalidraw.md|🖋 Edit in Excalidraw]]%%
![[Pasted image 20260410153127.png]]
* Proactive kvcache eviction：针对不同的请求 (token)需要采用不同的KVCache管理方式，主动的控制offload到DRAM中的kvcache是属于哪些token的，例如，对于低优先级请求，当其生成完成后，主动的将其对应的kvcache卸载到DRAM中 (需要考虑RadixCache的影响)，而高优先级请求，则尽可能的将KVCache全部驻留在HBM内；同时，为了避免KVCache超出可用大小，而触发evict操作，需要提前采用group offloaded操作 (根据实际的占用情况，在每次forward_decode(...)与forward_extend(...)前进行判断，不再需要tricky的$\lambda$)，采用二进制退避方法，逐步减少有效请求在HBM中驻留的KVCache总量；当有高优先级请求完成后，需要再进行决策是否回退；