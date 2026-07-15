原先定义高低优先级的逻辑是，先来的请求作为高优先级请求，后来oppotunistic加入的作为低优先级请求，然而，这显然并不合理而且太incremental，因此，我们需要在论文里写一个章节，仔细介绍我们的建模与优化目标；
# Problem Formulation
## Turn-level agentic bubble：
就像motivation中所说，agentic rl场景下multi-turn的交互范式，使得带来了 turn-level agentic bubble 的新问题，在swe-agent中，单条轨迹的环境交互时间占总时间的比例是多少 (别的论文是46%)，rollout worker的实际bsz并不高，decode吞吐受限，原先的trajectory-level的scheduling显然不再符合需求；因此我们提出了action-level scheduling，将rollout worker controller和environment worker controller完全解耦，保证了rollout worker永远不会陷入 "starvation"；
## Maximizing Effective Sample Throughput
就像rollflash做的那样，action-level scheduling能够很好的解决GPU starvation的问题，保证throughput达到理论上的最大值，然而，我们发现，Agentic Rollout的效率并不等同于Rollout Worker的throughput，即 "Higher throughput is not Equal to higher speed"，仅仅关注最大化吞吐并不充分；
在全异步的实现中，Rollout侧不断复制prompt，生成和处理Sample，当凑够一个prompt的完整的Group后，计算reward和advantage，将其传给Trainer进行消费，而Trainer凑够Micro_batch_size个训练Group后即可开始训练积累梯度，其中，我们设Group g为$g = \{r_{g,1}, r_{g,2}, \dots, r_{g,K}\}$，其中K代表强化算法PPO/GRPO中的Group Size，特定的对PPO，K=1；而对于g而言，其真正对Rollout pipeline起到contribution的点在于整个group均完成rollout的时刻，也即，$T_g = \max_{i=1}^{K} T_{g,i}$，其中，$T_{g,i} = T_{env\_init} + (T_{pre\_gen}, T_{gen}, T_{pre\_act}, T_{act})^{N_{g,i}}$，也即代表一个Sample的完整multi-turn的生成总时间；
对于Trainer，我们发现，真正制约其开始训练的时间点在于真正获得 B 个可用于Sample Group个时刻，设在不同的编排 $\pi$下，$T_{\text{next}}^\pi=\inf\left\{t:|\mathcal{G}_{ready}(t)| \ge B\right\}$，其中，$\mathcal{G}_{ready}(t)$代表在时刻t已经完成等待消费的group集合，也即，最大化端到端Rollout效率实则为最小化$\inf\left\{t:|\mathcal{G}_{ready}(t)| \ge B\right\}$，其中，最大化前B个可供Trainer消费的Group的吞吐才是真正的有效请求吞吐，而非先前仅仅关注rollout worker的吞吐；
# Priority-based Scheduling
## Critical Group Definition
为了实现最大化的优化目标，首先需要先确定，真正应该被优先执行的B个group，也即critical group到底应该如何定义，其中，在时刻$t$，已经ready的group集合为$\mathcal{G}_{ready}(t)=\{g \in \mathcal{G}(t): c_g  \geq K\}$，对于Trainer而言，当前还差$D(t)=B - |\mathcal{G}_{ready}(t)|$个group才能开启一次训练；对于任意候选group集合，$S\subseteq\mathcal{G(t)}\setminus \mathcal{G}_{ready}(t), \quad |S| = D(t)$，定义$\Phi_t(S)$为在当前时间点 t 的系统状态下，使集合 S 中所有 group 都 ready 所需的最短时间跨度：$\Phi_t(S)=\min_{\pi}\left\{\Delta \ge 0:\forall g \in S,\,\sum_{r \in \mathcal{U}_g(t)}\mathbf{1}\left[C_r^\pi \le t+\Delta\right]\ge K-c_g(t)\right\}$，因此critical group的精确定义应该为，在时刻t，能够获得 $D(t)$个group ready的最小时间所对应的$S$集合，也即$\mathcal{C}^*(t)\in\arg\min_{S \subseteq \mathcal{G}(t)\setminus \mathcal{G}_{ready}(t),\,|S|=D(t)}\Phi_t(S)$；
然而，解决这个问题本身是个NP-hard问题，我们定义一个决策版本，给定当前状态与deadline $\Delta$，问
$\exists S,\ |S|=D(t) \quad \text{s.t.} \quad \Phi_t(S) \le \Delta?$，也即是否存在$D(t)$个group，可以在$\Delta$时间内完成，这个状态包含，目前每个Sample Group内单个Sample的实际请求长度，目前正处在的状态，kvcache的占用压力等等，我们首先考虑极简的一种case来尝试证明该问题的Np-hard特性，考虑目前每个Sample都处在初始状态，且只有single\_turn的交互，设Group大小为1，GPU总共可以同时运行的bsz=2，此时总共有n条请求 $a_1$到$a_n$，且$\sum_j a_j = 2T$，此时，设$\Delta = T$，也即求解问题，能否把$S$拆成两个集合，使得满足
$\sum_{j\in A} a_j = T$ 与 $\sum_{j\notin A} a_j = T$，这是一个经典的two-machine makespan scheduling的NPC问题，如果该partiton问题有解，那么$\Phi_t(S) \le T$有解，反之，若$\Phi_t(S) \le T$有解，则两个GPU slot最多能够处理2T的请求总时间，且必须同时被填满，因此，该问题的解实际上对应partition的问题的一个解，因此，该问题被证明为是NP-hard问题，尤其当引入了目前系统所属的状态以及multi-turn后，该问题的求解复杂度将更加难以接受，尤其当面对系统的状态可能以非常快的频率更新时，精确的解法无解；
因此，我们转而考虑，是否存在近似解，能够很好的表达和解决该NP-hard问题，我们转而期望表达每个group剩余的完成成本，选择完成成本最小的$D_t$个group，也即找到$\text{priority}(g) \approx -\text{estimated closure cost}(g)$；
为了能够很好的表达每个group实际的完成成本，我们提出了启发式的算法来表达上面NP-hard的问题，这个启发式的算法核心基于两点observation：
* Observation 1：为了保证系统的吞吐，我们通过action-level agentloop，使得rollout worker能够始终在任意时刻有充足的bsz个sample可供推理，也即，rollout worker会在HBM存储允许情况下，始终维护理论最大的 $bsz^{'} , (bsz^{'} \leq bsz)$，因此，在这种setting下，我们能够假设，每条sample的per token latency也即TPOT可以近似为相同；
* Observation 2：就像Heddle所做的那样，维护一个0.5B的小模型在线fine-tuning，由于一条sample输出长度是否长，往往根据第一个turn的plan就能大致估计，且能够较为精准的随着step轮数的增长，预测出每条sample实际的请求可能长度，通过长度来确定执行的优先级；
  * [ ] 收集历史离线数据，输入为 context，输出为predict_length，用0.6B的小model离线fine-tune一个预测模型；
因此我们采用如下的启发式算法，我们通过每个Group的剩余Sample的最长预估剩余推理长度，近似等同于该group的优先级，在实际执行时，根据每个turn的context变化，动态更新group优先级；
## Priority-aware Scheduler
### Priority-aware Scheduling Algorithm
将整个rollouter Scheduler的调度算法讲清楚，给出算法伪代码，讲明目前的priority主要体现在gpu slot/env resources以及kvcache usage上；
### Reference-based KVCache Management
具体的kvcache设计如下：
由于在decode阶段引入layerwise的kvcache load/store，显然会严重损害decode侧的吞吐，因而，无法采用上面的在HBM中只驻留部分层的策略，来期望维持bsz同时最大化吞吐；
分析目前引入priority-based scheduler后的问题
KVCache trashing
- Rollout engine对实际驻留在HBM中的非活跃KVCache无感知，当新请求到达时，由于不存在充足的可用kvcache，需要evict HBM中的KVCache，此时应该对KVCache有状态设置，high_ref与low_ref，high_ref>0代表高优Token cache，low_ref>0代表低优token cache，high_ref=0 && low_ref=0代表可被直接换出cache；在需要evict请求时，优先evict无用cache，否则evict低优先级cache，仅当请求为高优先级请求时才允许将高优先cache也evict到HBM中，evict时，采用按需evict策略，从radix tree的叶子结点往上evict，一旦到达所需cache page，则立刻停止evict；
- 若将所有可用的kvcache都evict到DRAM中，仍然无法满足该请求的所需的kvcache，则考虑，高优先级请求停止某个低优先级请求，直到满足存在可用kvcache；
- ~~若仍然还未满足，则观察目前已partial page存放的kvcache，通过离线profiling结果，判断当前请求改为partial page后，load/store开销小于TPOT，则作为partial page后，load/store开销小于TPOT，则作为partial page存放在HBM中，将其加入bsz中；(待定)~~
![[Pasted image 20260426125654.png]]
### Runtime Trajectory re-dispatching
