我会先把问题拆成两层：单个 sandbox 的最优 CPU/内存配置，以及节点/集群层如何避免碎片化；并先确认 e2b、MicroVM、调度器与 guest kernel 的能力边界，再判断哪些能直接落地、哪些需要修改 VMM 或内核。

接下来我会把方案分成两类：先用现成能力（如 cgroup v2、PSI、内存热插拔）做控制面，再评估是否要改 VMM/guest 去补齐在线调 vCPU。

现在结论更明确了：关键不是继续细化固定规格，而是把 sandbox 抽象成可伸缩、可迁移、可休眠的资源对象。内存以 virtio-mem 为主做真弹性、balloon 只做回收提示；CPU 则走“在线核数/配额/放置”联合控制，并配合快照、共享池和重排来压实碎片。

Thought for 25m 17s

可以，而且我认为**能有效同时解决 1）单 sandbox 规格不匹配**和 **2）节点级碎片**；但前提是把问题从“给每个 sandbox 选一个静态 SKU”改成“把 sandbox 变成一个可弹性控制、可整理、可压缩的对象”。Firecracker 本身已经有 host 侧 oversubscribe、cgroup/cpuset/cpu quota、balloon、virtio-mem、snapshot/UFFD 这些原语；公开的 vCPU hot-plug 请求则仍停在 parked 状态，所以不要把方案建立在“等 Firecracker 现成支持运行时加 vCPU”上。E2B 这边已经把模板/快照做成一等公民，模板是从运行态快照生成的，后续可以从该状态快速拉起。

我的结论先说在前面：**最好的抽象不是“2c4g / 1c2g 这样的固定规格”，而是三段式资源模型**。

CPU 用三个量描述：`possible vCPU`、`online vCPU`、`entitled CPU`。

内存也用三个量描述：`base memory`、`hotplugged/requested memory`、`protected working set`。

这样 1）变成在线控制问题，2）变成 allocator/compactor 问题，而不是 SKU 选择问题。

### 1）单 sandbox 的 CPU / 内存怎么做到“按需最优”

**CPU 不要直接等价成“核数”，而要拆成三层。**

第一层是 **possible vCPU**，也就是 Firecracker 在启动前创建的 guest CPU 上限。Firecracker 的设计是每个 guest CPU 对应一个 vCPU 线程，所以这个上限不能无脑开很大；我会只保留少量 ceiling class，比如 2 和 4。好处是：E2B 模板缓存允许不同 CPU/RAM 变体复用共同层，维护少量 ceiling class 的工程成本是可控的。

第二层是 **online vCPU**，这一步其实可以**不靠 Firecracker vCPU hotplug**，而靠 **guest 内核自己的 CPU hotplug** 来做。Linux 文档明确支持 `maxcpus=n` 只让一部分 CPU 在启动时 online，并且可以通过 `/sys/devices/system/cpu/cpuN/online` 动态在线/离线 CPU。也就是说，你可以让 microVM 一开始就带着 2 个 possible CPU，但默认只 online 1 个；当判断这个 sandbox 的确能从并行中获益时，再把第二个 CPU online。这样 guest 里的线程池、`nproc`、`/proc/cpuinfo`、很多运行时的并行度判断都会看到真实的在线 CPU 数，而不是“表面 2 核、实际只给 1 核 quota”的假象。

第三层是 **entitled CPU**，也就是 host 真正给它多少算力。这里用 cgroup v2 就够了：`cpu.max` 做长期带宽，`cpu.max.burst` 给短时突发，`cpu.uclamp.min/max` 区分“串行但延迟敏感”和“真正可并行”的任务；只有极少数 tail-latency 敏感实例，再进 `cpuset.cpus.partition=isolated` 的专用 burst 池。内核文档也明确写了 `uclamp` 是给 scheduler/performance point 的 hint，而且在 `schedutil` 下会影响频率选择。所以对“1c 和 2c 交互时间没差异”的任务，我不会马上给第二核，而是优先保留 1 个 online CPU，再提高 `uclamp`；只有当它确实表现出并行收益时，才 online 第二个 CPU 并上调 `cpu.max`。这一步很关键，因为它直接把“多核没收益”的那类任务，从“多给一核”转成“单核跑快一点”。

内存也同理，不要再是静态 4GiB。Firecracker 现在已经支持 `virtio-mem`：先在启动前配置一个可 hotplug 的总区间，运行时再通过 `requested_size_mib` 增减 guest 可用内存，而且这个区间本身就是按 block/slot 管理的。Firecracker 文档对 guest kernel 的最低要求是 x86_64 5.16、aarch64 5.18；E2B sandboxes 当前跑的是 6.1 LTS kernel，所以这条路在能力上是通的。我的做法会是：给每个 sandbox 一个较小的 base mem，再配一个按 128MiB 一类 slot 增长的 hotplug region。

然后再叠加 **balloon + free-page reporting + memcg working-set protection**。Firecracker 的 balloon 可以在运行中改 target，free-page reporting 会把 guest 已知空闲的页范围 `MADV_DONTNEED` 掉，从而降低 RSS。host 侧则用 `memory.low` 只保护热工作集，用 `memory.high` 施加回收/节流而不直接 OOM，用 `memory.max` 做硬上限，必要时用 `memory.reclaim` 主动回收。热工作集估计我会用 DAMON 或 guest agent 来做，而不会把整个上限都当“保留内存”；因为内核文档也明确提醒，`memory.low` / `memory.min` 保护过度会带来持续 OOM 风险。

顺带一提，**弹性池里我不会开 hugepages**。Firecracker 官方文档写得很直接：传统 balloon 以 4K 粒度报告页，不能回收 hugepage backing 并降低 RSS；同时 dirty page tracking 又会抵消 hugepage 的收益。所以 hugepages 只适合另一类“静态、长寿命、CPU-bound”的 VM，不适合你这种要做回收、快照、密度压榨的 agent sandbox 池。

### 2）节点级碎片怎么解

- *CPU 碎片其实是最好消掉的。**只要大多数 sandbox 不要求 exclusive core，而是吃共享 CPU 池里的 `cpu.max`/burst 带宽，那么 CPU 就从“整数核装箱”变成“带宽装箱”。这时 0.7c、1.2c、1.8c 都可以 pack 到同一节点，整数 core fragmentation 基本消失。exclusive core 只保留给非常少的 tail-sensitive sandbox，并放到 `cpuset.cpus.partition=isolated` 的小池子里。

**内存碎片要靠“槽位化 + 工作集化 + compaction”一起解。**

第一，槽位化：`virtio-mem` 天然就是 block/slot 模型，我会把额外内存做成固定槽位分配，像 buddy allocator 那样管理，而不是给每个 sandbox 一个任意大小的静态 limit。

第二，工作集化：只保护 hot set，不保护 cap。

第三，compaction：空闲 sandbox 直接 pause。E2B 的 pause 会保存文件系统和内存状态，支持超时后自动 pause；这对 agent sandbox 特别有用，因为很多实例的生命周期是“短 burst + 长 idle”。但它的 pause/resume 不是零成本，文档给的量级是大约 4 秒 / GiB 的 pause 和约 1 秒的 resume，所以这招适合冷实例整理，不适合热路径上的频繁抖动。

NUMA 上我会更保守：**placement 时一次性定好 `cpuset.mems`，后面尽量迁整个 sandbox，而不是反复改 mems**。因为内核文档明确说改 `cpuset.mems` 会触发内存迁移，有成本，而且可能迁不干净。所以节点整理更像“搬箱子”，不是“在箱子里抖内存页”。

如果你还想再榨一层密度，可以做**模板感知的去重**。KSM 只会作用在 `MADV_MERGEABLE` 的匿名页上，内核文档还给了 `general_profit` / `ksm_process_profit` 等指标来判断到底值不值得开。所以正确做法不是全局盲开 KSM，而是对“同模板、同 trust domain、读多写少”的那部分匿名页选择性 `MADV_MERGEABLE`。这对 E2B 这种模板/快照强依赖的场景很适合。

### 真正把碎片压到最低：做 snapshot / UFFD compaction

再往上，就是**快照驱动的整理与冷迁移**。Firecracker 支持把 snapshot 在另一个 Firecracker process 中恢复；用 UFFD 模式时，页错误可以交给用户态 handler，因此你完全可以做“shared base + per-sandbox COW overlay”或者至少做“按需分页恢复 + 冷实例快速搬迁”。从“能不能做”这个角度，我认为是能做的，而且这一步对密度提升会非常大。

但这一层要非常清楚边界：Firecracker 文档也写了，snapshot load 对软硬件同构有要求；snapshot/restore 会 reset vsock，网络连续性要自己处理；更重要的是，同一个 snapshot 多次恢复会复制随机数、token、用户态唯一状态。Firecracker 的 VMGenID 能让 Linux 在 snapshot resume 后重种内核 PRNG，但**用户态**缓存的随机值、唯一 ID、token 还是会复制。所以 shared-base/COW 这类技术我只会用在模板基态、暂停/冷实例、或者明确无连接状态的阶段；不会对正在跑交互连接的热实例直接这么做。还有，UFFD handler 如果崩了，恢复过程会卡住，这个 handler 必须作为一等控制面组件来做健康检查和回收。

### 我会怎么落地

我会按下面的顺序做，而不是一步到位上所有黑科技：

1. 先把固定 2c4g 改成少量 **possible-capacity class**，例如 2-vCPU ceiling 和 4-vCPU ceiling；E2B 的模板缓存足够支撑这点复杂度。
2. guest 默认只 online 1 个 CPU，host 用 `cpu.max` + `cpu.max.burst` + `uclamp` 做快速控制；判定有并行收益时再 online 第二个 CPU。
3. 内存改成 base + `virtio-mem` 槽位，配 balloon/free-page reporting，host 侧用 `memory.low/high/max/reclaim` 做 working-set 控制。
4. 指标不要依赖 E2B 5 秒流和 Firecracker 60 秒流；那是观测，不是控制。快速环路直接看 cgroup PSI 和 host 侧实时统计。
5. 对 idle 实例 aggressively auto-pause，把长尾占用从 RAM/CPU 挪到 snapshot/storage；再慢慢做同模板去重和冷迁移 compaction。

**最终判断**：

这件事不是“能不能”，而是“该不该用静态规格去做”。如果你继续用固定 2c4g，再聪明的调度都只是在一个很差的抽象上打补丁。真正正确的解法是：

- CPU：`possible` / `online` / `entitled`
- Memory：`base` / `requested` / `protected hotset`
- Node：共享池 + 少量 isolated burst 池
- Cluster：周期性 compaction + pause + cold migration

这样 1）会从“选 SKU”变成“在线估计边际收益”，2）会从“静态碎片”变成“可整理碎片”。在给定 P99 交互延迟和安全余量的前提下，这套方法能把密度推得很高；要追求数学意义上的绝对最优在线密度，则不现实，因为未来负载未知，而且迁移/去重/回收本身都有成本。

如果你愿意，我下一步可以直接把它整理成一个**可实现的系统设计文档**：包括控制环、状态机、数据结构、placement 策略和失败回退路径。