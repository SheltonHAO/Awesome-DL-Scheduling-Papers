# Online Orchestration of Elastic Inference Jobs on Heterogeneous GPU Clusters

## 问题定义与数学模型（工作稿）

> 本文档用于固定第一版论文的问题边界、核心符号和建模假设。
>
> 具体的 RL/DRL 结构、组合优化算法和实验方案将在后续文献调研与数据分析后确定。

---

## 1. 一句话问题定义

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排问题。任务随时间动态到达，并提交完成期限、带噪声的预估卡时、兼容 GPU 卡型对应的固定 Gang 配置和最大副本数。上层编排器周期性决定任务准入以及每个任务在不同卡型上的副本数量，并确保底层 K8s/Volcano 调度器能够完成 Gang placement。任务处理速度和完成时间在启动前未知，运行后根据实际进度逐步揭露；未来任务到达和可用容量同样未知。研究目标按任务离开系统后确定的相对价值奖励交付、惩罚接纳后的 SLA 违约，并采用 SLA-first learning + JCT-aware optimization 降低平均与尾部 Job Completion Time（JCT）。

---

## 2. 研究动机与系统边界

### 2.1 研究动机

异构 GPU 集群中的空闲算力具有明显的时间变化：

- 在线业务释放的资源可能在之后被重新收回；
- 离线资源预算也会随集群状态变化；
- 京东弹性计算框架 Aether 支持任务在运行过程中增加、减少或暂停副本，使弹性离线推理任务能够充分利用这些动态容量。

Aether 提供动态调整副本的执行能力；本文研究其上层的在线决策，即何时接纳任务，以及每个决策时点应为各任务分配多少副本。

因此，调度器需要持续回答：

1. 当前到达的任务是否值得接纳；
2. 已接纳任务应当分配多少副本；
3. 不同任务的副本应当使用哪些兼容 GPU 卡型；
4. 当前资源应当用于加速已有任务，还是为未来未知任务保留。

### 2.2 第一版论文的抽象层级

论文采用两层调度结构：

1. **上层在线编排器**
   - 决定任务准入；
   - 决定任务在各 GPU 卡型上的副本数量；
   - 只输出当前节点状态下能够完成 Gang placement 的方案。

2. **底层弹性执行与调度系统**
   - 由 Aether 弹性框架和现有 K8s/Volcano 调度器共同承担；
   - Aether 根据上层决策执行副本增加、减少或暂停；
   - K8s/Volcano 将每个副本的 Gang Pods 放置到具体节点；
   - 返回实际 placement 或不可行结果；
   - 执行 Pod 启停及资源生命周期管理。

论文不显式区分在线资源域和离线资源域，也不建模公司内部的资源回收与驱逐机制。所有容量变化统一视为外生环境变化，并在每个决策时点更新为当前可用集群状态。

论文不显式建模 setup 时间、setup 状态或 setup 成本。镜像拉取、Pod 启动等影响由实际执行进度和 JCT 自然反映，但不进入核心决策变量与约束。

上层不把运行中副本迁移作为决策。副本数量或卡型配置发生变化时，通过保留、删除和新增副本实现；具体 Pod 生命周期与节点操作由底层调度器处理。

---

## 3. 任务与执行层级

### 3.1 Task–Replica–Pod 层级

系统执行层级为：

```math
\text{Task}
\longrightarrow
\text{Replicas}
\longrightarrow
\text{Gang Pods}
\longrightarrow
\text{K8s/Volcano}
\longrightarrow
\text{Nodes/GPUs}.
```

- **Task**：用户提交的完整 batch 推理任务；
- **Replica**：包含一份完整模型参数、能够独立处理 batch 分片的执行副本；
- **Pod**：一个副本经过固定切分后形成的部署单元；
- **Node/GPU**：由底层调度器选择的物理执行资源。

模型切分方式由相邻的训练/推理框架预先确定，不是上层编排器的决策。

### 3.2 Gang scheduling 语义

对于任务 $`j`$ 和兼容 GPU 卡型 $`k`$：

- 一个副本由 $`p_{jk}`$ 个 Pod 构成；
- 每个 Pod 需要 $`g_{jk}`$ 张类型 $`k`$ 的 GPU；
- 一个 Pod 必须完整放置在一个节点上；
- 同一副本的不同 Pod 可以位于相同或不同节点；
- 同一副本的全部 Pod 必须使用同一种 GPU 卡型；
- 全部 Pod 必须能够同时启动，副本才可以运行；
- 不同副本可以同时使用不同的兼容 GPU 卡型。

定义一个副本在卡型 $`k`$ 上消耗的总 GPU 数为

```math
q_{jk}=p_{jk}g_{jk}.
```

$`q_{jk}`$ 用于上层容量估算，而完整 Gang shape

```math
\Gamma_{jk}=(p_{jk},g_{jk})
```

会传递给底层 K8s/Volcano 调度器进行节点级可行性检查。仅满足总 GPU 数量并不自动意味着 Gang placement 可行。

少量任务的单 Pod GPU 需求可以为 $`0.5`$，大多数需求为一张或多张整数 GPU。Pod 始终是不可拆分的放置单元。

---

## 4. 任务提交与可行配置

任务 $`j`$ 的提交信息为

```math
\mathcal J_j=
\left(
a_j,\,
d_j,\,
\widetilde H_j,\,
\mathcal K_j,\,
\overline r_j,\,
\{\Gamma_{jk}\}_{k\in\mathcal K_j}
\right),
```

其中：

- $`a_j`$：实际提交时刻；
- $`d_j`$：任务完成期限，即业务 SLA；
- $`\widetilde H_j`$：用户申报的预计 GPU 卡时，是准入时可见的带噪声请求信息；
- $`\mathcal K_j\subseteq\mathcal K`$：任务可使用的 GPU 卡型集合；
- $`\overline r_j`$：任务允许同时运行的最大副本总数；
- $`\Gamma_{jk}=(p_{jk},g_{jk})`$：卡型 $`k`$ 下一个副本的固定 Gang 配置。

任务不提交最小副本数，统一采用

```math
\underline r_j=0.
```

因此，已接纳任务可以在内部队列中等待，也可以在运行过程中缩容至零，并在后续决策时点重新获得副本。

$`\widetilde H_j`$ 仅用于准入时估计资源需求和履约风险，不是精确工作量、硬资源预算或到期查杀阈值。任务的实际完成时间和资源占用只有在执行过程中才能逐步观察；用于终局奖励的相对价值也不在准入时显式预测，而是在任务离开系统后统一结算。

### 4.1 配置语义

对每个 $`k\in\mathcal K_j`$，定义可行单副本配置

```math
c_{jk}=(k,\Gamma_{jk}).
```

配置 $`c_{jk}`$ 同时给出 GPU 卡型和固定 Gang shape，表示一个能够独立处理 batch 分片的完整副本。卡型差异直接体现在副本的资源形状、当前可用容量和 Gang placement 可行性上。

任务可以同时运行多个副本，不同副本可以选择不同的兼容配置。每个副本产生的实际处理进度都由执行系统以统一的 batch 进度口径汇总。

### 4.2 未知处理速度与完成时间

上层编排器不假设任务提交时已知准确完成时间或固定处理速度。系统可以根据 $`\widetilde H_j`$、任务属性和历史数据，为尚未运行的配置提供初始吞吐估计；任务运行后，执行系统根据实际进度产生新的吞吐观测。论文直接使用外部估计器提供的当前吞吐估计，不研究估计器本身。

---

## 5. 在线决策过程

### 5.1 决策时点

时间被离散为

```math
\mathcal T=\{0,1,\ldots,T\},
```

相邻决策时点之间的实际间隔为 $`\Delta`$。第一版取 $`\Delta=10`$ 分钟。

实际到达时刻为 $`a_j`$ 的任务，在最近的下一个决策时点参加准入：

```math
\tau_j
=
\min\{t\in\mathcal T:t\Delta\ge a_j\}.
```

固定周期决策是第一版设定。事件触发或“固定周期 + 事件触发”的混合机制属于后续扩展。

### 5.2 决策时点可观测信息

在时点 $`t`$，上层编排器观察：

1. **新到达任务**：本周期首次参加准入、尚未作出接纳或拒绝决策；
2. **排队任务**：已经接纳但当前副本总数为零；
3. **运行任务**：已经接纳且当前至少运行一个副本；
4. 新任务申报的带噪声卡时 $`\widetilde H_j`$；
5. 每个已接纳任务的剩余 batch 工作比例；
6. 每个已接纳任务最近一次可用的吞吐估计及其观测时间；
7. 当前各任务在各卡型上的副本数量；
8. 当前节点、GPU、Pod 和已有 placement 状态；
9. 当前由集群环境提供的可用 GPU 容量。

调度器不知道：

1. 未来任务到达及其参数；
2. 未来集群容量变化；
3. 未来资源回收；
4. 未来真实处理速度；
5. 新任务最终结算的相对价值 $`v_j`$；
6. 下一次有效吞吐观测何时产生。

历史数据可以用于训练 RL/DRL 模型；实际在线决策只能使用当前和过去已经观测到的信息，不能查看尚未发生的任务到达、容量变化或执行结果。

### 5.3 两类上层决策

上层编排器在每个决策时点进行：

1. **Admission control**
   - 对本周期新到达任务作出一次性接纳或拒绝决策；
   - 已接纳任务进入内部任务集合；
   - 当前请求被拒绝后永久离开；
   - 用户修改 DDL、兼容配置或最大副本数后重新提交时，视为新任务。

2. **Heterogeneous replica orchestration**
   - 决定每个任务在每种兼容卡型上的副本数；
   - 允许增加、减少、保持或清零副本；
   - 不同副本可以同时使用不同卡型；
   - 总副本数不超过用户提交上限；
   - 输出方案必须能够由底层调度器完成 Gang placement。

任务一旦被接纳，其准入状态和 SLA 承诺便不再撤销；调度器不能在后续将其改为“拒绝”以逃避违约责任。

在第一版模型中，任务接纳后可以滚动修改的上层决策只有各兼容卡型上的副本数量，包括增加、减少、清零或从零恢复。准入状态保持不变，具体 Pod placement 由底层系统根据新的副本方案重新完成。

每轮只下发并执行当前周期的副本配置。为估计任务能否在 DDL 前完成，编排器可以计算后续周期的假设性副本方案；这些尚未执行的方案不会提前锁定 GPU，并会在下一决策时点根据最新状态重新计算。

---

## 6. 最近吞吐观测与完成时间估计

### 6.1 任务进度

令 $`W_{jt}\in[0,1]`$ 表示决策时点 $`t`$ 任务 $`j`$ 的剩余 batch 工作比例。任务提交时：

```math
W_{j,\tau_j}=1.
```

任务完成状态由 batch 数据是否处理完决定。$`W_{jt}`$ 由执行系统维护，不由上层编排器预测或控制。

### 6.2 最近一次可用吞吐

处理速度不要求在每个决策时点更新。对任务 $`j`$ 的配置 $`c_{jk}`$，令 $`\mathcal O_{jk}(t)`$ 表示截至时点 $`t`$ 已产生有效吞吐观测的时点集合。若该集合非空，定义最近观测时点：

```math
\ell_{jk}(t)=\max \mathcal O_{jk}(t).
```

调度器使用的单副本吞吐估计为

```math
\overline\mu_{jkt}
=
\begin{cases}
\mu^{\mathrm{obs}}_{jk,\ell_{jk}(t)},
& \mathcal O_{jk}(t)\neq\varnothing,\\
\mu^{\mathrm{prior}}_{jk},
& \mathcal O_{jk}(t)=\varnothing.
\end{cases}
```

其中 $`\mu^{\mathrm{prior}}_{jk}`$ 是系统为尚未运行的配置提供的初始估计，$`\mu^{\mathrm{obs}}_{jk,\ell_{jk}(t)}`$ 是最近一次有效观测。若当前周期没有新观测，则继续沿用 $`\overline\mu_{jkt}`$。所有配置的吞吐都使用“单位时间完成的 batch 工作比例”这一统一口径。

吞吐估计模块被视为系统提供的外部能力。第一版论文不研究初始估计器、吞吐预测器、P95/P99 完成时间预测或概率校准。

### 6.3 规划中的完成时间估计

给定从时点 $`t`$ 开始的候选副本计划 $`\{n_{jk\tau}\}`$，调度器使用最近估计预测剩余工作：

```math
\widehat W_{j,\tau+1}
=
\left[
\widehat W_{j\tau}
-
\Delta
\sum_{k\in\mathcal K_j}
\overline\mu_{jk\tau}n_{jk\tau}
\right]^+,
\qquad
\widehat W_{jt}=W_{jt}.
```

对应的预计完成时点为

```math
\widehat C_j(t;\mathbf n)
=
\Delta
\inf
\left\{
\tau\ge t:
\widehat W_{j\tau}=0
\right\}.
```

在没有新吞吐观测时，计划期内使用最近一次估计。该未来副本序列只是用于估算完成时间的临时方案：上层只下发当前周期的 $`n_{jkt}`$，下一决策时点再根据最新进度、吞吐与容量重新计算其余周期。

### 6.4 实际进度更新

实际状态不使用一个假设已知的“真实每期吞吐”进行更新。令

```math
\delta^{\mathrm{obs}}_{jt}\in[0,W_{jt}]
```

表示执行系统在区间 $`[t\Delta,(t+1)\Delta)`$ 内实际完成的 batch 工作比例，则：

```math
W_{j,t+1}
=
W_{jt}
-
\delta^{\mathrm{obs}}_{jt}.
```

$`\delta^{\mathrm{obs}}_{jt}`$ 在区间结束后由实际任务进度得到，是环境反馈而非决策变量。资源回收、启动延迟、运行波动或部分周期执行都会自然反映在该实际进度中，无需分别建模。

---

## 7. Gang-feasible 上层抽象

### 7.1 为什么不能只使用总 GPU 容量

若仅满足

```math
\sum_j q_{jk}n_{jkt}\le C_{kt},
```

仍可能因节点碎片而无法部署。例如，一个副本要求四个 Pod、每个 Pod 占用四张 GPU，总需求虽然是 16 张卡，但可用卡分散在不满足单 Pod 需求的节点上时，Gang 仍无法启动。

因此，上层决策必须同时满足聚合容量和实际 Gang placement 可行性。

### 7.2 下层 Gang placement

上层决定每个任务在各卡型上运行多少个副本；下层决定这些副本产生的 Pod 分别放到哪些节点。

时点 $`t`$ 的下层输入包括：

- 当前节点状态 $`s_t^{\mathrm{node}}`$：每个节点的 GPU 卡型、空闲卡数和已有 Pod；
- 上层给出的候选副本方案 $`\widetilde{\mathbf n}_t=(\widetilde n_{jkt})_{j,k}`$；
- 每个任务—卡型配置的 Gang shape $`\Gamma_{jk}=(p_{jk},g_{jk})`$。

给定 $`\widetilde{\mathbf n}_t`$ 后，下层不再决定副本数量，而是决定保留或停止哪些现有副本，以及新增副本的每个 Pod 放到哪个节点。现有 Pod 不迁移；缩容按完整副本释放，扩容按完整 Gang 副本放置。

候选方案中的一个副本会展开为 $`p_{jk}`$ 个 Pod，每个 Pod 占用 $`g_{jk}`$ 张类型 $`k`$ 的 GPU。下层由此求解一个带 Gang 约束的装箱问题：

- 节点是箱子，空闲 GPU 数是箱子容量；
- Pod 是不可拆分的物品，必须完整放在一个兼容节点；
- 同一副本的全部 Pod 必须全部放置成功并能够同时启动，否则该副本不能启动；
- 不同副本或任务的 Pod 可以共享节点；
- 新放置与已有 Pod 的总占用不能超过任何节点的容量。

下层输出为：

```math
\mathrm{GangPlace}
\left(
s_t^{\mathrm{node}},
\widetilde{\mathbf n}_t
\right)
\longrightarrow
\begin{cases}
\pi_t, & \text{all induced Pods can be placed},\\
\mathrm{infeasible}, & \text{otherwise},
\end{cases}
```

其中 $`\pi_t`$ 是具体的 Pod-to-node placement。上层可行副本集合定义为

```math
\mathcal F_t(s_t^{\mathrm{node}})
=
\left\{
\widetilde{\mathbf n}_t:
\mathrm{GangPlace}
\left(
s_t^{\mathrm{node}},
\widetilde{\mathbf n}_t
\right)
\text{ returns a placement}
\right\}.
```

因此，$`\mathcal F_t(s_t^{\mathrm{node}})`$ 是一组可装入当前节点的副本方案；最终选中的 $`\mathbf n_t`$ 只是其中一个元素。约束

```math
\mathbf n_t\in\mathcal F_t(s_t^{\mathrm{node}})
```

表示上层选出的副本数方案存在一个可执行的 Gang placement，而不是 $`\mathbf n_t`$ 与 $`\mathcal F_t`$ 相等。$`\pi_t`$ 由底层 Gang packer 或 K8s/Volcano 生成，不作为 RL 动作，也不在核心模型中显式建立 Pod-to-node 二元变量。

例如，两台 HC 节点各剩 4 张卡，且 $`\Gamma_{j,\mathrm{HC}}=(2,2)`$。一个副本产生两个各占 2 张卡的 Pod。两个副本可以分别装入两台节点，因此方案 $`n_{j,\mathrm{HC},t}=2`$ 可行；三个副本共需 12 张卡，因此不可行。

> **待讨论：下层是否考虑时间维度？** 当前定义只检查本周期的节点快照。后续可以研究是否利用运行中任务的预计完成时间和预计资源释放时间，求解 rolling-horizon 的时空装箱问题。但由于完成时间估计存在误差，未来 placement 是否应提前规划、规划到多长时间以及是否形成资源预留，仍属于算法设计问题。

---

## 8. 紧凑数学模型

### 8.1 集合、参数与状态

- $`\mathcal J`$：研究周期内的全部任务；
- $`\mathcal J_t^{\mathrm{new}}`$：时点 $`t`$ 首次参加准入的新任务；
- $`\mathcal J_t^{\mathrm{acc}}`$：时点 $`t`$ 已接纳但尚未完成的任务；
- $`\mathcal K`$：GPU 卡型集合；
- $`\mathcal K_j`$：任务 $`j`$ 的兼容卡型集合；
- $`\mathcal T`$：离散决策时点集合；
- $`C_{kt}`$：当前可供弹性推理任务使用的卡型 $`k`$ 聚合 GPU 容量；
- $`s_t^{\mathrm{node}}`$：底层节点容量和已有 placement 状态；
- $`\widetilde H_j`$：准入时可见的用户申报预计 GPU 卡时；
- $`v_j\ge 0`$：已接纳任务离开系统时结算的终局相对价值；拒绝任务约定为 $`v_j=0`$；
- $`W_{jt}`$：任务剩余 batch 工作比例；
- $`\overline\mu_{jkt}`$：任务 $`j`$ 的配置 $`c_{jk}`$ 最近一次可用的单副本吞吐估计。

未来的 $`C_{kt}`$、$`s_t^{\mathrm{node}}`$、任务到达和吞吐观测均未知。

系统维护一个由历史完成任务组成的参考池，并根据已接纳任务离开系统时观测到的实际 GPU 卡时、batch 等信息，将其映射为分位数归一化的终局相对价值 $`v_j`$。

### 8.2 决策变量

#### 准入变量

```math
z_j\in\{0,1\},
```

其中 $`z_j=1`$ 表示任务 $`j`$ 在其首次决策点被接纳。该变量一旦确定便不再修改。

#### 卡型—副本数量

```math
n_{jkt}\in\mathbb Z_+,
```

表示时点 $`t`$，上层编排器要求任务 $`j`$ 在卡型 $`k`$ 上运行的副本数。

与原来的 Pod-to-node 变量相比，核心动作只具有任务、卡型和时间三个维度。具体副本身份和节点位置由底层调度器管理。

### 8.3 核心约束

#### 准入与兼容性

任务不能使用不兼容卡型，已经完成的任务不再获得副本：

```math
n_{jkt}=0,
\qquad
k\notin\mathcal K_j
\ \text{or}\
W_{jt}=0.
```

#### 最大副本数

```math
0
\le
\sum_{k\in\mathcal K_j}n_{jkt}
\le
\overline r_jz_j,
\qquad
\forall j,t.
```

下界为零，因此已接纳任务可以排队或暂停。

#### 聚合 GPU 容量

```math
\sum_{j:k\in\mathcal K_j}
q_{jk}n_{jkt}
\le
C_{kt},
\qquad
\forall k,t.
```

该约束用于快速筛除明显不可行的候选方案，但不能替代节点级 Gang 检查。

#### Gang placement 可行性

```math
\mathbf n_t
\in
\mathcal F_t(s_t^{\mathrm{node}}),
\qquad
\forall t.
```

这是连接上层优化器与底层 K8s/Volcano 调度器的关键约束。

#### 估计进度

在线规划使用：

```math
\widehat W_{j,t+1}
=
\left[
W_{jt}
-
\Delta
\sum_{k\in\mathcal K_j}
\overline\mu_{jkt}n_{jkt}
\right]^+.
```

下一决策时点再用实际观测 $`\delta^{\mathrm{obs}}_{jt}`$ 修正状态。

### 8.4 完成、DDL 与 JCT

任务实际完成时刻为

```math
C_j
=
\Delta
\inf\{t\in\mathcal T:W_{jt}=0\}.
```

若任务在研究周期内未完成，则令 $`C_j=+\infty`$。

任务的 JCT 从用户实际提交时刻开始计算：

```math
\mathrm{JCT}_j=C_j-a_j.
```

定义请求 $`j`$ 的按时完成指标：

```math
S_j=
\begin{cases}
1, & z_j=1\text{ and }C_j\le d_j,\\
0, & \text{otherwise}.
\end{cases}
```

定义接纳后 SLA 违约指标：

```math
V_j=
\begin{cases}
1, & z_j=1\text{ and }C_j>d_j,\\
0, & \text{otherwise}.
\end{cases}
```

若任务在 $`d_j`$ 时仍未完成，则从该时刻起记为 $`V_j=1`$，且一次 SLA 违约只结算一次。DDL 是服务承诺的结算时点，不是任务失效或强制终止时点；逾期任务仍保留在 active set 中，直到完成、用户取消或研究周期结束。

为比较不同承诺窗口下的迟交程度，定义相对迟交比例：

```math
\rho_j
=
\frac{(C_j-d_j)^+}
{\max\{d_j-a_j,\Delta\}}.
```

对于有限的迟交，令交付价值随相对迟交递减：

```math
b(\rho_j)
=
\omega^{\rho_j/0.05},
\qquad
0<\omega<1.
```

$`\omega`$ 表示每增加相当于原始提交—DDL 窗口 $`5\%`$ 的迟交后，剩余交付价值所保留的比例。定义任务交付因子。若任务被拒绝或始终未完成，则

```math
B_j=0.
```

若任务被接纳且按时完成，则

```math
B_j=1.
```

若任务被接纳但迟交，则

```math
B_j=b(\rho_j).
```

因此，任何有限的迟交完成都严格优于永远不完成，但越晚交付，交付价值越低。由于未来容量、回收和处理速度未知，论文不声称对所有随机环境提供确定性零违约保证，而是在目标中显式惩罚接纳后的违约。

---

## 9. 优化目标

### 9.1 首要目标：SLA 服务效用

系统不额外设置人工业务优先级；任务间的终局价值差异统一由 $`v_j`$ 表示。任务 $`j`$ 的终局服务效用为

```math
U_j
=
v_j
\left(
B_j-\kappa V_j
\right),
\qquad
\kappa>0.
```

$`\kappa`$ 是全局统一的、无量纲的 SLA 违约惩罚系数。它表示 SLA 违约相对于按时交付价值有多重要，而不是任务优先级、GPU 价格或违约概率。将任务效用除以 $`v_j`$ 后，每单位相对价值的含义非常直接：

| 任务结果 | 每单位相对价值的效用 |
|---|---:|
| 准入时拒绝 | $`0`$ |
| 按时完成 | $`+1`$ |
| 迟交完成 | $`b(\rho_j)-\kappa`$ |
| 始终未完成 | $`-\kappa`$ |

例如，$`\kappa=2`$ 表示：一单位已接纳任务价值发生 SLA 违约所产生的惩罚，是该单位价值按时交付收益的两倍。所有任务使用相同的 $`\kappa`$；任务 $`j`$ 的总效用等于表中的单位效用乘以 $`v_j`$。

有限迟交比永远不完成多获得 $`v_jb(\rho_j)>0`$ 的交付价值，因此策略不会仅因任务已经错过 DDL 就将其永久饿死。策略 $`\pi`$ 的首要目标为

```math
\max_\pi
\mathbb E_\pi
\left[
\sum_{j\in\mathcal J}
U_j
\right].
```

下面用一个仅用于解释 $`\kappa`$ 的简化例子说明其对准入的影响。暂时忽略资源竞争和迟交后的剩余交付价值，设调度器根据准入时可见信息判断任务 $`j`$ 按时完成的可能性为 $`p_j`$。接纳该任务后，每单位相对价值的平均效用为

```math
p_j-\kappa(1-p_j).
```

其中，第一项是按时完成带来的价值，第二项是接纳后违反 SLA 的风险成本。该平均效用为正需要

```math
p_j
>
\frac{\kappa}{1+\kappa}.
```

因此，若 $`\kappa=1`$，按时完成的可能性需要高于 $`50\%`$；若 $`\kappa=4`$，则需要高于 $`80\%`$。$`p_j`$ 只是帮助理解的概念，不要求第一版系统显式训练或输出一个按时完成概率。实际算法基于准入时能够观察到的任务信息和系统状态，联合选择本周期的接纳任务子集，并通过终局反馈学习长期收益，同时考虑迟交价值、资源竞争、未来容量机会成本和 Gang 可行性。

论文不为 $`\kappa`$ 指定唯一数值，而是扫描多个取值，刻画按时服务价值与 SLA 违约率之间的 Pareto 前沿。$`\kappa`$ 越大，策略越重视避免已经作出的 SLA 承诺被违反；部署时可根据目标风险水平选择运行点。

### 9.2 次要目标：JCT

令

```math
\mathcal J^{\mathrm{comp}}
=
\{j\in\mathcal J:z_j=1,\ C_j<+\infty\}.
```

所有最终完成的已接纳任务，包括迟交任务，均进入 JCT 统计：

```math
\mathrm{MeanJCT}
=
\frac{1}{|\mathcal J^{\mathrm{comp}}|}
\sum_{j\in\mathcal J^{\mathrm{comp}}}
(C_j-a_j).
```

本文不把 JCT 奖励直接加入 RL 的终局效用，因为任意有限权重都会把严格的 SLA-first 目标改成普通加权和。学习模块优化长期服务效用或未来容量价值，内层组合优化器在服务效用相同或近似相同的候选动作中最小化预测 JCT。

令 $`\mathcal A_t(s_t)`$ 为当前 Gang-feasible 动作集合，$`F_t^{\mathrm{svc}}(a)`$ 为结合 learning 输出得到的当前服务目标，先求

```math
F_t^\star
=
\max_{a\in\mathcal A_t(s_t)}
F_t^{\mathrm{svc}}(a),
```

再求

```math
\min_{a\in\mathcal A_t(s_t)}
G_t^{\mathrm{JCT}}(a)
\quad
\text{s.t.}
\quad
F_t^{\mathrm{svc}}(a)
\ge
F_t^\star-\epsilon_F.
```

$`G_t^{\mathrm{JCT}}`$ 是基于当前进度与吞吐估计得到的预测完成时间代价。$`\epsilon_F=0`$ 表示只在服务目标完全相同的动作之间优化 JCT；较小的 $`\epsilon_F>0`$ 允许以近似无损的服务效用改善 JCT。第一版方法定位为 **SLA-first learning + JCT-aware optimization**，不声称训练了一个全局严格词典序 RL policy。TailJCT 可采用 P95、P99 或 CVaR。

### 9.3 评价指标与训练奖励

业务评价指标与 RL 训练奖励分开定义。任务数量口径至少报告：

```math
\mathrm{OnTimeJobRate}
=
\frac{\sum_jS_j}{|\mathcal J|},
\qquad
\mathrm{AdmissionRate}
=
\frac{\sum_jz_j}{|\mathcal J|},
```

以及

```math
\mathrm{ViolationRate}
=
\frac{\sum_jV_j}
{\max\{1,\sum_jz_j\}}.
```

同时报告终局价值口径：

```math
\mathrm{OnTimeValueShare}
=
\frac{\sum_{j:z_j=1}v_jS_j}
{\max\{\varepsilon,\sum_{j:z_j=1}v_j\}},
\qquad
\mathrm{ValueWeightedViolationRate}
=
\frac{\sum_{j:z_j=1}v_jV_j}
{\max\{\varepsilon,\sum_{j:z_j=1}v_j\}},
```

其中 $`\varepsilon>0`$ 只用于避免分母为零，拒绝任务的终局效用为零，无需为其事先估计 $`v_j`$。主实验扫描 $`\kappa`$，展示按时服务价值与 value-weighted SLA 违约率之间的 Pareto 前沿；同时按申报卡时分桶报告全部请求的 admission rate，并在已接纳任务中按终局价值分位报告按时完成率与违约率。

最终业务评价仍由服务效用、接纳后违约和 JCT 决定。为了缓解任务结束前奖励稀疏和长期信用分配困难，RL 可以使用基于 DDL laxity 的 potential-based reward shaping。

令 $`s_t`$ 汇总时点 $`t`$ 的任务进度、最近吞吐估计、当前副本、可用容量和节点状态。内层优化器基于 $`s_t`$ 生成一个确定的参考计划，并得到任务 $`j`$ 的预计完成时刻 $`\widehat C_j(s_t)`$。定义计划 laxity：

```math
L_{jt}
=
d_j-\widehat C_j(s_t).
```

$`L_{jt}>0`$ 表示预计提前完成，$`L_{jt}=0`$ 表示刚好满足 DDL，$`L_{jt}<0`$ 表示当前计划存在违约风险。为便于比较不同 DDL 长度的任务，定义归一化 laxity：

```math
\widetilde L_{jt}
=
\frac{L_{jt}}
{\max\{d_j-t\Delta,\Delta\}}.
```

状态 potential $`\Phi(s_t)`$ 汇总所有已接纳未完成任务的 SLA 风险。例如：

```math
\Phi(s_t)
=
-
\sum_{j\in\mathcal J_t^{\mathrm{acc}}}
\mathrm{softplus}
\left(
-\frac{\widetilde L_{jt}}{\sigma}
\right),
\qquad
\sigma>0.
```

laxity 越小，$`\Phi(s_t)`$ 越低。采用有限时域、无时间折扣的回报，即 $`\gamma=1`$，避免仅因大任务完成较晚而产生额外的小任务偏好。训练奖励采用

```math
r_t^{\mathrm{train}}
=
r_t^{\mathrm{service}}
+
\lambda_{\mathrm{shape}}
\left[
\Phi(s_{t+1})-\Phi(s_t)
\right],
```

其中 $`r_t^{\mathrm{service}}`$ 只累计本周期终局结算的 $`U_j`$。任务跨过 DDL 时只更新 $`V_j`$ 和风险状态，不提前结算终局效用；当任务完成、被用户取消或在评估期末统一结算时，系统根据其实际执行信息得到 $`v_j`$，并一次性结算 $`U_j`$。训练奖励不加入 JCT bonus；JCT 由内层优化器的第二阶段目标处理。完成或外部取消的任务从 active state 中移除，但仅到达 DDL 的未完成任务继续保留；episode 终止状态的 potential 设为零。Potential 差值为扩容、暂停和准入决策提供中间风险信号，但不替代最终服务效用。具体 potential 形式和 $`\lambda_{\mathrm{shape}}`$ 将通过消融实验验证。

### 9.4 解释性指标

以下指标用于解释策略行为，但不高于 SLA 与 JCT：

- GPU 资源占用率与有效利用率；
- 各卡型利用情况；
- 任务接纳率与拒绝率；
- 按申报卡时 $`\widetilde H_j`$ 分桶的接纳率，以及已接纳任务按 $`v_j`$ 分位统计的按时完成率与违约率；
- 申报卡时与实际 GPU 卡时的偏差分布及 prior 噪声敏感性；
- 迟交完成率、相对迟交 $`\rho_j`$ 和未完成率；
- 副本 scale-up、scale-down 和暂停次数；
- 吞吐估计误差及观测陈旧程度；
- 上层在线决策时间；
- Gang placement 成功率与底层放置时间；
- 仅使用聚合容量时产生的 placement infeasibility rate。

---

## 10. Learning + Optimization 方法边界

### 10.1 为什么需要序贯学习

当前决策会影响未来：

- 接纳会形成 SLA 承诺，之后违约比准入时拒绝具有更高成本；
- 接纳任务会占用未来 GPU 容量；
- 扩容可以加速任务完成并提前释放容量；
- 暂停任务可以释放当前资源，但会消耗 DDL slack；
- 使用某一卡型会改变其他兼容任务未来可用的异构容量；
- 最近吞吐观测可能陈旧，其预测误差会影响准入与扩缩容；
- 当前 Gang placement 会影响后续节点碎片和可放置副本数量。

调度器需要在未知未来到达、容量和处理速度下权衡：

```math
\text{current acceleration}
\quad\text{vs.}\quad
\text{future capacity opportunity}.
```

这使问题具有序贯决策和长期信用分配结构。

### 10.2 Learning 模块

候选学习对象包括：

- 长期状态价值 $`V_\theta(s_t)`$；
- 异构 GPU 未来容量的机会成本；
- 任务准入、扩容、缩容和暂停的长期优先级；
- 吞吐观测陈旧程度对应的风险；
- Gang-feasible 候选方案的搜索或排序信号。

RL/DRL 不直接生成 Pod-to-node placement。

最终评价目标与训练奖励严格分开：$`U_j=v_j(B_j-\kappa V_j)`$ 表示业务终局效用，laxity potential 仅用于提供中间学习信号。训练采用无时间折扣回报，不加入 JCT bonus；算法实验必须分别报告无 shaping、使用 shaping 和非 RL 基线，验证收益是否来自长期决策而不是奖励设计偏差。

### 10.3 Optimization 模块

组合优化器根据 learning 输出决定：

- 新任务是否接纳；
- 每个任务在每种兼容卡型上的副本数；
- 哪些候选方案进入底层 Gang feasibility check。

同一周期到达的全部新任务进行联合准入，而不是按固定顺序逐个独立判断；优化器联合选择接纳子集和所有已接纳任务的副本方案。优化采用 9.2 节的两阶段结构：先优化 $`F_t^{\mathrm{svc}}`$，再在服务目标相同或 $`\epsilon_F`$-近似相同的可行动作中最小化 $`G_t^{\mathrm{JCT}}`$。

最终动作必须同时满足：

- GPU 卡型兼容；
- 最大副本数；
- 当前聚合容量；
- $`\mathbf n_t\in\mathcal F_t(s_t^{\mathrm{node}})`$；
- 当前任务进度与 DDL 风险要求。

底层 Gang packer 或 K8s/Volcano dry-run 为候选动作返回 placement certificate。若不可行，上层优化器必须选择其他候选方案，而不能直接执行。

### 10.4 吞吐估计器

吞吐估计器仅在获得有效运行信息时更新，并向上层提供配置 $`c_{jk}`$ 最近一次可用的单副本吞吐估计 $`\overline\mu_{jkt}`$。没有新观测时继续沿用旧值；尚未运行的配置使用系统提供的初始估计。

论文研究重点是：在给定这种不规则更新、可能陈旧的完成时间估计时，如何进行长期在线准入与弹性资源编排；吞吐估计器本身不是第一版论文的主要 learning contribution。

---

## 11. 第一版明确不建模的内容

第一版暂不研究：

1. 在线与离线资源域的显式区分；
2. 公司内部资源回收、驱逐和 victim-selection 算法；
3. setup 时间、setup 状态和 setup 成本；
4. 显式 Pod-to-node 二元决策变量；
5. K8s/Volcano 内部调度算法的重新设计；
6. 吞吐预测器和 P95/P99 完成时间预测器；
7. 多队列 Max-min、公平性和任务优先级；
8. CPU、内存、网络带宽、机架和通信拓扑；
9. 模型切分、张量并行或流水并行方案选择；
10. 运行中跨卡型迁移与 checkpoint；
11. 用户修改参数并重新提交的策略行为；
12. 用户博弈、定价和机制设计；
13. 秒级在线服务 autoscaling；
14. 端到端 RL 直接输出全部调度动作；
15. 事件触发决策机制；
16. 完整生产系统上线与所有控制面机制。

资源回收、启动延迟和执行波动可以影响实际任务进度、容量与 JCT，但在核心数学模型中统一作为环境反馈处理。

---

## 12. 紧凑学术表述

### 中文

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排。任务在线到达，并提交完成期限、带噪声的预估卡时、最大副本数以及不同兼容 GPU 卡型对应的固定 Gang 配置。上层编排器周期性联合决定新任务准入以及各任务在不同卡型上的副本数量，并通过 Gang-feasible allocation set 确保底层 K8s/Volcano 调度器能够完成实际放置。任务处理速度和完成时间在启动前未知，运行后根据实际进度与最近一次有效吞吐观测逐步揭露；未来任务到达与可用容量同样未知。终局服务效用由任务离开系统后、基于实际 GPU 卡时与 batch 信息在历史完成任务池中得到的分位数归一化相对价值加权，奖励按时或迟交完成，并惩罚接纳后的 SLA 违约。方法采用 SLA-first learning + JCT-aware optimization：RL/DRL 学习长期服务价值，DDL-laxity shaping 提供中间训练信号，组合优化器在服务等价方案中进一步降低 JCT，并由 Gang feasibility check 保证动作可执行。

### English

We study online orchestration of elastic offline-inference jobs in high-load heterogeneous GPU clusters. Jobs arrive with completion deadlines, noisy GPU-hour estimates, maximum replica counts, and fixed gang configurations for their compatible GPU types. At each decision epoch, an upper-level orchestrator jointly selects a subset of newly arrived jobs and the number of replicas assigned to each configuration. Its decisions must belong to a gang-feasible allocation set so that the underlying K8s/Volcano scheduler can construct a valid pod-to-node placement. Processing rates and completion times are initially unknown and are progressively revealed from execution, while future arrivals and available capacity remain unknown. Terminal service utility is weighted by a percentile-normalized relative job value assigned after the job leaves the system from its realized GPU-hour footprint and batch information; it assigns diminishing positive value to late completion and penalizes post-admission SLA violations. We combine SLA-first RL/DRL with JCT-aware combinatorial optimization and gang-feasibility verification; DDL-laxity shaping supplies intermediate learning signals without adding JCT to the terminal reward.

---

## 13. 后续研究顺序

1. 建立历史完成任务参考池，分析实际 GPU 卡时、batch 等特征，并确定终局相对价值 $`v_j`$ 的分位数归一化口径；
2. 对申报卡时注入不同幅度噪声，评估准入与 SLA 对 prior 质量的敏感性；
3. 分析初始吞吐估计与运行后观测之间的偏差、更新频率和陈旧程度；
4. 明确底层 K8s/Volcano 可行性接口或 Gang packer 的输入输出；
5. 建立不使用 learning 的在线优化和启发式基线；
6. 建立一个已知完整测试轨迹的离线上界，仅用于衡量实时策略与理想方案的差距；
7. 测量 elasticity gap、foresight gap 和 heterogeneity gap；
8. 比较 aggregate-only 与 gang-aware 编排的可行性和性能差距；
9. 按申报卡时分桶报告准入行为，并按终局价值分位报告已接纳任务结果，检查系统性偏差；
10. 扫描 $`\kappa`$，绘制按时服务价值—SLA 违约率 Pareto 前沿；
11. 比较稀疏终局奖励、laxity reward shaping 和非 RL 方法；
12. 根据 gap experiments 收敛 RL/DRL 的学习对象并设计 learning + optimization 算法；
13. 将数学模型和算法逐步沉淀为论文正文。
