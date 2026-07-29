# Online Orchestration of Elastic Inference Jobs on Heterogeneous GPU Clusters

## 问题定义与数学模型（工作稿）

> 本文档用于固定第一版论文的问题边界、核心符号和建模假设。
>
> 具体的 RL/DRL 结构、组合优化算法和实验方案将在后续文献调研与数据分析后确定。

---

## 1. 一句话问题定义

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排问题。任务随时间动态到达，并提交完成期限、兼容 GPU 卡型对应的固定 Gang 配置和最大副本数。上层编排器周期性决定任务准入以及每个任务在不同卡型上的副本数量，并确保底层 K8s/Volcano 调度器能够完成 Gang placement。任务处理速度在启动前未知，运行后根据最近一次有效吞吐观测更新；未来任务到达和可用容量同样未知。研究目标是奖励按时完成、惩罚接纳后的 SLA 违约，并在 SLA 优先的前提下降低平均与尾部 Job Completion Time（JCT）。

---

## 2. 研究动机与系统边界

### 2.1 研究动机

异构 GPU 集群中的空闲算力具有明显的时间变化：

- 在线业务释放的资源可能在之后被重新收回；
- 离线资源预算也会随集群状态变化；
- 当前空闲的 GPU 时间不能储存到未来；
- 弹性离线推理任务可以通过增加、减少或暂停副本利用这些动态容量。

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

2. **底层执行调度器**
   - 由现有 K8s/Volcano 调度器承担；
   - 将每个副本的 Gang Pods 放置到具体节点；
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
\mathcal K_j,\,
\overline r_j,\,
\{\Gamma_{jk}\}_{k\in\mathcal K_j}
\right),
```

其中：

- $`a_j`$：实际提交时刻；
- $`d_j`$：任务完成期限，即业务 SLA；
- $`\mathcal K_j\subseteq\mathcal K`$：任务可使用的 GPU 卡型集合；
- $`\overline r_j`$：任务允许同时运行的最大副本总数；
- $`\Gamma_{jk}=(p_{jk},g_{jk})`$：卡型 $`k`$ 下一个副本的固定 Gang 配置。

任务不提交最小副本数，统一采用

```math
\underline r_j=0.
```

因此，已接纳任务可以在内部队列中等待，也可以在运行过程中缩容至零，并在后续决策时点重新获得副本。

### 4.1 配置语义

对每个 $`k\in\mathcal K_j`$，定义可行单副本配置

```math
c_{jk}=(k,\Gamma_{jk}).
```

配置 $`c_{jk}`$ 同时给出 GPU 卡型和固定 Gang shape，表示一个能够独立处理 batch 分片的完整副本。卡型差异直接体现在副本的资源形状、当前可用容量和 Gang placement 可行性上。

任务可以同时运行多个副本，不同副本可以选择不同的兼容配置。每个副本产生的实际处理进度都由执行系统以统一的 batch 进度口径汇总。

### 4.2 未知处理速度

上层编排器不假设任务提交时已知准确完成时间或固定处理速度。系统可以为尚未运行的配置提供初始吞吐估计；任务运行后，执行系统根据实际进度产生新的吞吐观测。论文直接使用每个可行配置的吞吐估计。

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

1. 本周期新到达任务；
2. 已接纳但尚未完成的任务；
3. 每个任务的剩余 batch 工作比例；
4. 每个任务最近一次可用的吞吐估计及其观测时间；
5. 当前各任务在各卡型上的副本数量；
6. 当前节点、GPU、Pod 和已有 placement 状态；
7. 当前由集群环境提供的可用 GPU 容量。

调度器不知道：

1. 未来任务到达及其参数；
2. 未来集群容量变化；
3. 未来资源回收；
4. 未来真实处理速度；
5. 下一次有效吞吐观测何时产生。

历史数据可以用于训练 RL/DRL 模型，但在线策略不能访问真实未来。实验中的 clairvoyant oracle 是唯一允许使用未来信息的对照方法。

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

准入决策不可撤销，但接纳后的副本配置可以在后续决策时点修改：

```math
\text{irrevocable admission}
+
\text{adaptive replica recourse}.
```

每轮只执行当前周期决策。为判断 DDL 风险而生成的未来计划不构成不可修改的资源预留。

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

在没有新吞吐观测时，计划期内使用最近一次估计。下一次获得新观测后，调度器重新计算剩余完成时间和资源方案。未来计划只用于风险评估，不构成不可修改的资源预留。

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

### 7.2 Gang-feasible allocation set

令 $`s_t^{\mathrm{node}}`$ 表示时点 $`t`$ 的实际节点与已有 placement 状态。定义：

```math
\mathcal F_t(s_t^{\mathrm{node}})
=
\left\{
\mathbf n_t:
\begin{array}{l}
\text{K8s/Volcano can construct a valid Pod-to-node}\\
\text{Gang placement for all requested replicas in }\mathbf n_t
\end{array}
\right\}.
```

其中

```math
\mathbf n_t=(n_{jkt})_{j,k}
```

是上层期望在下一周期运行的任务—卡型副本矩阵。

$`\mathbf n_t\in\mathcal F_t(s_t^{\mathrm{node}})`$ 表示存在一个 placement certificate $`\pi_t`$，满足：

- 每个副本使用提交的 Gang shape $`\Gamma_{jk}`$；
- 每个 Pod 完整位于一个节点；
- 同一副本的全部 Pod 使用同一卡型并能够同时启动；
- 多个 Pod 可以共享节点；
- 所有节点 GPU 容量约束均满足；
- 当前保留的 Pod 与已有 placement 状态得到正确处理。

上层不显式建立 Pod-to-node 二元变量。算法实现时，可以通过 K8s/Volcano dry-run、可行性接口或独立的 Gang packer：

```math
\mathrm{GangPlace}
\left(
s_t^{\mathrm{node}},
\mathbf n_t
\right)
\longrightarrow
\begin{cases}
\pi_t, & \mathbf n_t\text{ is feasible},\\
\mathrm{infeasible}, & \text{otherwise}.
\end{cases}
```

只有返回 placement certificate 的候选方案才能成为最终上层动作。这样，上层模型保持低维，同时不会向底层提交无法执行的副本安排。

最终执行动作可以写为

```math
a_t=
\left(
\{z_j:j\in\mathcal J_t^{\mathrm{new}}\},
\mathbf n_t,
\pi_t
\right),
```

其中 $`\pi_t`$ 是底层 Gang packer 或 K8s/Volcano 为 $`\mathbf n_t`$ 返回的可执行 placement certificate。$`\pi_t`$ 不作为 RL 动作，也不通过显式 Pod-to-node 二元变量进入核心数学模型。

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
- $`W_{jt}`$：任务剩余 batch 工作比例；
- $`\overline\mu_{jkt}`$：任务 $`j`$ 的配置 $`c_{jk}`$ 最近一次可用的单副本吞吐估计。

未来的 $`C_{kt}`$、$`s_t^{\mathrm{node}}`$、任务到达和吞吐观测均未知。

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

若任务在 $`d_j`$ 时仍未完成，也记为 $`V_j=1`$。按时完成表示平台履行了承诺；拒绝表示平台未作出承诺；接纳后超时表示已经作出的 SLA 承诺被违反。由于未来容量、回收和处理速度未知，论文不声称对所有随机环境提供确定性零违约保证，而是在目标中显式惩罚接纳后的违约。

---

## 9. 优化目标

### 9.1 首要目标：SLA 服务效用

所有任务具有相同的按时完成收益和相同的 SLA 违约成本。令 $`\kappa>0`$ 为接纳后违约的惩罚系数，策略 $`\pi`$ 的首要目标为

```math
\max_\pi
\mathbb E_\pi
\left[
\sum_{j\in\mathcal J}
\left(
S_j-\kappa V_j
\right)
\right].
```

按时完成的效用为 $`+1`$，拒绝的效用为 $`0`$，接纳后违约的效用为 $`-\kappa`$。因此，接纳不再弱优于拒绝。即使暂不考虑资源竞争，若任务按时完成的估计概率为 $`p_j`$，接纳的期望效用为

```math
p_j-\kappa(1-p_j),
```

仅当

```math
p_j>\frac{\kappa}{1+\kappa}
```

时才具有正的直接效用。在完整系统中，接纳还会消耗异构 GPU 容量并影响其他已接纳任务，因此 admission control 同时考虑任务自身的履约风险和容量机会成本。$`\kappa`$ 越大，策略越重视避免已经承诺的任务违反 SLA。

### 9.2 次要目标：JCT

令

```math
\mathcal J^{\mathrm{succ}}
=
\{j\in\mathcal J:S_j=1\}.
```

成功任务的平均 JCT 为

```math
\mathrm{MeanJCT}
=
\frac{1}{|\mathcal J^{\mathrm{succ}}|}
\sum_{j\in\mathcal J^{\mathrm{succ}}}
(C_j-a_j).
```

在首要 SLA 服务效用相同的情况下，进一步降低按时完成任务的平均和尾部 JCT：

```math
\text{lexicographically optimize}
\left(
\max
\mathbb E_\pi
\left[
\sum_j(S_j-\kappa V_j)
\right],\;
\min\mathrm{MeanJCT},\;
\min\mathrm{TailJCT}
\right).
```

TailJCT 可在实验中采用 P95、P99 或 CVaR。SLA 的优先级严格高于 JCT。

### 9.3 评价指标与训练奖励

业务评价指标与 RL 训练奖励分开定义。实验至少报告：

```math
\mathrm{OnTimeYield}
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

最终业务评价仍由按时完成、接纳后违约和 JCT 决定。为了缓解任务结束前奖励稀疏和长期信用分配困难，RL 可以使用基于 DDL laxity 的 potential-based reward shaping。

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

laxity 越小，$`\Phi(s_t)`$ 越低。训练奖励采用

```math
r_t^{\mathrm{train}}
=
r_t^{\mathrm{terminal}}
+
\lambda_{\mathrm{shape}}
\left[
\gamma\Phi(s_{t+1})-\Phi(s_t)
\right],
```

其中

```math
r_t^{\mathrm{terminal}}
=
\begin{cases}
+1, & \text{任务在 DDL 前完成},\\
-\kappa, & \text{已接纳任务在 DDL 时仍未完成},\\
0, & \text{其他情况}.
\end{cases}
```

完成或到期任务从 active state 中移除，并将终止状态 potential 设为零。Potential 差值为扩容、暂停和准入决策提供中间风险信号，但不替代最终 SLA 服务效用。具体 potential 形式和 $`\lambda_{\mathrm{shape}}`$ 属于方法设计，将通过消融实验验证。

### 9.4 解释性指标

以下指标用于解释策略行为，但不高于 SLA 与 JCT：

- GPU 资源占用率与有效利用率；
- 各卡型利用情况；
- 任务接纳率与拒绝率；
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

最终评价目标与训练奖励严格分开：$`S_j-\kappa V_j`$ 表示业务终局效用，laxity potential 仅用于提供中间学习信号。算法实验必须分别报告无 shaping、使用 shaping 和非 RL 基线，验证收益是否来自长期决策而不是奖励设计偏差。

### 10.3 Optimization 模块

组合优化器根据 learning 输出决定：

- 新任务是否接纳；
- 每个任务在每种兼容卡型上的副本数；
- 哪些候选方案进入底层 Gang feasibility check。

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

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排。任务在线到达，并提交完成期限、最大副本数以及不同兼容 GPU 卡型对应的固定 Gang 配置。上层编排器周期性决定任务准入以及各任务在不同卡型上的副本数量，并通过 Gang-feasible allocation set 确保底层 K8s/Volcano 调度器能够完成实际放置。任务处理速度在启动前未知，运行后根据最近一次有效吞吐观测更新；未来任务到达、可用容量和处理速度同样未知。研究目标是奖励按时完成、惩罚接纳后的 SLA 违约，并在 SLA 服务效用优先的前提下降低平均与尾部 JCT。方法采用 RL/DRL 与组合优化相结合的结构，使用 DDL laxity 提供中间训练信号，由优化与 Gang feasibility check 保证动作可执行。

### English

We study online orchestration of elastic offline-inference jobs in high-load heterogeneous GPU clusters. Jobs arrive with completion deadlines, maximum replica counts, and fixed gang configurations for their compatible GPU types. At each decision epoch, an upper-level orchestrator determines job admission and the number of replicas assigned to each configuration. Its decisions must belong to a gang-feasible allocation set so that the underlying K8s/Volcano scheduler can construct a valid pod-to-node placement. Processing rates are unknown before execution and are revised only when new runtime observations become available. The objective rewards on-time completion, explicitly penalizes SLA violations after admission, and then minimizes mean and tail job completion time. We combine RL/DRL for long-term decisions with combinatorial optimization and gang-feasibility verification, while DDL-laxity shaping supplies intermediate training signals without replacing the terminal service objective.

---

## 13. 后续研究顺序

1. 分析初始吞吐估计与运行后观测之间的偏差；
2. 分析吞吐观测的更新频率、陈旧程度与误差；
3. 明确底层 K8s/Volcano 可行性接口或 Gang packer 的输入输出；
4. 建立不使用 learning 的在线优化和启发式基线；
5. 建立 clairvoyant dynamic oracle；
6. 测量 elasticity gap、foresight gap 和 heterogeneity gap；
7. 比较 aggregate-only 与 gang-aware 编排的可行性和性能差距；
8. 根据 gap experiments 确定 RL/DRL 的学习对象；
9. 比较稀疏终局奖励、laxity reward shaping 和非 RL 方法；
10. 设计 learning + optimization 算法；
11. 将数学模型和算法逐步沉淀为论文正文。
