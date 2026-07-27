# Online Orchestration of Elastic Inference Jobs on Heterogeneous GPU Clusters

## 问题定义与数学模型（工作稿）

> 本文档用于固定第一版论文的问题边界、核心符号和建模假设。
>
> 具体的 RL/DRL 结构、组合优化算法和实验方案将在后续文献调研与数据分析后确定。

---

## 1. 一句话问题定义

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排问题。任务随时间动态到达，用户提交估计标准卡时、完成期限、兼容 GPU 卡型集合、最大副本数和固定 Gang 配置；上层编排器周期性决定任务准入以及每个任务在不同卡型上的副本数量，并确保该决策能够被底层 K8s/Volcano 调度器实际完成 Gang placement。未来任务到达、可用容量和任务处理速度均未知，研究目标是在所有到达请求上首先最大化 DDL SLA 达成数量，其次降低平均与尾部 Job Completion Time（JCT）。

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

对于任务 $j$ 和兼容 GPU 卡型 $k$：

- 一个副本由 $p_{jk}$ 个 Pod 构成；
- 每个 Pod 需要 $g_{jk}$ 张类型 $k$ 的 GPU；
- 一个 Pod 必须完整放置在一个节点上；
- 同一副本的不同 Pod 可以位于相同或不同节点；
- 同一副本的全部 Pod 必须使用同一种 GPU 卡型；
- 全部 Pod 必须能够同时启动，副本才可以运行；
- 不同副本可以同时使用不同的兼容 GPU 卡型。

定义一个副本在卡型 $k$ 上消耗的总 GPU 数为

```math
q_{jk}=p_{jk}g_{jk}.
```

$q_{jk}$ 用于上层容量估算，而完整 Gang shape

```math
\Gamma_{jk}=(p_{jk},g_{jk})
```

会传递给底层 K8s/Volcano 调度器进行节点级可行性检查。仅满足总 GPU 数量并不自动意味着 Gang placement 可行。

少量任务的单 Pod GPU 需求可以为 $0.5$，大多数需求为一张或多张整数 GPU。Pod 始终是不可拆分的放置单元。

---

## 4. 用户提交信息

任务 $j$ 的提交信息为

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

- $a_j$：实际提交时刻；
- $d_j$：任务完成期限，即业务 SLA；
- $\widetilde H_j$：用户申报的估计标准卡时；
- $\mathcal K_j\subseteq\mathcal K$：任务可使用的 GPU 卡型集合；
- $\overline r_j$：任务允许同时运行的最大副本总数；
- $\Gamma_{jk}=(p_{jk},g_{jk})$：卡型 $k$ 下一个副本的固定 Gang 配置。

任务不提交最小副本数，统一采用

```math
\underline r_j=0.
```

因此，已接纳任务可以在内部队列中等待，也可以在运行过程中缩容至零，并在后续决策时点重新获得副本。

### 4.1 估计卡时的语义

$\widetilde H_j$ 是准入时可获得的工作量先验，用于：

- 在任务尚未产生有效吞吐观测时估计其初始完成时间；
- 判断当前容量和 DDL 是否足以支持任务；
- 作为 SLA 风险评估的一部分。

$\widetilde H_j$ 不是准确的真实工作量，也不是强制查杀上限。任务实际完成状态由 batch 数据是否处理完决定。

### 4.2 异构 GPU 换算

所有用户卡时采用统一的参考 GPU 口径。定义

```math
\eta_k>0
```

为卡型 $k$ 相对于参考卡型 HC 的全局换算系数，并令

```math
\eta_{\mathrm{HC}}=1.
```

第一版模型假设 $\eta_k$ 只与 GPU 卡型有关，不随任务变化。任务可以同时使用多种兼容卡型，不同卡型完成的有效工作量可以累加。

---

## 5. 在线决策过程

### 5.1 决策时点

时间被离散为

```math
\mathcal T=\{0,1,\ldots,T\},
```

相邻决策时点之间的实际间隔为 $\Delta$。第一版取 $\Delta=10$ 分钟。

实际到达时刻为 $a_j$ 的任务，在最近的下一个决策时点参加准入：

```math
\tau_j
=
\min\{t\in\mathcal T:t\Delta\ge a_j\}.
```

固定周期决策是第一版设定。事件触发或“固定周期 + 事件触发”的混合机制属于后续扩展。

### 5.2 决策时点可观测信息

在时点 $t$，上层编排器观察：

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
   - 用户修改 DDL、估计卡时或兼容卡型后重新提交时，视为新任务。

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

### 6.1 初始估计

令 $W_{jt}\in[0,1]$ 表示决策时点 $t$ 任务 $j$ 的剩余 batch 工作比例。任务提交时：

```math
W_{j,\tau_j}=1.
```

在尚未获得运行时吞吐观测时，用户申报卡时给出初始的 HC 等价单位 GPU 处理效率：

```math
\nu_{j0}=\frac{1}{\widetilde H_j}.
```

其单位为“每 HC 等价 GPU-hour 可完成的任务比例”。

### 6.2 最近一次可用吞吐

吞吐估计不要求在每个决策时点更新。令 $\mathcal O_j(t)$ 表示截至时点 $t$，任务 $j$ 已产生有效吞吐观测的时点集合。若该集合非空，定义最近观测时点：

```math
\ell_j(t)=\max \mathcal O_j(t).
```

调度器使用的 HC 等价处理效率为

```math
\overline\nu_{jt}
=
\begin{cases}
\nu^{\mathrm{obs}}_{j,\ell_j(t)},
& \mathcal O_j(t)\neq\varnothing,\\
\nu_{j0},
& \mathcal O_j(t)=\varnothing.
\end{cases}
```

如果当前决策周期没有新观测，则继续沿用最近一次 $\overline\nu_{jt}$。因此，该估计是事件更新、周期使用的，而不是假设每个决策时点都会重新预测。

运行时估计能力统一输出 HC 等价吞吐。对于其他兼容卡型，上层使用卡型换算关系估计一个副本的处理速度：

```math
\widehat\mu_{jkt}
=
q_{jk}\eta_k\overline\nu_{jt}.
```

其中 $\widehat\mu_{jkt}$ 仅是时点 $t$ 可用于规划的估计值，不代表未来真实速度。

吞吐估计模块被视为系统提供的外部能力。第一版论文不研究吞吐预测模型、P95/P99 完成时间预测或概率校准。

### 6.3 规划中的完成时间估计

给定从时点 $t$ 开始的候选副本计划 $\{n_{jk\tau}\}$，调度器使用最近估计预测剩余工作：

```math
\widehat W_{j,\tau+1}
=
\left[
\widehat W_{j\tau}
-
\Delta
\sum_{k\in\mathcal K_j}
\widehat\mu_{jk\tau}n_{jk\tau}
\right]^+,
\qquad
\widehat W_{jt}=W_{jt}.
```

对应的预计完成时点为

```math
\widehat C_j(t)
=
\Delta
\inf
\left\{
\tau\ge t:
\widehat W_{j\tau}=0
\right\}.
```

在没有新吞吐观测时，计划期内使用最近一次估计。下一次获得新观测后，再重新计算剩余完成时间与资源方案。

### 6.4 实际进度更新

实际状态不使用一个假设已知的“真实每期吞吐”进行更新。令

```math
\delta^{\mathrm{obs}}_{jt}\in[0,W_{jt}]
```

表示执行系统在区间 $[t\Delta,(t+1)\Delta)$ 内实际完成的 batch 工作比例，则：

```math
W_{j,t+1}
=
W_{jt}
-
\delta^{\mathrm{obs}}_{jt}.
```

$\delta^{\mathrm{obs}}_{jt}$ 在区间结束后由实际任务进度得到，是环境反馈而非决策变量。资源回收、启动延迟、运行波动或部分周期执行都会自然反映在该实际进度中，无需分别建模。

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

令 $s_t^{\mathrm{node}}$ 表示时点 $t$ 的实际节点与已有 placement 状态。定义：

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

$\mathbf n_t\in\mathcal F_t(s_t^{\mathrm{node}})$ 表示存在一个 placement certificate $\pi_t$，满足：

- 每个副本使用提交的 Gang shape $\Gamma_{jk}$；
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

其中 $\pi_t$ 是底层 Gang packer 或 K8s/Volcano 为 $\mathbf n_t$ 返回的可执行 placement certificate。$\pi_t$ 不作为 RL 动作，也不通过显式 Pod-to-node 二元变量进入核心数学模型。

---

## 8. 紧凑数学模型

### 8.1 集合、参数与状态

- $\mathcal J$：研究周期内的全部任务；
- $\mathcal J_t^{\mathrm{new}}$：时点 $t$ 首次参加准入的新任务；
- $\mathcal J_t^{\mathrm{acc}}$：时点 $t$ 已接纳但尚未完成的任务；
- $\mathcal K$：GPU 卡型集合；
- $\mathcal K_j$：任务 $j$ 的兼容卡型集合；
- $\mathcal T$：离散决策时点集合；
- $C_{kt}$：当前可供弹性推理任务使用的卡型 $k$ 聚合 GPU 容量；
- $s_t^{\mathrm{node}}$：底层节点容量和已有 placement 状态；
- $W_{jt}$：任务剩余 batch 工作比例；
- $\overline\nu_{jt}$：最近一次可用的 HC 等价处理效率。

未来的 $C_{kt}$、$s_t^{\mathrm{node}}$、任务到达和吞吐观测均未知。

### 8.2 决策变量

#### 准入变量

```math
z_j\in\{0,1\},
```

其中 $z_j=1$ 表示任务 $j$ 在其首次决策点被接纳。该变量一旦确定便不再修改。

#### 卡型—副本数量

```math
n_{jkt}\in\mathbb Z_+,
```

表示时点 $t$，上层编排器要求任务 $j$ 在卡型 $k$ 上运行的副本数。

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
q_{jk}\eta_k\overline\nu_{jt}n_{jkt}
\right]^+.
```

下一决策时点再用实际观测 $\delta^{\mathrm{obs}}_{jt}$ 修正状态。

### 8.4 完成、DDL 与 JCT

任务实际完成时刻为

```math
C_j
=
\Delta
\inf\{t\in\mathcal T:W_{jt}=0\}.
```

任务的 JCT 从用户实际提交时刻开始计算：

```math
\mathrm{JCT}_j=C_j-a_j.
```

定义请求 $j$ 的 SLA 达成指标：

```math
S_j=
\begin{cases}
1, & z_j=1\text{ and }C_j\le d_j,\\
0, & \text{otherwise}.
\end{cases}
```

因此：

- 被拒绝任务的 $S_j=0$；
- 已接纳但超过 DDL 的任务 $S_j=0$；
- 只有被接纳且按时完成的任务 $S_j=1$。

DDL 是高违约成本的业务 SLA。由于未来容量、回收和处理速度未知，论文目标是最大化实际 SLA 达成，而不是声称对所有随机环境提供确定性零违约保证。

---

## 9. 优化目标

### 9.1 首要目标：SLA 达成

所有到达请求进入评价分母：

```math
\text{SLA Attainment}
=
\frac{\sum_{j\in\mathcal J}S_j}
{|\mathcal J|}.
```

首要目标为

```math
\max \sum_{j\in\mathcal J}S_j.
```

所有任务具有相同的准入收益和 SLA 违约代价，不设置任务价值或优先级权重。拒绝和接纳后超时均记为 SLA 未达成，避免“拒绝所有任务”形成退化解。

### 9.2 次要目标：JCT

在 SLA 达成数量相同的情况下，进一步降低完成任务的平均和尾部 JCT：

```math
\text{lexicographically optimize}
\left(
\max\sum_jS_j,\;
\min\mathrm{MeanJCT},\;
\min\mathrm{TailJCT}
\right).
```

TailJCT 可在实验中采用 P95、P99 或 CVaR。SLA 的优先级严格高于 JCT。

### 9.3 解释性指标

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

- 长期状态价值 $V_\theta(s_t)$；
- 异构 GPU 未来容量的机会成本；
- 任务准入、扩容、缩容和暂停的长期优先级；
- 吞吐观测陈旧程度对应的风险；
- Gang-feasible 候选方案的搜索或排序信号。

RL/DRL 不直接生成 Pod-to-node placement。

### 10.3 Optimization 模块

组合优化器根据 learning 输出决定：

- 新任务是否接纳；
- 每个任务在每种兼容卡型上的副本数；
- 哪些候选方案进入底层 Gang feasibility check。

最终动作必须同时满足：

- GPU 卡型兼容；
- 最大副本数；
- 当前聚合容量；
- $\mathbf n_t\in\mathcal F_t(s_t^{\mathrm{node}})$；
- 当前任务进度与 DDL 风险要求。

底层 Gang packer 或 K8s/Volcano dry-run 为候选动作返回 placement certificate。若不可行，上层优化器必须选择其他候选方案，而不能直接执行。

### 10.4 吞吐估计器

吞吐估计器仅在获得有效运行信息时更新，并向上层提供最近一次 HC 等价处理效率 $\overline\nu_{jt}$。没有新观测时继续沿用旧值。

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

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排。任务在线到达，并提交估计标准卡时、完成期限、兼容 GPU 卡型集合、最大副本数和固定 Gang 配置。上层编排器周期性决定任务准入以及各任务在不同卡型上的副本数量，并通过 Gang-feasible allocation set 确保决策能够由底层 K8s/Volcano 调度器实际放置。任务处理速度在启动前未知，运行后也只在获得有效观测时更新；调度器使用最近一次 HC 等价吞吐及卡型换算关系估计剩余完成时间。未来任务到达、可用容量和处理速度均未知。研究目标是首先最大化 DDL SLA 达成数量，其次降低平均与尾部 JCT，并采用 RL/DRL 与组合优化相结合的方法进行求解。

### English

We study online orchestration of elastic offline-inference jobs in high-load heterogeneous GPU clusters. Jobs arrive over time with an estimated normalized GPU-hour demand, a completion deadline, a set of compatible GPU types, a maximum replica count, and a fixed gang configuration. At each decision epoch, an upper-level orchestrator determines job admission and the number of replicas assigned to each GPU type. Its decisions must belong to a gang-feasible allocation set so that the underlying K8s/Volcano scheduler can construct a valid pod-to-node placement. Job processing rates are unknown before execution and are updated only when new runtime observations become available; otherwise, the scheduler reuses the latest HC-equivalent throughput estimate and extrapolates across GPU types. The primary objective is to maximize deadline-SLA attainment, followed by minimizing mean and tail job completion time. We pursue a learning-plus-optimization approach that combines RL/DRL for long-term decision making with combinatorial optimization and gang-feasibility verification.

---

## 13. 后续研究顺序

1. 用真实数据验证估计卡时与实际完成进度之间的偏差；
2. 分析吞吐观测的更新频率、陈旧程度与误差；
3. 明确底层 K8s/Volcano 可行性接口或 Gang packer 的输入输出；
4. 建立不使用 learning 的在线优化和启发式基线；
5. 建立 clairvoyant dynamic oracle；
6. 测量 elasticity gap、foresight gap 和 heterogeneity gap；
7. 比较 aggregate-only 与 gang-aware 编排的可行性和性能差距；
8. 根据 gap experiments 确定 RL/DRL 的学习对象；
9. 设计 learning + optimization 算法；
10. 将数学模型和算法逐步沉淀为论文正文。
