# Online Orchestration of Elastic Inference Jobs on Heterogeneous GPU Clusters

## 问题定义与数学模型（工作稿）

> **Working title:** *Online Orchestration of Elastic Inference Jobs on Heterogeneous GPU Clusters*
>
> **文档目的：** 固定论文的问题语义、系统边界和数学模型，作为后续文献调研、算法设计、模拟器开发与论文写作的共同基准。本文档暂不锁定具体的 RL/DRL 架构、训练算法或组合优化求解器。

---

## 1. 一句话问题定义

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线资源编排问题：任务随时间动态到达，用户提交估计卡时、完成期限、兼容卡型和最大副本数；统一调度器周期性联合决定任务准入、资源域与卡型选择、Gang Pod 放置和副本扩缩容，在未来任务到达、可用容量和真实处理速度均未知的情况下，首先最大化任务 SLA 达成率，其次降低平均与尾部 Job Completion Time（JCT）。

---

## 2. 研究动机与运行场景

京东零售当前采用在线、离线 GPU 集群相对隔离的部署模式。高优先级在线和离线业务会形成大量零散、短时、位置分散且持续时间未知的空闲 GPU。弹性离线推理任务可暂停、恢复和动态调整副本数量，因而适合利用这些不可储存的闲置算力。

本研究面向未来资源占用率维持高位（例如不低于 95%）的运行场景，重点不是简单填满当前空闲资源，而是在高负载和不确定环境中：

1. 尽可能保障任务在 DDL 前完成；
2. 降低任务的平均 JCT 和尾部 JCT；
3. 合理利用不同资源域、GPU 卡型和节点上的动态空闲资源；
4. 避免因短视准入、频繁启动或不合理放置导致后续 SLA 违约。

这里的“高负载”是问题的运行环境和实验条件，不是通过无意义地增加 GPU 占用来实现的优化目标。GPU 利用率作为重要评价指标，但其优先级低于 SLA 和 JCT。

项目背景与原始目标参见：[异构 GPU 集群在线动态资源编排项目计划书](./异构%20GPU%20集群在线动态资源编排项目计划书%20(1).pdf)。

---

## 3. 系统边界

### 3.1 统一调度器与两个资源域

统一调度器管理两个相对隔离的资源域：

1. **在线资源域（online resource domain）**
   - 资源来自在线集群暂时释放的潮汐 GPU；
   - 可用资源与具体节点绑定；
   - 在线业务需要资源时，会在原节点回收；
   - 回收可能使已放置的 Pod 立即停止。

2. **离线资源域（offline resource domain）**
   - 资源来自离线集群当前可用于弹性推理的 GPU；
   - 本文研究的弹性推理任务同属一个队列；
   - 不建模队列之间的 Max-min 配额、优先级和公平性约束。

同一任务的不同副本可以同时运行在两个资源域中，也可以使用不同的兼容 GPU 卡型。但是，一个副本的全部 Pod 必须完整部署在同一个资源域中。

### 3.2 外生容量变化

公司内部的驱逐、回收和容量生成算法持续演进，不属于本文的控制决策。本文将其视为外生环境：

- 调度器在决策时点观察当前可用节点和 GPU 容量；
- 两个决策时点之间可能发生资源释放或回收；
- 回收策略如何选择触发时机和 victim 不显式建模；
- 调度器只观察回收后的资源状态和存活副本，并在后续重新编排。

在线执行时，不假设已知未来容量变化的概率分布。

---

## 4. 任务与执行层级

### 4.1 弹性离线推理任务

每个任务对应一个有限的离线 batch。任务完成的定义是其全部 batch 数据均已处理完成。任务的多个副本可以并行处理互不重叠的数据分片，总处理速度近似等于各个活跃副本处理速度之和。

为统一不同任务的数据规模，可将每个任务的总 batch 工作量归一化为 1。令 \(W_{jt}\in[0,1]\) 表示决策时点 \(t\) 的剩余工作比例，则任务进入系统时 \(W_{j,\tau_j}=1\)，完成时 \(W_{jt}=0\)，其中 \(\tau_j\) 是任务到达后的首个决策时点。真实系统实现时，也可以将其替换为剩余样本数、文件数或请求数。

所有任务对内部平台具有相同的重要性：

- 每个任务的 SLA 达成收益相同；
- 每个任务的 SLA 违约代价相同；
- 不引入用户等级、任务价值或优先级权重。

### 4.2 Task–Replica–Pod–Node 层级

系统执行层级为：

\[
\text{Task}
\longrightarrow
\text{Replicas}
\longrightarrow
\text{Gang Pods}
\longrightarrow
\text{Nodes/GPUs}.
\]

- **Task**：用户提交的完整 batch 推理任务；
- **Replica**：包含一份完整模型参数、能够独立处理 batch 分片的执行副本；
- **Pod**：一个副本经过固定切分后形成的部署单元；
- **Node/GPU**：Pod 实际占用的物理资源。

一个副本可以由多个 Pod 组成。模型切分方案由相邻的训练/推理框架预先确定，调度器不决定如何切分模型。

### 4.3 Gang scheduling

一个副本内部实行 Gang Scheduling：

- 一个 Pod 必须完整部署在一个节点上，不能跨节点拆分；
- 同一副本的不同 Pod 可以部署在同一节点，也可以部署在不同节点；
- 同一副本的全部 Pod 必须位于同一资源域并使用同一种 GPU 卡型；
- 全部 Pod 都成功放置并完成 setup 后，副本才能开始处理数据；
- 任意一个 Pod 被回收或失效，整个副本停止贡献吞吐；
- 同一任务的不同副本相互独立，一个副本失效不影响其他副本继续执行。

### 4.4 GPU 资源粒度

对于任务 \(j\) 和 GPU 卡型 \(k\)，执行框架提供固定的副本配置：

- \(p_{jk}\)：一个副本包含的 Pod 数量；
- \(g_{jk}\)：每个 Pod 占用的 GPU 数量；
- \(q_{jk}=p_{jk}g_{jk}\)：一个副本的总 GPU 需求。

允许

\[
g_{jk}\in\{0.5,1,2,3,\ldots\}.
\]

少量任务的单 Pod 需求可以为 0.5 GPU；大多数 Pod 需要一张或多张整数 GPU。即使 GPU 需求为 0.5，Pod 仍是不可拆分的离散放置单元。

---

## 5. 用户提交信息

任务 \(j\) 的提交参数定义为

\[
\mathcal J_j=
\left(
a_j,\,
d_j,\,
\widetilde H_j,\,
\mathcal K_j,\,
\overline r_j,\,
\{p_{jk},g_{jk}\}_{k\in\mathcal K_j}
\right),
\]

其中：

- \(a_j\)：任务到达时刻；
- \(d_j\)：任务完成期限（DDL）；
- \(\widetilde H_j\)：用户申报的标准卡时，是任务真实资源需求的初始估计；
- \(\mathcal K_j\)：任务允许使用的 GPU 卡型集合；
- \(\overline r_j\)：任务在所有资源域和卡型上的最大副本总数；
- \(p_{jk}\)：卡型 \(k\) 下一个副本所需的 Pod 数量；
- \(g_{jk}\)：卡型 \(k\) 下每个 Pod 的 GPU 需求。

任务不提交最小副本数，统一采用

\[
\underline r_j=0.
\]

因此，已接纳任务可以排队、运行、缩容至零、暂停和再次恢复。

### 5.1 标准卡时与卡型换算

所有用户卡时采用统一口径。令

\[
\eta_k>0
\]

表示 GPU 卡型 \(k\) 相对于参考卡型的全局处理能力换算系数，并令

\[
\eta_{\mathrm{HC}}=1.
\]

第一版模型假设 \(\eta_k\) 只与 GPU 卡型有关，而不随任务变化。该换算关系用于任务启动前的准入估算，以及尚未在某种卡型上运行时的跨卡型吞吐外推。

\(\widetilde H_j\) 不是准确的真实工作量，也不是任务的强制查杀阈值。它只是准入和初始资源规划所使用的先验估计。

在任务尚未运行、没有实时吞吐观测时，申报卡时给出初始的单位标准 GPU 处理效率：

\[
\widehat\nu_{j0}
=
\frac{1}{\widetilde H_j},
\]

其中 \(\widehat\nu_{j0}\) 表示一标准 GPU-hour 预计能够完成的任务比例。

---

## 6. 处理速度的不确定性与在线更新

### 6.1 逐步揭露的处理速度

任务启动前，其真实处理速度难以准确获知。系统为任务维护一个初始基准吞吐估计。任务运行后，调度器可以根据最近一个决策周期内完成的 batch 数量更新吞吐和剩余完成时间估计。

令：

- \(\mu_{jkt}\)：任务 \(j\) 在卡型 \(k\) 上、时点 \(t\) 的真实单副本处理速度；
- \(\widehat\mu_{jkt}\)：调度器在时点 \(t\) 可获得的单副本处理速度估计。
- \(\widehat\nu_{jt}\)：根据任务历史进度更新的单位标准 GPU 处理效率估计。

真实的 \(\mu_{jkt}\) 在任务运行前未知，并通过实际执行逐步揭露。对于尚未直接观测的卡型，可利用全局换算关系估计：

\[
\widehat\mu_{jkt}
=
q_{jk}\eta_k\widehat\nu_{jt}.
\]

任务尚未运行时采用 \(\widehat\nu_{jt}=\widehat\nu_{j0}\)。任务运行后，系统根据已完成工作比例、实际运行时间和已使用 GPU 更新 \(\widehat\nu_{jt}\)。因此，卡型换算关系保持全局统一，而任务自身的处理效率通过在线观测动态修正。

运行时的吞吐估计模块视为系统已经或即将具备的外部能力。第一版论文不以 P95 完成时间预测、概率校准或吞吐预测模型本身作为主要研究贡献。

### 6.2 线性副本加速近似

任务的不同活跃副本处理独立 batch 分片。第一版采用副本间近似线性加速：

\[
\text{aggregate throughput}
\approx
\sum_{\text{active replicas}}
\text{per-replica throughput}.
\]

本文不建模训练任务中的全局同步、AllReduce 或最慢 worker 阻塞效应。

---

## 7. Setup、伸缩与进度语义

### 7.1 Setup

新增副本需要经历拉取镜像、加载模型等准备过程，通常持续约 10–20 分钟。

- 副本进入 setup 时，其全部 Pod 已绑定节点并占用 GPU；
- setup 期间副本不贡献处理吞吐；
- 全部 Pod 完成 setup 后，副本转为 active；
- 一个任务首次成功启动副本后，后续副本的 setup 时间可视为一个固定的任务参数；
- 从零副本恢复为正副本数同样属于 scale-up，需要重新 setup。

### 7.2 无迁移

第一版不考虑运行中迁移：

- 已有副本可以保持原位置；
- 调度器可以删除副本或新增副本；
- 改变已有副本的节点、资源域或 GPU 卡型，等价于删除旧副本并重新启动新副本；
- 重新启动需要重新经历 setup。

### 7.3 缩容和回收时的任务进度

任务已经处理完成的 batch 进度永久保留。副本被主动缩容或外生回收时：

- 已完成数据不需要重做；
- 少量尚未完成的 in-flight batch 可以重新进入待处理集合；
- 第一版不单独建模 in-flight batch 的重做成本；
- 不建模 checkpoint、恢复带宽或大规模工作量回滚。

因此，资源中断的主要成本是：

1. 当前处理吞吐消失；
2. 重新启动副本需要 setup；
3. 中断可能增加任务 JCT 和 DDL 违约风险。

---

## 8. 在线决策过程

### 8.1 决策时点

时间被离散为决策时点

\[
\mathcal T=\{0,1,\ldots,T\},
\]

其中索引 \(t\) 对应实际时间 \(t\Delta\)，相邻时点的实际时间间隔为 \(\Delta\)。第一版取 \(\Delta=10\) 分钟。对实际到达时刻为 \(a_j\) 的任务，定义其首次准入决策时点为

\[
\tau_j
=
\min\{t\in\mathcal T:t\Delta\ge a_j\}.
\]

任务可以在两个决策点之间到达，但在最近的下一个决策点参加准入。资源释放、资源回收、吞吐变化和任务进度也在下一决策点汇总进入系统状态。

事件触发或“固定周期 + 事件触发”的混合决策机制属于后续扩展。

### 8.2 在线可观测信息

在决策时点 \(t\)，调度器观察：

1. 本周期新到达的任务集合；
2. 已接纳但尚未完成的任务；
3. 每个任务的剩余 batch 工作量；
4. 当前吞吐估计；
5. 当前 setup、active、paused 和 failed replica 状态；
6. 每个 Pod 和 replica 的当前放置；
7. 两个资源域中各节点、卡型的当前可用 GPU；
8. 已发生的资源释放和回收结果。

调度器不知道：

1. 未来任务到达；
2. 未来任务参数；
3. 未来节点和 GPU 容量；
4. 未来资源回收；
5. 未来真实处理速度。

历史数据可以用于训练 RL/DRL 模型或估计未来状态价值，但在线算法不能访问真实未来。实验中的 clairvoyant oracle 是唯一允许使用真实未来信息的对照方法。

### 8.3 三类联合决策

每个决策时点联合进行：

1. **Admission control**
   - 对本周期新到达任务作出一次性接受或拒绝决策；
   - 接纳后进入内部任务集合；
   - 被拒绝的当前请求永久离开；
   - 用户调整 DDL、估计卡时或兼容卡型后再次提交时，视为一个全新的任务请求。

2. **Resource orchestration**
   - 选择资源域；
   - 选择兼容 GPU 卡型；
   - 将副本的全部 Gang Pods 放置到满足容量要求的节点。

3. **Replica scaling**
   - 为已接纳任务增加、保持或删除副本；
   - 允许将任务缩容至零；
   - 任意时点的副本总数不超过用户提交的最大副本数。

准入决策不可撤销，但接纳后的资源计划可以在后续决策时点修改。因此，这是：

\[
\text{irrevocable admission}
+
\text{adaptive resource recourse}.
\]

调度器可以生成未来资源计划用于判断当前动作，但只执行当前决策周期的动作。未来计划不构成不可修改的资源预留。

---

## 9. 数学模型

### 9.1 集合与索引

- \(\mathcal J\)：研究周期内到达的任务集合；
- \(\mathcal T\)：离散决策时点集合；
- \(\mathcal D=\{\mathrm{on},\mathrm{off}\}\)：在线与离线资源域；
- \(\mathcal K\)：GPU 卡型集合；
- \(\mathcal N_{dk}\)：资源域 \(d\) 中配备卡型 \(k\) 的节点集合；
- \(\mathcal R_j=\{1,\ldots,\overline r_j\}\)：任务 \(j\) 的候选副本索引；
- \(\mathcal P_{jk}=\{1,\ldots,p_{jk}\}\)：任务 \(j\) 在卡型 \(k\) 上一个副本包含的 Pod 索引。

### 9.2 资源参数与状态

- \(G_n\)：节点 \(n\) 的物理 GPU 容量；
- \(C_{nt}\le G_n\)：时点 \(t\) 环境授予弹性任务的节点 \(n\) GPU 容量预算，其中包含当前弹性副本已经占用的容量；
- \(W_{jt}\in[0,1]\)：时点 \(t\) 任务 \(j\) 的剩余 batch 工作比例；
- \(\widehat\mu_{jkt}\)：时点 \(t\) 任务 \(j\) 在卡型 \(k\) 上的估计单副本吞吐；
- \(\mu_{jkt}\)：执行环境实际实现的单副本吞吐；
- \(L_j=\lceil s_j/\Delta\rceil\)：任务 \(j\) 的 setup 所需决策周期数。

\(C_{nt}\) 和 \(\mu_{jkt}\) 是在线变化且未来未知的环境量。

### 9.3 决策变量

#### 准入

\[
z_j\in\{0,1\},
\]

其中 \(z_j=1\) 表示任务 \(j\) 被接纳。

#### 副本资源域与卡型选择

\[
y_{jrdkt}\in\{0,1\},
\]

其中 \(y_{jrdkt}=1\) 表示任务 \(j\) 的副本 \(r\) 在时点 \(t\) 被部署到资源域 \(d\)、GPU 卡型 \(k\)。该变量同时覆盖 setup 和 active 状态。

#### Pod 放置

\[
x_{jrdkpnt}\in\{0,1\},
\]

其中 \(x_{jrdkpnt}=1\) 表示任务 \(j\) 的副本 \(r\) 中第 \(p\) 个 Pod，在时点 \(t\) 被放置到节点 \(n\in\mathcal N_{dk}\)。

#### 活跃状态

\[
v_{jrdkt}\in\{0,1\},
\]

其中 \(v_{jrdkt}=1\) 表示该副本已经完成 setup，并在时点 \(t\) 实际贡献吞吐。

在完整 MILP 实现中，还可以引入 launch、termination 和 setup-residual 辅助变量，对 setup 和无迁移约束进行线性化。本文档先给出其必要语义，不预先绑定具体线性化方式。

### 9.4 核心约束

#### 9.4.1 准入一致性

只有已接纳任务可以获得资源：

\[
y_{jrdkt}\le z_j,
\qquad
\forall j,r,d,k,t.
\]

对已经接纳的任务，后续不能由调度器撤销准入。

#### 9.4.2 单副本唯一配置

一个副本在同一时点最多位于一个资源域并使用一种 GPU 卡型：

\[
\sum_{d\in\mathcal D}
\sum_{k\in\mathcal K_j}
y_{jrdkt}
\le 1,
\qquad
\forall j,r,t.
\]

#### 9.4.3 最大副本数

任务在所有资源域和卡型上的副本总数不得超过其提交上限：

\[
\sum_{r\in\mathcal R_j}
\sum_{d\in\mathcal D}
\sum_{k\in\mathcal K_j}
y_{jrdkt}
\le
\overline r_j z_j,
\qquad
\forall j,t.
\]

由于最小副本数为零，该约束允许任务排队或暂停。

#### 9.4.4 Gang Pod 完整放置

当副本选择资源域 \(d\) 和卡型 \(k\) 时，其每个 Pod 都必须恰好放置到一个兼容节点：

\[
\sum_{n\in\mathcal N_{dk}}
x_{jrdkpnt}
=
y_{jrdkt},
\qquad
\forall j,r,d,k,p,t.
\]

该约束允许同一副本的多个 Pod 放置到同一节点，但不允许单个 Pod 跨节点拆分。

#### 9.4.5 节点 GPU 容量

\[
\sum_j
\sum_{r\in\mathcal R_j}
\sum_{p\in\mathcal P_{jk}}
g_{jk}x_{jrdkpnt}
\le
C_{nt},
\qquad
\forall d,k,n\in\mathcal N_{dk},t.
\]

setup 和 active 副本都消耗 GPU 容量。

#### 9.4.6 Setup 与活跃状态

活跃副本必须已分配完整资源：

\[
v_{jrdkt}\le y_{jrdkt}.
\]

副本只有在同一资源域、同一卡型和同一组节点上连续占用资源达到 \(L_j\) 个周期后，才能从 setup 转为 active。setup 期间：

\[
v_{jrdkt}=0.
\]

若副本被删除、任意 Pod 被回收，或改变资源域、卡型和节点放置，则其 active 状态失效；未来重新启动时 setup 计时重新开始。

#### 9.4.7 无迁移

对于连续保留的副本，其 Pod 放置保持不变。若放置发生变化，必须先终止原副本，再将新的放置视为一次 scale-up，并重新执行 Gang setup。

该约束将在具体优化算法中通过 placement-retention 或 launch/termination 变量线性化。

### 9.5 任务进度状态转移

实际执行过程中，剩余 batch 工作量按照真实活跃吞吐更新：

\[
W_{j,t+1}
=
\left[
W_{jt}
-
\Delta
\sum_{r\in\mathcal R_j}
\sum_{d\in\mathcal D}
\sum_{k\in\mathcal K_j}
\mu_{jkt}v_{jrdkt}
\right]^+.
\]

setup 副本不贡献吞吐。不同资源域和 GPU 卡型上的活跃副本处理量可以累加。

若副本在两个决策点之间被回收，则 \(\mu_{jkt}\) 表示该周期内实际实现的平均有效吞吐，因此只累计回收前真正完成的工作；下一决策时点再将该副本标记为失效。

在线决策时，调度器不能使用未知的 \(\mu_{jkt}\)，而是使用当前估计 \(\widehat\mu_{jkt}\) 生成计划：

\[
\widehat W_{j,t+1}
=
\left[
W_{jt}
-
\Delta
\sum_{r,d,k}
\widehat\mu_{jkt}v_{jrdkt}
\right]^+.
\]

### 9.6 完成、DDL 与 JCT

任务完成时刻为

\[
C_j
=
\Delta\inf\{t\in\mathcal T: W_{jt}=0\}.
\]

任务的 JCT 从用户提交时刻开始计算：

\[
\operatorname{JCT}_j=C_j-a_j.
\]

JCT 包含：

- 等待最近决策点的时间；
- 准入后的排队时间；
- setup 时间；
- 缩容至零后的暂停时间；
- 实际 batch 处理时间；
- 外生资源回收导致的等待和重新启动时间。

被拒绝任务没有 JCT。实验中的平均与尾部 JCT 在实际完成的任务上统计；尚未完成或被用户取消的任务计为 SLA 未达成，并单独报告未完成率或取消率。为避免有限实验窗口造成选择偏差，trace replay 应在停止接收新任务后继续运行至已接纳任务完成或取消。

任务的 SLA 达成指标定义为

\[
S_j
=
\mathbf 1\{z_j=1,\ C_j\le d_j\}.
\]

因此：

- 被拒绝任务满足 \(S_j=0\)；
- 接纳后超过 DDL 的任务满足 \(S_j=0\)；
- 只有接纳且在 DDL 前完成的任务满足 \(S_j=1\)。

DDL 是高违约成本的业务 SLA，而不是在未知吞吐和未知资源回收下能够绝对保证的确定性约束。任务超过 DDL 后可以继续执行，也可能由用户主动取消；无论后续行为如何，该请求的 SLA 均已判定为未达成。

---

## 10. 优化目标

### 10.1 首要目标：SLA 达成率

所有到达任务都进入 SLA 达成率的分母：

\[
\operatorname{SLA\ Attainment}
=
\frac{\sum_{j\in\mathcal J}S_j}
{|\mathcal J|}.
\]

首要目标为

\[
\max \sum_{j\in\mathcal J} S_j.
\]

该定义同时惩罚拒绝和接纳后超时，避免“拒绝所有任务即可获得 100% SLA”的退化结果。

### 10.2 次要目标：JCT

在 SLA 达成数量相同的情况下，进一步最小化任务 JCT。问题采用字典序目标：

\[
\operatorname{lexicographically\ optimize}
\left(
\max \sum_j S_j,\;
\min \operatorname{MeanJCT},\;
\min \operatorname{TailJCT}
\right).
\]

其中 TailJCT 可在实验中采用 P95、P99 或 CVaR 表示。SLA 的优先级严格高于 JCT。

### 10.3 其他评价指标

以下指标用于解释算法行为和系统代价，不高于 SLA 与 JCT：

- GPU 资源占用率与有效利用率；
- 两个资源域和各卡型的利用情况；
- 任务拒绝率；
- 副本 scale-up、scale-down 和 setup 次数；
- setup 占用的 GPU-hours；
- 资源回收后的恢复时间；
- 节点碎片；
- 在线决策和组合优化求解时间。

---

## 11. 在线学习问题的本质

当前决策会改变未来系统状态：

- 接纳任务会消耗未来 GPU 容量；
- 扩容可加速任务完成并提前释放未来资源；
- 扩容也会立即占用资源，并产生 setup 空窗；
- 暂停任务可释放当前资源，但消耗其 DDL slack；
- 将副本放在易被回收的节点上可能增加未来重启风险；
- 使用某一卡型会改变其他兼容任务未来可使用的异构容量。

调度器必须在未知未来到达、容量和吞吐下权衡：

\[
\text{current acceleration}
\quad\text{vs.}\quad
\text{future capacity opportunity}.
\]

这使问题具有序贯决策和长期信用分配结构，适合使用 RL/DRL 学习未来状态价值或资源机会成本。

同时，完整动作包含准入、资源域选择、卡型选择、Gang Pod 放置和副本伸缩，动作空间巨大，并包含大量不可行动作。因此，端到端 RL 直接输出完整调度方案并不现实。

---

## 12. Learning + Optimization 方法边界

论文方法方向确定为：

\[
\boxed{
\text{RL/DRL-based learning}
+
\text{combinatorial optimization}
}.
\]

当前只固定职责边界，不预先确定具体算法。

### 12.1 Learning 模块负责

候选学习对象包括：

- 长期状态价值 \(V_\theta(s_t)\)；
- 异构 GPU 未来容量的机会成本或影子价格；
- 任务准入、扩容和暂停的长期优先级；
- 不同资源域和节点的回收风险价值；
- 组合优化器的候选动作或搜索引导信号。

具体选择需要通过文献调研、数据分析和 gap experiments 决定。

### 12.2 Optimization 模块负责

组合优化器负责把 learning 输出转化为当前可执行的调度动作，并处理：

- GPU 卡型兼容；
- 节点容量；
- 小数与整数 GPU 需求；
- 最大副本总数；
- Gang Pod 完整放置；
- setup 和 active 状态；
- 无迁移；
- 当前任务进度与 DDL 风险。

优化器可以采用 MILP、CP-SAT、分解算法、列生成、局部搜索或专用启发式；此处暂不锁定。

### 12.3 吞吐估计器的边界

实时吞吐估计器为调度器提供 \(\widehat\mu_{jkt}\)，但不是第一版论文的主要 learning contribution。论文重点研究如何在给定动态估计的情况下进行长期在线资源决策。

---

## 13. 明确不属于第一版的问题

第一版暂不研究：

1. 完成时间 P95/P99 预测器及其概率校准；
2. 公司内部资源回收、驱逐和 victim-selection 算法；
3. 多队列 Max-min、公平性和优先级机制；
4. CPU、内存、网络带宽、机架和跨机通信拓扑；
5. 模型切分、张量并行或流水并行方案选择；
6. 一个 replica 跨越在线与离线两个资源域；
7. 运行中无成本迁移；
8. checkpoint 和大规模工作量回滚；
9. 用户如何修改参数并重新提交的策略行为；
10. 用户之间的博弈、定价和机制设计；
11. 秒级在线服务 autoscaling；
12. 端到端 RL 直接生成全部 Pod-to-node 决策；
13. 事件触发决策机制；
14. 完整生产系统上线、故障恢复和所有控制面机制。

这些问题可以作为后续扩展，但不应进入第一版核心数学模型。

---

## 14. 问题的紧凑学术表述

### 中文

本文研究高负载异构 GPU 集群中弹性离线推理任务的在线动态资源编排。任务在线到达，并提交估计标准卡时、完成期限、兼容 GPU 卡型集合、最大副本数及固定 Gang Pod 配置。统一调度器周期性联合决定任务准入、资源域与卡型选择、节点级 Gang 放置和副本扩缩容。GPU 可用容量和任务真实处理速度均随时间变化且仅在线揭露；新增副本还会占用资源并经历 setup。研究目标是在所有到达请求上首先最大化 DDL SLA 达成率，其次降低平均与尾部 JCT。方法上采用 RL/DRL 学习未知未来的长期价值，并由组合优化器生成满足当前物理约束的可执行决策。

### English

We study online orchestration of elastic offline-inference jobs in high-load heterogeneous GPU clusters. Jobs arrive over time with an estimated normalized GPU-hour demand, a completion deadline, a set of compatible GPU types, a maximum replica count, and a fixed gang-pod configuration. At each decision epoch, a unified scheduler jointly determines admission, resource-domain and GPU-type selection, gang placement, and replica scaling. Available GPU capacity and true processing rates evolve over time and are only revealed online, while newly launched replicas occupy resources during setup before becoming productive. The primary objective is to maximize deadline-SLA attainment over all arrivals, followed by minimizing mean and tail job completion time. We pursue a learning-plus-optimization approach in which RL/DRL learns the long-term value of future capacity and a combinatorial optimizer produces executable decisions under physical scheduling constraints.

---

## 15. 后续工作顺序

在本文档获得确认后，后续工作按以下顺序开展：

1. 使用统一模板系统阅读相关文献，核实问题级 novelty；
2. 将上述语义整理为论文的 formal problem formulation；
3. 建立 deterministic rolling-horizon optimization 基线；
4. 建立 trace-driven simulator 和 clairvoyant oracle；
5. 测量 elasticity、foresight、heterogeneity 和 setup gap；
6. 根据 gap 结果确定 RL 的学习对象；
7. 设计 learning + optimization 算法；
8. 将问题定义、相关工作和方法持续沉淀到投稿论文。
