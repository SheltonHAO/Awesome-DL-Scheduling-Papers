# Cluster Scheduling Reading Map from diandiangu

Source page: [GPU Cluster Scheduling | 论文与代码阅读笔记](https://diandiangu.github.io/scheduling/)

This file is a structured reading map extracted and lightly reorganized from the source page on 2026-06-26. It is intended as a personal reading index rather than a verbatim mirror.

## Survey

- Deep Learning Workload Scheduling in GPU Datacenters: Taxonomy, Challenges and Vision (arXiv 2022)

## Cluster Workload Analysis

- Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads (ATC 2019)
- MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters (NSDI 2022)

## GPU Cluster Scheduling

### Early and Foundational Systems

- Gandiva: Introspective Cluster Scheduling for Deep Learning (OSDI 2018)
- Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters (EuroSys 2018)
- Tiresias: A GPU Cluster Manager for Distributed Deep Learning (NSDI 2019)
- DL2: A Deep Learning-driven Scheduler for Deep Learning Clusters (arXiv 2019)

### Fairness, Heterogeneity, and Sharing

- Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning (EuroSys 2020)
- Themis: Fair and Efficient GPU Cluster Scheduling (NSDI 2020)
- Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications (MLSys 2020)
- Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads (OSDI 2020)
- HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees (OSDI 2020)
- Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters (OSDI 2022)

### Elasticity and Dynamic Scaling

- AntMan: Dynamic Scaling on GPU Clusters for Deep Learning (OSDI 2020)
- Elan: Towards Generic and Efficient Elastic Training for Deep Learning (ICDCS 2020)
- Elastic Resource Sharing for Distributed Deep Learning (NSDI 2021)
- Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning (OSDI 2021)
- ElasticFlow (ASPLOS 2023)
- Sia (SOSP 2023)

### Placement, Parallelism, and Optimization

- PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications (OSDI 2020)
- Job Scheduling for Large-Scale Machine Learning Clusters (CoNEXT 2020)
- Optimizing Distributed Training Deployment in Heterogeneous GPU Clusters (CoNEXT 2020)
- Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs (SoCC 2021)
- Lucid: A Non-intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs (ASPLOS 2023)

## Classical Cluster Scheduling Background

- Dominant Resource Fairness: Fair Allocation of Multiple Resource Types (NSDI 2011)
- Improving Service Level Agreements for a Job Scheduler by Visualizing Simulations (MIT thesis)
- Apollo: Scalable and Coordinated Scheduling for Cloud-Scale Computing (OSDI 2014)
- HUG: Multi-Resource Fairness for Correlated and Elastic Demands (NSDI 2016)
- Altruistic Scheduling in Multi-Resource Clusters (OSDI 2016)
- Scheduling Mix-flows in Commodity Datacenters with Karuna (SIGCOMM 2016)
- Swayam: Distributed Autoscaling to Meet SLAs of Machine Learning Inference Services with Resource Efficiency (Middleware 2017)
- Providing SLOs for Resource-Harvesting VMs in Cloud Platforms (OSDI 2020)

## How I Plan to Use This Map

- Use workload analysis papers first to understand real traces and bottlenecks.
- Use the GPU scheduler line to study elasticity, fairness, heterogeneity, and deadline tradeoffs.
- Use the classical cluster scheduling papers as background for resource allocation and SLA ideas.
- Cross-check training-focused systems with inference and serving papers when moving toward AI Infra or LLM scheduling.
