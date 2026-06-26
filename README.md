# Awesome-DL-Scheduling-Papers

A personal reading map for cluster scheduling, deep learning scheduling, and adjacent AI Infra systems.

This repository starts from two external sources and reorganizes them into a structure that is easier to extend with personal reading notes:

- A cluster scheduling blog map: [GPU Cluster Scheduling | 论文与代码阅读笔记](https://diandiangu.github.io/scheduling/)
- The S-Lab awesome list: [S-Lab-System-Group/Awesome-DL-Scheduling-Papers](https://github.com/S-Lab-System-Group/Awesome-DL-Scheduling-Papers)

## Repository Structure

- [maps/cluster-scheduling-from-diandiangu.md](./maps/cluster-scheduling-from-diandiangu.md)
  - A categorized reading map extracted from the diandiangu blog.
- [references/s-lab-awesome-dl-scheduling-papers.md](./references/s-lab-awesome-dl-scheduling-papers.md)
  - A mirrored markdown reference from the S-Lab awesome list.

## Suggested Reading Paths

### 1. Workload Analysis First

Start here if the goal is to understand real cluster behavior before discussing algorithms.

- Philly / ATC 2019
- MLaaS in the Wild / NSDI 2022

### 2. Core DL Training Scheduler Line

Start here if the goal is training-job scheduling and elastic allocation.

- Gandiva
- Tiresias
- Themis
- Gavel
- Pollux
- Chronus
- Lucid
- Sia

### 3. Classical Cluster Scheduling Background

Start here if the goal is fairness, multi-resource allocation, and SLA-oriented cluster design.

- DRF
- Apollo
- HUG
- Swayam

### 4. Inference Scheduling and Serving

Use the S-Lab reference list as the starting index.

## Notes

This repo is currently a bootstrap version. The next natural step is to add:

- personal tags such as `elasticity`, `fairness`, `heterogeneity`, `deadline`, `serving`
- paper-by-paper notes
- links to code, traces, and benchmark artifacts
- a more explicit map for cluster scheduling vs. model serving vs. LLM serving
