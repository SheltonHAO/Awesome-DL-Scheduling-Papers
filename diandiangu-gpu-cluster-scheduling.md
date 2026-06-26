# GPU Cluster Scheduling | 论文与代码阅读笔记

Source: https://diandiangu.github.io/scheduling/

<div id="page"><div id="article-container"><h2 id="集群调度"><a href="#集群调度" class="headerlink" title="集群调度"></a>集群调度</h2><h3 id="Survey"><a href="#Survey" class="headerlink" title="Survey"></a>Survey</h3><ul>
<li><input disabled="" type="checkbox"> <strong>Deep Learning Workload Scheduling in GPU Datacenters: Taxonomy, Challenges and Vision</strong> (ArXiv’22) [<a target="_blank" rel="noopener" href="https://arxiv.org/abs/2205.11913">PDF</a>][<a target="_blank" rel="noopener" href="https://github.com/S-Lab-System-Group/Awesome-DL-Scheduling-Papers">Awesome list</a>]</li>
</ul>
<h3 id="集群数据分析"><a href="#集群数据分析" class="headerlink" title="集群数据分析"></a>集群数据分析</h3><ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads</strong> (ATC’19) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/atc19-jeon.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/msr-fiddle/philly-traces">Dataset</a>] [<a href="https://diandiangu.github.io/2020/09/21/atc19jeon/">阅读笔记</a>]<ul>
<li>微软Philly集群衍生出的一系列工作之一，分析了多租户GPU集群中的workload的特点，可以为各类工作提供motivation</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters</strong> (NSDI’22) [<a target="_blank" rel="noopener" href="https://qzweng.github.io/files/2022NSDI-MLaaS-Weng.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/alibaba/clusterdata">Dataset</a>]</li>
</ul>
<h3 id="GPU集群调度"><a href="#GPU集群调度" class="headerlink" title="GPU集群调度"></a>GPU集群调度</h3><ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Gandiva: Introspective Cluster Scheduling for Deep Learning</strong> (OSDI’18) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/osdi18-xiao.pdf">PDF</a>]<ul>
<li>微软的调度器，可以通过job migration, time slicing, grown-shrink, packing等方式充分利用集群资源并减少fragmentation</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters</strong></li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Tiresias: A GPU Cluster Manager for Distributed Deep Learning</strong> (NSDI’19) [<a target="_blank" rel="noopener" href="https://cloud.tencent.com/developer/article/1408935?from=information.detail.attained">阅读笔记(outer link)</a>] [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/nsdi19-gu.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/SymbioticLab/Tiresias/">Code</a>] <ul>
<li>密歇根大学与微软、阿里巴巴、UNIST、字节跳动合作的开源GPU集群调度器</li>
<li>Discretized-2DAS（age-based）scheduler: 无需对DL任务全面的信息就可以减小JCT；model profile placement：无需来自用户的额外信息就可以把任务放在合适的位置，提高集群资源利用率且不会影响模型的performance；可以在避免starvation和最小化平均JCT上寻找trade off</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>DL2: A Deep Learning-driven Scheduler for Deep Learning Clusters</strong> (ArXiv’19) [<a target="_blank" rel="noopener" href="https://arxiv.org/pdf/1909.06040.pdf">PDF</a>]<ul>
<li>香港大学与阿里巴巴合作的工作，运用了强化学习，可以实现动态worker/parameter server调整，可以实现<strong>弹性训练</strong></li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning</strong> (EuroSys’20) [<a href="https://diandiangu.github.io/2021/03/07/Gandiva-fair/">阅读笔记</a>] [<a target="_blank" rel="noopener" href="https://www.microsoft.com/en-us/research/uploads/prod/2020/05/gandiva-fair-eurosys20.pdf">PDF</a>]<ul>
<li>微软印度的工作，异构设备的调度，可以为一个训练任务分配异构的GPU</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Themis: Fair and Efficient GPU Cluster Scheduling</strong> (NSDI’20) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/nsdi20-paper-mahajan.pdf">PDF</a>]<ul>
<li>威斯康星大学和微软合作的GPU调度器</li>
<li>应用场景为一个DL app包含多个深度学习模型调参的job，提出了Finish-time fairness这样的一个新的metric</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications</strong></li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads</strong> (OSDI’20) [<a target="_blank" rel="noopener" href="https://cs.stanford.edu/~matei/papers/2020/osdi_gavel.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/stanford-futuredata/gavel">Code</a>]<ul>
<li>微软和斯坦福合作，实现了调度器Gavel，在资源调度的时候考虑到了不同任务在不同GPU设备上的实际效率，异构是针对cluster而言的，在一个任务里不能有不同型号的GPU</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications</strong> (OSDI’20) [<a href="https://diandiangu.github.io/2020/12/07/PipeSwitch/">阅读笔记</a>] [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/osdi20-bai.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/netx-repo/PipeSwitch">Code</a>]<ul>
<li>通过训练和上下文切换的pipeline式overlap，来达到高效的上下文切换</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees</strong> (OSDI’20) [<a href="https://diandiangu.github.io/2020/12/22/HiveD/">阅读笔记</a>] [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/osdi20-zhao_hanyu.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/microsoft/hivedscheduler">Code</a>]<ul>
<li>MSRA与北京大学合作，抽象集群中的GPU资源，保证“共享安全”</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>AntMan: Dynamic Scaling on GPU Clusters for Deep Learning</strong> (OSDI’20) [<a href="https://diandiangu.github.io/2020/12/02/AntMan-0/">阅读笔记</a>)] [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/osdi20-xiao.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/alibaba/GPU-scheduler-for-deep-learning">Code</a>]<ul>
<li>GPU集群中内存和op的细粒度调度，可以提升集群的利用率</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>Job Scheduling for Large-Scale Machine Learning Clusters</strong> (CoNEXT’20) [<a target="_blank" rel="noopener" href="https://dl.acm.org/doi/pdf/10.1145/3386367.3432588">PDF</a>]<ul>
<li>弗吉尼亚大学的工作，实现了调度器MLFS，设计了对于数据并行、模型并行、以及混合情况下都适用的调度，并采用了强化学习的方法学习调度策略</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Optimizing Distributed Training Deployment in Heterogeneous GPU Clusters</strong> (CoNEXT’20) [<a target="_blank" rel="noopener" href="https://i.cs.hku.hk/~cwu/papers/xdyi-conext20.pdf">PDF</a>]<ul>
<li>香港大学和阿里巴巴的合作，实现了调度模块HeteroG</li>
<li>一个任务中可能包含多种异构的GPU，通过强化学习的方法学习parallelism的最佳策略</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Elan: Towards Generic and Efficient Elastic Training for Deep Learning</strong> (ICDCS’20 <strong>Best Paper Honorable Mention Award</strong>) [<a href="https://diandiangu.github.io/2021/03/07/Elan/">阅读笔记</a>] [<a target="_blank" rel="noopener" href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&amp;arnumber=9355755">PDF</a>]<ul>
<li>清华大学与商汤合作的工作，实现了一个面向深度学习负载的弹性调度系统，可以提高运行深度学习负载的数据中心资源利用率</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Elastic Resource Sharing for Distributed Deep Learning</strong> (NDSI’21) [<a href="https://diandiangu.github.io/2021/04/13/AFS/">阅读笔记</a>] [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/nsdi21-hwang.pdf">PDF</a>]<ul>
<li>KAIST的工作，提出了算法AFS，设计了一个需要elastic DL的调度算法，并实现了高效调度的系统CoDDL（基于<a target="_blank" rel="noopener" href="https://chhwang.github.io/pubs/osdi18_poster_hwang.pdf">OSDI’18的一篇poster</a>）</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning</strong> (OSDI’21) [<a target="_blank" rel="noopener" href="https://arxiv.org/pdf/2008.12260v2.pdf">PDF</a>] [<a target="_blank" rel="noopener" href="https://github.com/petuum/adaptdl">Code</a>]<ul>
<li>Petuum磐腾科技提出的GPU集群调度方法</li>
<li>可以在训练时动态调整资源数量，为深度学习训练的不同阶段找到合适的batch size，从而使得训练效率和训练的准确率都能够得到保障</li>
<li>没有考虑到模型并行，且为了不同任务之间不互相影响，需要满足一个节点上不能有两个不同的任务，这样需要集群中资源非常充足。现实中可能会有很多集群不能满足这样的需求</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs</strong> (SoCC’21) [<a target="_blank" rel="noopener" href="https://dl.acm.org/doi/pdf/10.1145/3472883.3486978">PDF</a>]<ul>
<li>针对DL训练任务的SLO进行调度，将任务分为SLO jobs和Best-Effort jobs两类</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters</strong> (OSDI’22) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/osdi22-mohan.pdf">PDF</a>]<ul>
<li>这篇文章来自Microsoft Research。该工作提供了一种面向多租户集群的DNN任务调度器，在传统调度器的基础上将任务对CPU及内存资源分配的敏感度纳入考虑，从而更好地分配和利用现有的集群资源，提高平均任务完成时间。</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>Lucid: A Non-intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs</strong> (ASPLOS’23) [<a target="_blank" rel="noopener" href="https://dl.acm.org/doi/pdf/10.1145/3575693.3575705">PDF</a>]<ul>
<li>解决现有的深度学习作业调度器存在可扩展性有限、决策过程不透明等等问题</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> ElasticFlow (ASPLOS’23)</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> Sia (SOSP’23)</li>
</ul>
<h3 id="传统集群调度"><a href="#传统集群调度" class="headerlink" title="传统集群调度"></a>传统集群调度</h3><ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Doninant Resource Fairness: Fair Allocation of Multiple Resource Types</strong> (NSDI’11) [<a target="_blank" rel="noopener" href="https://people.eecs.berkeley.edu/~alig/papers/drf.pdf">PDF</a>]<ul>
<li>伯克利提出的Dominant Resoruce Fairness(DRF)算法，多资源分配的max-min fariness</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Improving Service Level Agreements for a Job Scheduler by Visualizing Simulations</strong> (MIT thesis) [<a target="_blank" rel="noopener" href="https://dspace.mit.edu/bitstream/handle/1721.1/76829/824133510-MIT.pdf">PDF</a>]<ul>
<li>“by deadline” SLA</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Apollo: Scalable and Coordinated Scheduling for Cloud-Scale Computing</strong> (OSDI’14) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-boutin_0.pdf">PDF</a>]<ul>
<li>微软提出的在集群中处理大量数据处理的产业级别调度框架</li>
<li>每个数据处理job可以用task组成的DAG来表示；每个job有单独的sheduler，用于调度每个job内部的的task</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>HUG: Multi-Resource Fairness for Correlated and Elastic Demands</strong> (NSDI’16) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/conference/nsdi16/nsdi16-paper-chowdhury.pdf">PDF</a>]<ul>
<li>将传统的maxmin fairness推广到需求相关的多资源环境，将DRF推广到弹性需求</li>
</ul>
</li>
</ul>
<ul>
<li><input disabled="" type="checkbox"> <strong>Altruistic Scheduling in Multi-Resource Clusters</strong> (OSDI’16)[<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/conference/osdi16/osdi16-grandl-altruistic.pdf">PDF</a>]</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Scheduling Mix-flows in Commodity Datacenters with Karuna</strong> (SIGCOMM’16) [<a target="_blank" rel="noopener" href="https://people.csail.mit.edu/alizadeh/papers/karuna-sigcomm16.pdf">PDF</a>]<ul>
<li>尽量保证有deadline的网络流的deadline，同时优化没有deadline的网络流的flow completion time</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Swayam: Distributed Autoscaling to Meet SLAs of Machine Learning Inference Services with Resource Efficiency</strong> (Middleware’17) [<a target="_blank" rel="noopener" href="https://www.microsoft.com/en-us/research/uploads/prod/2018/01/2017.Middleware.Swayam.TailLatencyInAzureML.pdf">PDF</a>]<ul>
<li>Meet SLA for inference jobs</li>
</ul>
</li>
</ul>
<ul>
<li><input checked="" disabled="" type="checkbox"> <strong>Providing SLOs for Resource-Harvesting VMs in Cloud Platforms</strong> (OSDI’20) [<a target="_blank" rel="noopener" href="https://www.usenix.org/system/files/osdi20-paper-ambati.pdf">PDF</a>]<ul>
<li>传统的云计算调度，不同的是，本文考虑了SLO（65% of the Harvest VMs will survive more than a week）<br><a target="_blank" rel="noopener" href="https://www2.seas.gwu.edu/~tlan/papers/NFS_INFOCOM_2015.pdf">https://www2.seas.gwu.edu/~tlan/papers/NFS_INFOCOM_2015.pdf</a><br><a target="_blank" rel="noopener" href="https://iqua.ece.toronto.edu/papers/lchen-infocom16.pdf">https://iqua.ece.toronto.edu/papers/lchen-infocom16.pdf</a></li>
</ul>
</li>
</ul>
</div><hr><div id="post-comment"><div class="comment-head"><div class="comment-headline"><i class="fas fa-comments fa-fw"></i><span> 评论</span></div></div><div class="comment-wrap"><div><div class="vcomment" id="vcomment"></div></div></div></div></div><div class="aside_content" id="aside_content"><div class="card-widget card-info"><div class="card-content"><div class="card-info-avatar is-center"><img class="avatar-img" src="/img/favicon.jpeg" onerror="this.onerror=null;this.src='/img/friend_404.gif'" alt="avatar"><div class="author-info__name">GDD</div><div class="author-info__description"></div></div><div class="card-info-data"><div class="card-info-data-item is-center"><a href="/archives/"><div class="headline">文章</div><div class="length-num">29</div></a></div><div class="card-info-data-item is-center"><a href="/tags/"><div class="headline">标签</div><div class="length-num">34</div></a></div></div><a class="button--animated" id="card-info-btn" target="_blank" rel="noopener" href="https://github.com/gudiandian"><i class="fab fa-github"></i><span>Follow Me on GitHub</span></a></div></div><div class="sticky_layout"><div class="card-widget card-recent-post"><div class="card-content"><div class="item-headline"><i class="fas fa-history"></i><span>最新文章</span></div><div class="aside-list"><div class="aside-list-item"><a class="thumbnail" href="/2024/05/31/DistFlashAtten/" title="DistFlashAtten：面向长上下文大语言模型训练的内存高效的分布式注意力机制"><img src="https://cdn.jsdelivr.net/npm/butterfly-extsrc@1/img/default.jpg" onerror="this.onerror=null;this.src='/img/404.jpg'" alt="DistFlashAtten：面向长上下文大语言模型训练的内存高效的分布式注意力机制"></a><div class="content"><a class="title" href="/2024/05/31/DistFlashAtten/" title="DistFlashAtten：面向长上下文大语言模型训练的内存高效的分布式注意力机制">DistFlashAtten：面向长上下文大语言模型训练的内存高效的分布式注意力机制</a><time datetime="2024-05-31T09:11:50.000Z" title="发表于 2024-05-31 17:11:50">2024-05-31</time></div></div><div class="aside-list-item"><a class="thumbnail" href="/2023/09/25/BPipe/" title="BPipe: 面向大语言模型训练的内存均衡的流水线并行"><img src="/img/BPipe/3.png" onerror="this.onerror=null;this.src='/img/404.jpg'" alt="BPipe: 面向大语言模型训练的内存均衡的流水线并行"></a><div class="content"><a class="title" href="/2023/09/25/BPipe/" title="BPipe: 面向大语言模型训练的内存均衡的流水线并行">BPipe: 面向大语言模型训练的内存均衡的流水线并行</a><time datetime="2023-09-25T10:52:07.000Z" title="发表于 2023-09-25 18:52:07">2023-09-25</time></div></div><div class="aside-list-item"><a class="thumbnail" href="/2023/04/25/Lucid/" title="Lucid：一个可扩展、可解释的实用型深度学习作业调度器"><img src="/img/Lucid/cover.png" onerror="this.onerror=null;this.src='/img/404.jpg'" alt="Lucid：一个可扩展、可解释的实用型深度学习作业调度器"></a><div class="content"><a class="title" href="/2023/04/25/Lucid/" title="Lucid：一个可扩展、可解释的实用型深度学习作业调度器">Lucid：一个可扩展、可解释的实用型深度学习作业调度器</a><time datetime="2023-04-25T03:12:39.000Z" title="发表于 2023-04-25 11:12:39">2023-04-25</time></div></div><div class="aside-list-item"><a class="thumbnail" href="/2022/11/13/Pheromone/" title="Pheromone：服务器无感知计算平台中以数据中为心的函数编排"><img src="/img/Pheromone/2.png" onerror="this.onerror=null;this.src='/img/404.jpg'" alt="Pheromone：服务器无感知计算平台中以数据中为心的函数编排"></a><div class="content"><a class="title" href="/2022/11/13/Pheromone/" title="Pheromone：服务器无感知计算平台中以数据中为心的函数编排">Pheromone：服务器无感知计算平台中以数据中为心的函数编排</a><time datetime="2022-11-13T15:19:48.000Z" title="发表于 2022-11-13 23:19:48">2022-11-13</time></div></div><div class="aside-list-item"><a class="thumbnail" href="/2022/11/04/TOPOOPT/" title="TOPOOPT：面向分布式训练作业的网络拓扑与并行策略协同优化"><img src="/img/TOPOOPT/1.png" onerror="this.onerror=null;this.src='/img/404.jpg'" alt="TOPOOPT：面向分布式训练作业的网络拓扑与并行策略协同优化"></a><div class="content"><a class="title" href="/2022/11/04/TOPOOPT/" title="TOPOOPT：面向分布式训练作业的网络拓扑与并行策略协同优化">TOPOOPT：面向分布式训练作业的网络拓扑与并行策略协同优化</a><time datetime="2022-11-04T07:52:35.000Z" title="发表于 2022-11-04 15:52:35">2022-11-04</time></div></div></div></div></div><div class="card-widget card-tags"><div class="card-content"><div class="item-headline"><i class="fas fa-tags"></i><span>标签</span></div><div class="card-tag-cloud"><a href="/tags/ASPLOS/" style="font-size: 1.1em; color: #999">ASPLOS</a> <a href="/tags/ATC/" style="font-size: 1.1em; color: #999">ATC</a> <a href="/tags/EuroSys/" style="font-size: 1.1em; color: #999">EuroSys</a> <a href="/tags/ICDCS/" style="font-size: 1.1em; color: #999">ICDCS</a> <a href="/tags/ICML/" style="font-size: 1.1em; color: #999">ICML</a> <a href="/tags/MLSys/" style="font-size: 1.16em; color: #999b9e">MLSys</a> <a href="/tags/NSDI/" style="font-size: 1.27em; color: #99a0a9">NSDI</a> <a href="/tags/OSDI/" style="font-size: 1.33em; color: #99a2af">OSDI</a> <a href="/tags/SIGCOMM/" style="font-size: 1.1em; color: #999">SIGCOMM</a> <a href="/tags/SOSP/" style="font-size: 1.21em; color: #999ea4">SOSP</a> <a href="/tags/serverless/" style="font-size: 1.1em; color: #999">serverless</a> <a href="/tags/%E4%B8%8A%E4%B8%8B%E6%96%87%E5%88%87%E6%8D%A2/" style="font-size: 1.1em; color: #999">上下文切换</a> <a href="/tags/%E4%BB%BB%E5%8A%A1%E8%B0%83%E5%BA%A6/" style="font-size: 1.1em; color: #999">任务调度</a> <a href="/tags/%E5%87%BD%E6%95%B0%E7%BC%96%E6%8E%92/" style="font-size: 1.1em; color: #999">函数编排</a> <a href="/tags/%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F/" style="font-size: 1.1em; color: #999">分布式系统</a> <a href="/tags/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/" style="font-size: 1.44em; color: #99a7ba">分布式训练</a> <a href="/tags/%E5%A4%A7%E6%A8%A1%E5%9E%8B/" style="font-size: 1.1em; color: #999">大模型</a> <a href="/tags/%E5%AE%9E%E8%AF%81%E7%A0%94%E7%A9%B6/" style="font-size: 1.1em; color: #999">实证研究</a> <a href="/tags/%E5%AE%B9%E9%94%99/" style="font-size: 1.1em; color: #999">容错</a> <a href="/tags/%E5%BC%B9%E6%80%A7%E8%AE%AD%E7%BB%83/" style="font-size: 1.21em; color: #999ea4">弹性训练</a> <a href="/tags/%E6%8E%A8%E7%90%86%E7%B3%BB%E7%BB%9F/" style="font-size: 1.16em; color: #999b9e">推理系统</a> <a href="/tags/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%B3%BB%E7%BB%9F/" style="font-size: 1.5em; color: #99a9bf">机器学习系统</a> <a href="/tags/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/" style="font-size: 1.16em; color: #999b9e">深度学习</a> <a href="/tags/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6/" style="font-size: 1.27em; color: #99a0a9">深度学习框架</a> <a href="/tags/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8/" style="font-size: 1.1em; color: #999">深度学习编译器</a> <a href="/tags/%E7%BB%88%E7%AB%AF/" style="font-size: 1.1em; color: #999">终端</a> <a href="/tags/%E7%BB%BF%E8%89%B2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/" style="font-size: 1.1em; color: #999">绿色机器学习</a> <a href="/tags/%E7%BD%91%E7%BB%9C/" style="font-size: 1.1em; color: #999">网络</a> <a href="/tags/%E7%BD%91%E7%BB%9C%E9%80%9A%E4%BF%A1/" style="font-size: 1.1em; color: #999">网络通信</a> <a href="/tags/%E8%83%BD%E8%80%97/" style="font-size: 1.1em; color: #999">能耗</a> <a href="/tags/%E8%B0%83%E7%A0%94/" style="font-size: 1.16em; color: #999b9e">调研</a> <a href="/tags/%E8%BD%AF%E4%BB%B6%E5%B7%A5%E7%A8%8B/" style="font-size: 1.1em; color: #999">软件工程</a> <a href="/tags/%E9%80%9A%E4%BF%A1%E8%B0%83%E5%BA%A6/" style="font-size: 1.16em; color: #999b9e">通信调度</a> <a href="/tags/%E9%9B%86%E7%BE%A4%E8%B0%83%E5%BA%A6/" style="font-size: 1.39em; color: #99a4b4">集群调度</a></div></div></div><div class="card-widget card-archives"><div class="card-content"><div class="item-headline"><i class="fas fa-archive"></i><span>归档</span></div><ul class="card-archive-list"><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2024/05/"><span class="card-archive-list-date">五月 2024</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2023/09/"><span class="card-archive-list-date">九月 2023</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2023/04/"><span class="card-archive-list-date">四月 2023</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2022/11/"><span class="card-archive-list-date">十一月 2022</span><span class="card-archive-list-count">2</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2022/02/"><span class="card-archive-list-date">二月 2022</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2022/01/"><span class="card-archive-list-date">一月 2022</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2021/11/"><span class="card-archive-list-date">十一月 2021</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item"><a class="card-archive-list-link" href="/archives/2021/09/"><span class="card-archive-list-date">九月 2021</span><span class="card-archive-list-count">1</span></a></li><li class="card-archive-list-item more is-center"><a class="card-archive-list-link-more" href="/archives/">
              <span>查看更多</span><i class="fas fa-angle-right"></i></a></li></ul></div></div><div class="card-widget card-webinfo"><div class="card-content"><div class="item-headline"><i class="fas fa-chart-line"></i><span>网站资讯</span></div><div class="webinfo"><div class="webinfo-item"><div class="item-name">文章数目 :</div><div class="item-count">29</div></div><div class="webinfo-item"><div class="item-name">最后更新时间 :</div><div class="item-count" id="last-push-date" data-lastpushdate="2024-06-09T10:10:22.149Z">2024-6-9</div></div></div></div></div></div></div>
