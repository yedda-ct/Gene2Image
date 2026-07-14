# 可学习通路掩码嵌入的基因到病理图像生成 Idea Report
> 生成时间：2026-06-06 | 状态：PENDING_REVIEW

---

## Part 1 Topic Overview

### 1 Motivation

从转录组数据生成组织病理图像是一个新兴的逆问题：给定细胞的基因表达谱，合成与之对应的 H&E 染色形态。该任务为虚拟染色、基因扰动的形态学模拟、罕见样本的数据增广以及疾病机制研究提供了全新的计算工具 [1]。空间转录组（Spatial Transcriptomics, ST）技术（如 10x Xenium）首次在单细胞分辨率上将转录组与配对的组织形态对齐，使这一逆映射的端到端学习成为可能 [1][11]。GeneFlow [1] 是该方向的开创性工作，它将注意力式 RNA 编码器与整流流（Rectified Flow）条件 UNet 结合，首次从单细胞基因表达生成高分辨率 H&E/DAPI 图像，并通过病理学家盲评验证了生成图像的诊断可用性。

然而，这一任务的核心瓶颈在于 RNA 编码器：如何从高维（约 300–5000 基因）、稀疏、带噪声的原始表达谱中，提取出对图像形态具有判别力、可解释、且能跨测序平台泛化的条件信号。现有方法在此存在两类相反的缺陷。一方面，GeneFlow [1] 的编码器将基因视为相互独立的输入特征，**完全不注入任何通路结构先验**——作者明确指出其架构 "without any explicit biological knowledge encoded"，并在 future work 中将"引入基因表达的生物结构/基础模型"列为应对高维 panel 的关键方向。另一方面，作为该任务最新的基础模型，MUPAD [2] 虽引入了通路信息，却采用**固定的 ssGSEA 式富集打分**——它复用 SurvPath [16] 预定义的通路签名，将数万基因压缩为 331 维标量分数后再注入扩散模型。这一压缩步骤独立于生成目标、不可学习，构成了一个信息论意义上的硬瓶颈：任何打分算法未保留的基因级变异在条件中被永久丢弃。

> 直白地说：GeneFlow 把两万个基因一股脑丢给模型自己消化，结构太松、信息过载；MUPAD 则先用一个固定公式把基因压成几百个分数再喂给模型，结构太死、信息被提前掐断。两者恰好是两个极端，中间缺一个"既有生物结构约束、又能随生成任务自我调整"的编码方式。

通路（pathway）天然提供了介于这两个极端之间的结构：它把数万基因组织为数百个功能模块，既引入了稀疏的结构化约束，又保留了模块内的基因级信息。在判别任务中，P-NET [17]、SurvPath [16]、TOSICA [3] 等工作已证明通路结构能同时提升性能与可解释性。但在**生成任务**中，"如何将通路结构注入条件信号"这一问题尚未被系统研究，且一个更根本的疑问悬而未决——通路结构带来的收益究竟源于真实的生物学语义，还是仅仅源于它引入的结构化稀疏？Sparsity is All You Need [4] 在判别任务上发现"随机通路掩码与真实通路性能相当"，使这一问题更显尖锐。

**本研究的必要性：**

- **应用必要性**：基因到图像生成的临床与生物落地，要求条件编码器既准确又可解释——使用者需要知道"是哪条通路的异常驱动了这一形态变化"，而非黑箱输出。GeneFlow [1] 的无结构编码器只能事后对单个基因做重要性排序，无法直接给出通路级解释；MUPAD [2] 的可解释性则完全"借自"SurvPath [16] 的固定先验，模型本身未学到任何通路-形态对应关系。一个端到端可学习、且解释内生于模型的通路瓶颈，对下游的扰动分析与诊断辅助具有直接价值 [1][16]。
- **理论必要性**：存在一个尚未被回答的科学问题——在生成（而非判别）任务中，注入生物通路结构的收益究竟来自"结构化稀疏"这一机制本身，还是来自真实通路的生物学语义？Sparsity is All You Need [4] 仅在判别任务上回答了此问题（发现随机≈真实），生成任务上无人验证；而 MUPAD [2] 采用固定通路打分，却**从未消融"固定打分 vs 端到端可学习编码"**。这两处认知空白正是本研究 RQ2/RQ3 的靶点，其方法论价值超出单一应用 [4][17]。
- **时机必要性**：三个条件刚好同时成熟。其一，GeneFlow [1] 于 2025 年 11 月开源了可复现的单细胞基因→图像基线与三个预处理 Xenium 数据集；其二，MUPAD [2] 将"通路打分作生成条件"推上台面，却留下了关键的消融空白；其三，Sparsity is All You Need [4]（2025）刚点燃"生物先验是否真正有用"的方法论争论。此刻正是用一个干净的受控实验同时回应这三股力量的窗口期。

> 注：以上三点均有文献支撑。其中"MUPAD 从未消融固定 vs 可学习"为对 MUPAD 论文 [2] 全文精读后的判断，已在精读记录中确认其消融章节仅覆盖 DCA 与对齐损失。

### 2 Research Questions

以下研究问题从 Section 1 的两个核心 gap 推导而来：GeneFlow [1] 的**结构缺失**与 MUPAD [2] 的**可学习性缺失**，并直面 Sparsity is All You Need [4] 提出的"机制 vs 语义"归因难题。

#### 主要研究问题（Primary RQ）

**RQ1：在从单细胞基因表达生成病理图像的逆问题中，将"基因→条件信号"的映射建模为端到端可学习的结构化通路瓶颈（固定通路-基因二值掩码约束稀疏性 + 每个 (通路, 基因) 对的可学习权重向量），是否比 GeneFlow 的无结构编码器以及固定 ssGSEA 通路打分（MUPAD 式）生成质量更高？**

- **对应 gap**：指向 Section 1 的两个相反局限——GeneFlow [1] 的无结构编码器（信息过载、无生物约束）与 MUPAD [2] 的固定打分瓶颈（信息提前掐断、不可学习）。
- **新颖性**：通路 token + 自注意力 + CLS 聚合的架构在判别任务中已被 TOSICA [3] 采用；通路打分作生成条件已被 MUPAD [2] 采用（但固定不可学习）。本研究的区别在于：**端到端可学习的 (通路, 基因) 权重 + 用于条件图像生成 + 与无结构/固定打分的系统受控对照**——此组合尚属空白。
- **可回答性**：可在一篇论文内回答。基线 GeneFlow [1] 代码与数据均已开源，本方法仅需替换其 RNA 编码器（约 2 个文件），消融变体设计清晰，评估指标沿用 GeneFlow 既有协议。

#### 次要研究问题（Secondary RQs）

**RQ2：结构化通路瓶颈的生成收益，多大程度来自"结构化稀疏 + 可学习"这一机制本身，多大程度来自真实通路的生物学语义？**

- **对应 gap**：Sparsity is All You Need [4] 在判别任务上发现"随机通路掩码 ≈ 真实通路"，但生成任务上无人验证；P-NET 复现报告 [17] 则量化了 Reactome 稀疏化的贡献，结论相反。生成任务上的归因是空白。
- **与 RQ1 的关系**：RQ2 是 RQ1 的机制解剖。通过对比 randPath（随机掩码 + 可学习权重）与 Gene2Image（真实通路 + 可学习权重），将 RQ1 的收益拆解为"机制下界"与"语义增量"两部分。无论结果如何，randPath > GeneFlow 都为"结构化瓶颈机制有效"提供下界证据。

**RQ3：端到端可学习的 (通路, 基因) 权重，是否优于固定的 ssGSEA 通路打分作为生成条件？**

- **对应 gap**：MUPAD [2] 采用固定 ssGSEA 式打分，却从未消融"固定 vs 可学习"，构成明确的实验空白。
- **与 RQ1 的关系**：RQ3 是 RQ1 的另一侧手术刀，直接隔离"可学习性"这一变量。通过对比 PathPrior（真实通路 + 固定 ssGSEA 打分）与 Gene2Image（真实通路 + 可学习权重），实证检验 MUPAD 式固定瓶颈的代价。

**RQ4：可学习通路瓶颈能否提供通路→形态的可解释映射（哪些通路对生成的细胞形态贡献最大），且与真实生物学一致？**

- **对应 gap**：GeneFlow [1] 仅能对单个基因做事后重要性排序（命中 EMT/ECM 通路）；MUPAD [2] 的可解释性借自固定先验，非模型内生。模型内生的通路-形态可解释性在生成任务中尚未被建立。
- **与 RQ1 的关系**：RQ4 是 RQ1 的价值放大器，对应"生成质量 + 可解释性双赢"定位中的可解释性一翼。利用 CLS token 的通路注意力权重，对照 GeneFlow 的基因重要性 GSEA 结果，检验解释的生物学一致性。

> 注：4 个 RQ 构成完整论证链——RQ1 立论，RQ2/RQ3 为两把分别对准 GeneFlow 与 MUPAD 的手术刀，RQ4 兜住可解释性。每个 RQ 均可由后续实验明确回答。

### 3 Key Works

本批文献覆盖四类与本研究直接相关的方向：基因→图像生成（任务基线与竞品）、通路引导的表示学习（创新来源域）、通路先验的方法论争议（核心威胁）、以及整流流生成主干（技术基座）。选取逻辑是既锚定本研究的直接对照对象，又覆盖支撑方法设计与实验论证的关键依据。

| 简称 | 会议/期刊 | 年份 | 核心贡献（一句话） | 对本研究的借鉴价值 |
|------|----------|------|-----------------|----------------|
| GeneFlow [1] | NeurIPS | 2025 | 整流流从单细胞基因生成病理图像 | 核心基线、代码、数据、评估协议 |
| MUPAD [2] | arXiv | 2026 | 固定通路打分作条件的病理生成基础模型 | 最直接竞品，RQ3 击穿对象 |
| TOSICA [3] | Nat. Commun. | 2023 | 通路 token+自注意力+CLS 做细胞注释 | 架构思想来源，须界定生成 vs 分类 |
| Sparsity [4] | arXiv | 2025 | 随机通路≈真实通路（判别任务） | 核心威胁，RQ2 回应对象 |
| SurvPath [16] | CVPR | 2024 | 通路 token 与组织交互的生存预测 | 通路 token 化与 MUPAD 打分来源 |
| P-NET [17] | Nature | 2021 | 通路稀疏分层网络与可解释性 | 通路稀疏网络范式与稀疏化贡献证据 |
| pmVAE [5] | ICML WCB | 2021 | 通路模块化 VAE 处理通路重叠 | 通路重叠/冗余问题的处理参考 |
| SD3 [12] | ICML | 2024 | 大规模整流流条件图像合成 | 整流流生成主干的理论依据 |

> 表格中每行对应下方一条详细条目，简称与下方标题保持一致。

#### GeneFlow（NeurIPS 2025）[1]
GeneFlow 首次实现单细胞基因表达到组织病理图像的逆映射生成。其 RNA 编码器由 per-cell 低秩基因-基因关系、全局基因注意力、逐细胞残差编码与多头细胞注意力聚合四部分组成，输出 patch 级条件向量注入整流流条件 UNet；整流流以高阶 ODE 求解器建立表达流形与图像流形间的确定性双射，处理"多对一"映射。在三个 Xenium 黑色素瘤样本上，整流流相对扩散基线 FID 低 3–6 倍，并获 86% 病理学家偏好。

> 借鉴价值：本研究直接基于 GeneFlow 代码库改进，复用其整流流+UNet 主干、三个预处理数据集与评估协议（FID/SSIM/PSNR + UNI2-h 病理指标 + 核形态/空间特征），仅替换其 RNA 编码器。其编码器"无任何通路先验"正是本研究的切入空白。[1]

#### MUPAD（arXiv 2026）[2]
MUPAD 是多模态病理生成基础模型，以 bulk RNA-seq 预训练，将基因表达通过 ssGSEA 式富集打分压缩为 331 维 pathway-level scores（复用 SurvPath [16] 的固定通路签名）后，经 Decoupled Cross-Attention 注入 SiT 流匹配扩散模型生成 H&E。在 RNA→H&E 任务上 FID 较 GeneFlow 降低约 23%，但其通路压缩为生成目标之外的、冻结的预处理步骤。

> 借鉴价值：MUPAD 是本研究最直接的竞品与 RQ3 的击穿对象。其"固定通路打分"的设计缺陷（信息瓶颈不可优化、丢弃基因级分辨率、可解释性借自先验），以及它对"固定 vs 可学习"消融的缺失，直接界定了本研究"可学习 (通路,基因) 权重"的优势论证空间。[2]

#### TOSICA（Nat. Commun. 2023）[3]
TOSICA 用多头自注意力 Transformer 做可解释细胞类型注释：将基因经掩码映射为通路 token，与一个 CLS token 一同输入 Transformer，用 CLS 与通路 token 间的注意力分数作为细胞嵌入，可回溯至原始特征以提供可解释性。在肿瘤浸润免疫细胞与 COVID-19 单核细胞上揭示稀有细胞类型与疾病轨迹。

> 借鉴价值：TOSICA 的"通路 token + 自注意力 + CLS 聚合"与本研究的细胞级编码（步骤 2.2–2.3）高度同构，是关键的架构思想来源。本研究须明确界定区别：TOSICA 用于判别（分类），(通路,基因) 映射偏固定；本研究用于条件生成，且 (通路,基因) 权重端到端可学习。[3]

#### Sparsity is All You Need（arXiv 2025）[4]
该工作系统比较 20 个通路先验深度学习模型与其"随机化"版本（保留网络稀疏性与结构、仅将生物先验替换为随机关联），在多个判别数据集上发现随机版本性能与生物先验版本相当，其中 MPVNN、DeepKEGG、PathDNN 三个模型随机版甚至显著更优，并质疑通路先验在特征选择/可解释性上的必要性。值得注意的是，该工作全部为判别任务（分类、生存分析、回归），未涉及任何生成任务。

> 借鉴价值：这是本研究最核心的威胁，也是 RQ2 的正面回应对象。本研究在生成任务上引入 randPath 变体复现其设定，将"随机≈真实"作为待检验假设而非既定结论，使 randPath 从"自相矛盾的卖点"转化为"机制下界证据"。[4]

#### SurvPath（CVPR 2024）[16]
SurvPath 提出从转录组学习生物通路 token 的 tokenizer，与组织 patch token 经记忆高效的多模态 Transformer 融合，做生存预测，并提供多层次可解释性框架可视化通路-形态的交叉注意力。在 TCGA 五个数据集上取得 SOTA。

> 借鉴价值：SurvPath 定义的通路签名是 MUPAD [2] 固定打分的直接来源，理解它有助于精确刻画 MUPAD 瓶颈。其通路-组织交叉注意力可解释性框架，为本研究 RQ4 的通路-形态可视化提供方法参考。[16]

#### P-NET（Nature 2021）[17]
P-NET 将约 3007 条 Reactome 通路编码为分层稀疏神经网络（基因→通路→生物过程），在前列腺癌分子分型上超越传统模型，并通过节点重要性排序发现 MDM4、FGFR1 等候选驱动基因。其复现报告进一步量化了 Reactome 稀疏化对性能的贡献。

> 借鉴价值：P-NET 是通路稀疏网络的范式代表。其复现报告"量化稀疏化贡献"的结论与 Sparsity [4] 相左，二者构成本研究 RQ2 的争议背景，为"机制 vs 语义"归因提供对立的文献参照。[17]

#### pmVAE（ICML WCB 2021）[5]
pmVAE 将每条通路构建为一个 VAE 模块以学习可解释的单细胞表示，并针对通路高度重叠（基因同时参与多条通路造成冗余）的问题给出模块化处理。

> 借鉴价值：通路重叠/冗余是本研究通路掩码设计必须面对的问题。pmVAE 的讨论为本研究"通路间自注意力如何处理跨通路冗余"提供参照。[5]

#### SD3（ICML 2024）[12]
Stable Diffusion 3 将整流流用于大规模高分辨率条件图像合成，沿最优传输的近直线路径连接噪声与数据分布，提升训练稳定性与少步生成质量。

> 借鉴价值：为本研究保留的整流流+UNet 生成主干提供理论依据，佐证整流流相对扩散在条件生成上的优势。[12]

---

## Part 2 Idea Design

### 1 Introduction

空间转录组技术的成熟，使得在单细胞分辨率上将基因表达与组织形态配对成为可能，并催生了一个新的逆问题：从基因表达直接合成 H&E 染色病理图像。这一能力为虚拟染色、基因扰动的形态学模拟、稀缺样本的数据增广以及疾病机制研究提供了全新的计算手段 [1][6]。GeneFlow [1] 首次在该任务上取得突破，以注意力式 RNA 编码器与整流流条件 UNet 从单细胞表达生成高分辨率图像，并通过病理学家盲评验证了诊断可用性；MUPAD [2] 进一步将其推广为多模态病理生成基础模型。该方向正快速成为计算病理与空间生物学的交叉前沿。

然而，这一逆问题的成败高度依赖 RNA 条件编码器：如何从高维（约 300–5000 基因）、稀疏、带噪声的原始表达谱中提取出对形态有判别力、可解释、且能跨平台泛化的条件信号。现有方法在此处恰好走向两个相反的极端。其一，以 RNA-GAN [7] 与 RNA-CDM [6] 为代表的早期方法用通用 VAE 降维压缩 bulk RNA-seq，丢弃了基因间的生物结构；GeneFlow [1] 虽改用注意力编码器，却仍将基因视为相互独立的输入，作者明确指出其架构 "without any explicit biological knowledge encoded"。这类无结构编码使模型在两万维稀疏信号中缺乏先验约束，信息过载。其二，作为最新的基础模型，MUPAD [2] 引入了通路信息，却采用固定的 ssGSEA 式富集打分——复用 SurvPath [16] 预定义的通路签名，将基因压缩为 331 维标量分数后再注入扩散模型。这一压缩独立于生成目标、不可学习，构成信息论意义上的硬瓶颈：任何打分未保留的基因级变异被永久丢弃，且其可解释性完全借自外部先验，模型本身从未学到通路与形态的对应关系。

更根本地，通路结构在生成任务中的价值来源至今无人厘清。在判别任务上，P-NET [17]、SurvPath [16]、TOSICA [3] 等工作表明通路先验能同时提升性能与可解释性；但 Sparsity is All You Need [4] 在 20 个判别模型上发现随机通路掩码与真实通路性能相当，质疑生物语义的必要性。这两股相反的证据全部局限于判别任务——通路结构注入生成条件后，其收益究竟来自"结构化稀疏"机制本身还是真实生物语义，仍是空白。

针对上述局限，本文提出一种端到端可学习的结构化通路瓶颈编码器，在保持 GeneFlow 整流流生成主干不变的前提下，替换其 RNA 编码器。核心思想是：以固定的通路-基因二值掩码将稀疏约束焊入结构（嵌入生物先验），同时为每个 (通路, 基因) 对赋予可学习的权重向量（适配生成任务），从而在 GeneFlow 的"无结构"与 MUPAD 的"固定打分"之间取得平衡。编码器将基因表达逐细胞映射为通路 token 序列，经通路间自注意力建模协同调控，以 CLS token 聚合为细胞嵌入，再复用 GeneFlow 的多头细胞注意力得到 patch 级条件信号。该设计不仅在保留生物约束的同时允许模型自适应调整基因贡献，更通过随机通路（randPath）、固定打分（PathPrior）等受控消融，首次在生成任务上系统拆解通路结构的收益来源。

本文的贡献如下：
- 提出**端到端可学习的结构化通路瓶颈编码器**，以固定通路-基因掩码约束稀疏性、以可学习 (通路, 基因) 权重向量保留基因级分辨率，填补"生成任务中可学习地注入通路结构"的方法空白。
- 设计**通路间自注意力 + CLS 聚合**的细胞级编码，使通路-形态的可解释映射成为模型内生产物，而非如 MUPAD 借自外部固定先验。
- 通过 randPath / PathPrior / noTrans / noMask 等受控变体，**首次在生成任务上拆解通路收益的"机制 vs 语义"来源**，并正面回应 Sparsity is All You Need [4] 的质疑与 MUPAD [2] 缺失的"固定 vs 可学习"消融。
- 在 GeneFlow 的三个 Xenium 黑色素瘤数据集上验证，本方法在 FID/SSIM 等生成质量指标上相对 GeneFlow 与固定打分基线取得提升，并给出生物学一致的通路可解释性。（占位：完成实验后用真实数据填写。）

### 2 Related Works

#### 2.1 基因表达到病理图像的生成

从基因表达合成组织病理图像是一个新近兴起的逆问题。早期工作集中于 bulk RNA-seq：RNA-GAN [7] 先用 VAE 压缩 RNA-seq 再以 GAN 生成组织 tile，RNA-CDM [6] 改用 β-VAE 编码加级联扩散生成 H&E，二者均缺乏单细胞分辨率与空间结构建模。GeneFlow [1] 首次借助空间转录组在单细胞分辨率上实现该映射，以整流流条件 UNet 生成高分辨率 H&E/DAPI 图像。MUPAD [2] 则将其扩展为多模态病理生成基础模型，以 bulk RNA 预训练并通过固定通路打分注入条件。

> 这一方向的共性局限在于 RNA 条件编码：RNA-GAN/RNA-CDM 用通用降维丢弃了生物结构 [6][7]；GeneFlow 用注意力编码器但不注入任何通路先验 [1]；MUPAD 注入通路但采用冻结的固定打分 [2]。如何在生成任务中可学习地注入通路结构，尚未被解决。

#### 2.2 通路引导的基因表示学习

在判别任务中，将通路先验编码进网络结构已被广泛研究。P-NET [17] 将 Reactome 通路编码为分层稀疏网络，VEGA、pmVAE [5] 以通路模块构建稀疏 VAE，TOSICA [3] 用通路 token 加多头自注意力与 CLS token 做可解释细胞注释，SurvPath [16] 学习通路 token 并与组织 patch 交互做生存预测。这些工作共同表明通路结构能同时提升性能与可解释性。

> 然而这些方法几乎全部面向判别/预测任务（注释、分型、生存），其通路表示服务于分类目标；将可学习通路表示用于条件图像生成的研究尚属空白 [3][16][17]。pmVAE [5] 指出的通路重叠冗余问题，也需在生成条件编码中加以处理。

#### 2.3 通路先验的有效性争议

通路先验的收益来源近期受到质疑。Sparsity is All You Need [4] 系统比较 15 个通路先验模型与其随机化版本，发现在多个判别数据集上随机通路与真实通路性能相当，主张收益主要源于结构化稀疏而非生物语义。与之相对，P-NET 的复现报告 [17] 量化了 Reactome 稀疏化的贡献，支持生物结构的价值。

> 这一争议至今仅在判别任务上展开，生成任务上"机制 vs 语义"的归因完全空白——本文将其作为可检验假设而非既定结论。[4]

#### 2.4 整流流生成

整流流以确定性 ODE 沿近直线路径连接噪声与数据分布，相比扩散训练更稳定、少步生成质量更高。SD3 [12] 将其用于大规模高分辨率条件合成，GeneFlow [1] 则将其用于基因到图像生成并以高阶 ODE 求解器处理多对一映射。

> 整流流为本文提供成熟且确定性的生成主干，本文不改动这一部分，以隔离 RNA 编码器的贡献。[1][12]

#### 2.5 研究空白

综合以上，现有方法在生成任务的 RNA 条件编码上卡在两个相反的极端：一端是 GeneFlow [1] 式的无结构编码器，将基因视为独立特征、信息过载且无生物约束；另一端是 MUPAD [2] 式的固定通路打分，将基因压成冻结标量、信息提前掐断且不可学习。同时，通路先验在生成任务中的有效性来源（机制 vs 语义）从未被检验 [4]，固定打分与可学习编码的对比也从未被消融 [2]。

> 一句话总结：现有方法要么不用通路结构、要么用死了通路结构；本文从"端到端可学习的结构化通路瓶颈"切入，在二者之间找到最优点，并系统检验其收益来源。

### 3 Method

#### 3.1 方法整体框架

本文保持 GeneFlow [1] 的整流流条件 UNet 生成主干不变，仅将其 RNA 编码器替换为一个通路结构化编码器。该编码器以固定的通路-基因二值掩码约束稀疏连接，为每个非零 (通路, 基因) 对赋予可学习的权重向量，将原始基因表达逐细胞映射为通路 token 序列；再经通路间自注意力建模协同调控，以 CLS token 聚合为细胞嵌入；随后复用 GeneFlow 的多头细胞注意力聚合为 patch 级条件向量，注入生成主干。

> 直觉：固定掩码像一张"基因只能流向自己所属通路"的布线图（嵌入生物先验），可学习权重则让模型在这张布线图上自行调节每根线的粗细（适配生成任务）。这恰好介于 GeneFlow 的"无布线图"与 MUPAD 的"布线图焊死且线粗固定"之间。

```text
基因表达 [B×C_max×G] → [A 掩码嵌入] → 通路token [B×C_max×P×D_token]
  → [B Pathway Transformer] → 增强token → [C CLS聚合] → 细胞嵌入 [B×C_max×D_cell]
  → [D 多头细胞注意力·复用GeneFlow] → patch条件 [B×512]
  → [E 整流流UNet·复用GeneFlow] → H&E图像 [B×4×256×256]
```

> 数据流说明：模块 A-C 为本文新增（通路编码），模块 D-E 完全复用 GeneFlow 并冻结其结构设计，使任何性能差异可干净归因到通路编码。

#### 3.2 方法流程详解

第一步，训练前一次性构造通路掩码：从 MSigDB Hallmark 取通路的基因列表，与当前数据集的基因名取交集，去除命中基因少于 3 个的通路，得到固定二值掩码——它规定每个基因只能影响其所属通路，把生物先验焊进结构。

第二步，把每个细胞的基因表达翻译成通路 token：对每条通路，将属于它的所有基因的表达值分别乘以一个可学习权重向量后求和，再加通路偏置，得到该通路的一个 token。权重可学习，使模型能自行决定每个基因在所属通路中对图像形态的重要性，等效隐式特征选择。

第三步，让通路之间互相交流：把一个细胞的所有通路 token 送入一个浅层 Transformer 做自注意力，建模通路间的协同调控（如细胞周期与 DNA 修复通路互相增强或抑制）；通路无序，故不加位置编码。

第四步，把一个细胞浓缩成一个细胞向量：在通路 token 序列前拼一个可学习 CLS token 一同过 Transformer，取 CLS 输出经线性层升维，得到细胞的功能状态向量；CLS 对各通路的注意力权重即为可解释性分析的素材。

第五步，把一个 patch 内多个细胞浓缩成一个 patch 向量：复用 GeneFlow 的多头细胞注意力，对 patch 内有效细胞加权聚合（屏蔽 padding），经特征门控输出 patch 级 RNA 条件向量。

第六步与第七步，训练与推理：复用 GeneFlow 的整流流条件 UNet，将条件向量注入速度场网络，训练时回归噪声到图像的速度场，推理时从噪声沿 ODE 积分生成图像。

> 直觉依据：先在细胞内沿"基因→通路→细胞"逐级浓缩（保留生物层级与可解释性），再在细胞间聚合（复用已验证的 GeneFlow 机制），顺序与生物组织的层级结构一致。

#### 3.3 基因-通路掩码嵌入

输入原始基因表达 $X \in \mathbb{R}^{B \times C_{max} \times G}$（已 log1p 归一化）。定义固定二值掩码 $A \in \{0,1\}^{P \times G}$，其中 $A_{p,g}=1$ 当且仅当基因 $g$ 属于通路 $p$。为每个非零对 $(p,g)$ 引入可学习权重向量 $W_{p,g} \in \mathbb{R}^{D_{token}}$，为每条通路引入可学习偏置 $b_p \in \mathbb{R}^{D_{token}}$。对细胞 $c$，其通路 $p$ 的 token 第 $r$ 维为：

$$
t_{c,p}[r] = \sum_{g:\, A_{p,g}=1} W_{p,g}[r] \cdot X_{b,c,g} + b_p[r]
$$

输出通路 token 张量 $T \in \mathbb{R}^{B \times C_{max} \times P \times D_{token}}$。

> 公式含义：每条通路的 token 是其所属基因表达的可学习加权和。$W_{p,g}$ 可学习意味着模型能把无关基因的权重压向零（隐式特征选择），$A$ 固定保证不发生跨通路的无意义信息流动。该操作等价于一个带掩码约束的稀疏线性变换。

> 实现上对每个非零 $(p,g)$ 的权重与表达值相乘后按通路索引 `scatter_add` 累加，避免显式构造稠密 $P \times G \times D_{token}$ 张量。默认 $D_{token}=48$。每数据集按自身基因名独立构造 $A$，以适配 C1/C2（约 300 基因）与 P1（约 5000 基因）的异质 panel。

> 本模块的设计动机直接对照 MUPAD [2]：MUPAD 用固定 ssGSEA 打分将每通路压为一个标量，本文以可学习向量保留 (通路,基因) 联合结构、并使映射端到端可优化。[2]
> 原文依据："compressed into 331 pathway-level enrichment scores using established pan-cancer gene signatures"（MUPAD Methods）

#### 3.4 Pathway Transformer

将每个细胞独立处理，把 $T$ 重组为 $(B \cdot C_{max}) \times P \times D_{token}$，即每个细胞对应长度为 $P$ 的通路 token 序列。用 $L=2$ 层、$H=8$ 头的 Transformer 编码器建模通路间交互，每层含多头自注意力、前馈网络（隐藏维 $4 D_{token}$）、层归一化与残差连接：

$$
T' = \mathrm{TransformerEncoder}(T), \quad T' \in \mathbb{R}^{B \times C_{max} \times P \times D_{token}}
$$

> 公式含义：自注意力在 $P$ 个通路 token 间进行，允许不同通路交换信息。因通路集合无序，不加位置编码。这建模了生物学中通路通过共享蛋白、交叉调控产生的高阶协同关系。

> 本模块受 TOSICA [3] 的通路 token 自注意力启发，但 TOSICA 用于细胞分类，本文用于生成任务的条件编码。消融变体 noTrans 去除本模块以检验通路间交互的价值。[3]

#### 3.5 CLS 聚合为细胞嵌入

在每个细胞的通路 token 序列前拼接可学习 CLS token $t_{cls} \in \mathbb{R}^{D_{token}}$，与通路 token 一同经过 3.4 的 Transformer（序列长度 $P+1$）。取 CLS 位置的输出 $h_{cls} \in \mathbb{R}^{D_{token}}$，经线性层映射到细胞嵌入：

$$
h_{cell} = \mathrm{Linear}(h_{cls}) \in \mathbb{R}^{D_{cell}}, \quad D_{cell}=256
$$

> 公式含义：CLS token 经自注意力聚合了所有通路的信息，成为细胞功能状态的紧凑表示。CLS 对各通路 token 的注意力权重可量化每条通路对该细胞状态的贡献，是 RQ4 可解释性分析的直接来源。

> 设计参考 TOSICA [3] 以 CLS 注意力作为可解释细胞嵌入的思路。[3]

#### 3.6 多头细胞注意力与生成主干（复用 GeneFlow）

细胞嵌入矩阵 $H \in \mathbb{R}^{B \times C_{max} \times D_{cell}}$ 经 GeneFlow 原始的多头细胞注意力聚合为 patch 级条件向量 $z \in \mathbb{R}^{512}$（以 `num_cells` 屏蔽 padding 细胞，后接特征门控）。随后复用 GeneFlow 整流流条件 UNet：以 GeneFlow 的**正弦插值**路径 $x(t)=\sin(\tfrac{\pi}{2}t)\,x_1+\big(1-\sin(\tfrac{\pi}{2}t)\big)\,x_0$ 连接噪声 $x_0 \sim \mathcal{N}(0,I)$ 与真实图像 $x_1$（训练时对 $x(t)$ 另加 $(1-t)\cdot0.05$ 量级的小随机扰动），网络 $v_\theta(x(t), t, z)$ 回归该路径的**解析速度场**，损失为

$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left\| v_\theta(x(t), t, z) - (x_1 - x_0)\,\tfrac{\pi}{2}\cos(\tfrac{\pi}{2}t) \right\|^2
$$

推理时从噪声沿 ODE 求解器积分至 $t=1$ 得到 $256 \times 256$ 的 H&E/DAPI 图像。

> 公式含义：网络回归的是**正弦路径的解析速度场** $(x_1-x_0)\tfrac{\pi}{2}\cos(\tfrac{\pi}{2}t)$（**注意：不是**线性流匹配常见的 $x_1-x_0$——代码 `rectified_flow.py:91` 用的是前者，论文须照此写）；条件向量 $z$ 经投影加到时间嵌入注入每个残差块。本文保持该速度场与损失、UNet 结构不变（唯一例外：DOPRI5 采样器加了一处欠积分守卫，步数耗尽时强制积分到 $t=1$，`rectified_flow.py:364-380`，不改速度场），使提点可干净归因到通路编码器；$z$ 维度硬对齐 GeneFlow 的 512，接口零改动。[1]

#### 3.7 Baseline 参考与评价指标

| Baseline | 来源 [n] | 选择理由 |
|---------|---------|---------|
| GeneFlow | Wang et al. [1]（NeurIPS 2025）| 核心基线，同主干无通路先验，直接对照 RQ1 |
| MUPAD（思路） | Xiang et al. [2]（arXiv 2026）| 固定通路打分竞品，以 PathPrior 变体复现其条件思路对照 RQ3 |
| Diffusion 同构 | Wang et al. [1]（NeurIPS 2025）| GeneFlow 的扩散对照，验证整流流主干一致性 |

> 选择逻辑：GeneFlow 与本方法共享生成主干、数据与评估协议，仅编码器不同，是最公平的直接对照；PathPrior 在本方法框架内将可学习权重替换为固定 ssGSEA 打分，等价于在受控条件下复现 MUPAD 的通路压缩思想，避免因主干差异污染对比。

| 评价指标 | 定义 | 选择依据 [n] |
|---------|------|------------|
| FID | 生成与真实图像 Inception 特征分布距离（越低越好）| GeneFlow [1] 主指标 |
| SSIM | 结构相似性（越高越好）| GeneFlow [1] |
| PSNR | 峰值信噪比（越高越好）| GeneFlow README [1] |
| UNI2-h FID | 病理基础模型特征下的 FID | GeneFlow [1] 病理专用指标 |
| 核形态相似度 | 分割核后 circularity/eccentricity/solidity 相似度 | GeneFlow [1] 生物学指标 |
| 通路注意力一致性 | CLS 通路注意力与 GSEA 富集通路的一致性 | 本文 RQ4，对照 GeneFlow [1] 基因重要性 |

> 这组指标沿用 GeneFlow 既有评估协议以保证可比性，并新增"通路注意力一致性"以量化 RQ4 的可解释性，覆盖生成质量与生物学可解释性两翼。

---

## References

> 格式：MLA。所有条目均经 web_search 或 PDF 精读验证。Part 1 与 Part 2 中的所有引用统一汇总于此。

[1] Wang, Mengbo, et al. "GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow." *Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2511.00119.

> **主要工作**：整流流从单细胞基因表达生成病理图像。
> **引用原因**：核心基线，提供代码、数据、主干与评估协议。
> **PDF**：`docs/papers/GeneFlow - Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow.pdf`

[2] Xiang, Jinxi, et al. "A Generative Foundation Model for Multimodal Histopathology." *arXiv preprint*, 2026. arXiv:2604.03635.

> **主要工作**：固定通路打分作条件的多模态病理生成基础模型。
> **引用原因**：最直接竞品，RQ3 击穿对象。
> **PDF**：`docs/papers/MUPAD - A Generative Foundation Model for Multimodal Histopathology.pdf`

[3] Chen, Jiawei, et al. "Transformer for One Stop Interpretable Cell Type Annotation." *Nature Communications*, vol. 14, 2023, p. 223.

> **主要工作**：通路 token + 多头自注意力 + CLS 做可解释细胞注释。
> **引用原因**：架构思想来源，须界定生成 vs 分类。
> **PDF**：`[PDF 不可用，仅摘要]` `docs/papers/TOSICA - Transformer for one stop interpretable cell type annotation.txt`

[4] Caranzano, Isabella, et al. "Sparsity is All You Need: Rethinking Biological Pathway-Informed Approaches in Deep Learning." *arXiv preprint*, 2025. arXiv:2505.04300.

> **主要工作**：发现随机通路≈真实通路（20 个判别模型）。
> **引用原因**：核心威胁，RQ2 回应对象。
> **PDF**：`docs/papers/Sparsity is All You Need - Rethinking Biological Pathway-Informed Approaches in Deep Learning.pdf`

[5] Gut, Gilles, Stefan G. Stark, Gunnar Rätsch, and Natalie R. Davidson. "pmVAE: Learning Interpretable Single-Cell Representations with Pathway Modules." *ICML 2021 Workshop on Computational Biology (WCB)*, 2021. bioRxiv:2021.01.28.428664.

> **主要工作**：通路模块化 VAE，处理通路重叠冗余。
> **引用原因**：通路重叠/冗余问题的处理参考。
> **PDF**：`[PDF 不可用，仅摘要]` `docs/papers/pmVAE - Learning Interpretable Single-Cell Representations with Pathway Modules.txt`

[6] Carrillo-Perez, Francisco, et al. "Generation of Synthetic Whole-Slide Image Tiles of Tumours from RNA-Sequencing Data via Cascaded Diffusion Models." *Nature Biomedical Engineering*, vol. 9, no. 3, 2025, pp. 320–332.

> **主要工作**：β-VAE 编码 bulk RNA-seq + 级联扩散生成 H&E tile（RNA-CDM）。
> **引用原因**：bulk RNA→图像的扩散前驱基线。
> **PDF**：`[PDF 不可用，仅摘要]` `docs/papers/RNA-GAN and RNA-CDM - gene expression infused image generation.txt`

[7] Carrillo-Perez, Francisco, et al. "Synthetic Whole-Slide Image Tile Generation with Gene Expression Profile-Infused Deep Generative Models." *iScience*, 2023. PMC10475789.

> **主要工作**：VAE + GAN 从 bulk RNA-seq 生成组织 tile（RNA-GAN）。
> **引用原因**：该任务的 GAN 开山工作。
> **PDF**：`[PDF 不可用，仅摘要]` `docs/papers/RNA-GAN and RNA-CDM - gene expression infused image generation.txt`

[11] Jaume, Guillaume, et al. "HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis." *Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track*, 2024. arXiv:2406.16192.

> **主要工作**：1k+ 配对 ST + H&E 数据集与 benchmark。
> **引用原因**：跨数据集泛化与单细胞分辨率背景。
> **PDF**：`docs/papers/HEST-1k - A Dataset for Spatial Transcriptomics and Histology Image Analysis.pdf`

[12] Esser, Patrick, et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *International Conference on Machine Learning (ICML)*, 2024. arXiv:2403.03206.

> **主要工作**：整流流大规模条件图像合成（SD3）。
> **引用原因**：整流流生成主干的理论依据。
> **PDF**：`docs/papers/SD3 - Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.pdf`

[16] Jaume, Guillaume, et al. "Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. arXiv:2304.06819.

> **主要工作**：通路 token 与组织 patch 交互的生存预测（SurvPath）。
> **引用原因**：通路 token 化与 MUPAD 固定打分的来源。
> **PDF**：`[PDF 不可用]`

[17] Elmarakeby, Haitham A., et al. "Biologically Informed Deep Neural Network for Prostate Cancer Discovery." *Nature*, vol. 598, no. 7880, 2021, pp. 348–352.

> **主要工作**：Reactome 通路分层稀疏网络与可解释性（P-NET）。
> **引用原因**：通路稀疏网络范式与稀疏化贡献证据。
> **PDF**：`[PDF 不可用]`

---

## 待核实清单
> 由 Claude 自动维护。人工核实后逐一打勾。

- [ ] TOSICA 的作者与卷期（位置：[3]，原因：PDF 不可用，仅 web_search 摘要；Nat. Commun. 2023 已确认，首作者名待核）
- [ ] SurvPath PDF 未下载（位置：[16]，原因：无 arXiv 自动下载成功，建议手动补全 arXiv:2304.06819）
- [ ] P-NET PDF 未下载（位置：[17]，原因：Nature 正刊无 arXiv，摘要已确认 DOI:10.1038/s41586-021-03922-4）
- [ ] RNA-GAN 准确出处期刊（位置：[7]，原因：仅摘要，iScience/PMC10475789 待精确核对卷期）
- [ ] MUPAD arXiv 编号 2604.03635（位置：[2]，原因：2026 年预印本，PDF 已下载精读，编号待最终核对）

---

> ⚠️ **阶段 B 完成节点**
> 请审查 Part 1 和 Part 2 后告知 Claude 是否可以进入实验设计阶段。
> - 可以继续 → 直接告知 Claude，进入阶段 C
> - 需要修改 → 提出修改意见，Claude 迭代更新后再次询问确认

---

## Part 3 Experiment Design

### 0 Baseline 实验调研

> 本节由阶段 C-2 对 GeneFlow 代码与论文的精读结果填入，作为后续实验设计的参考基准。

#### 0.1 GeneFlow（NeurIPS 2025）[1]

**论文**：GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow | **代码**：https://github.com/wangmengbo/GeneFlow（本地已 clone 于 `code/`）

**核心 idea**：注意力式 RNA 编码器 + 整流流条件 UNet，从单细胞基因表达生成 H&E/DAPI 图像。本研究直接基于其代码改进，仅替换 RNA 编码器。

**数据集**：

| 数据集 | 规模 | 划分方式 | 划分比例 / 说明 |
|-------|------|---------|--------------|
| Xenium C1（Base FFPE, ~300 基因）| 9394 patches / 106980 cells | 单次随机 random_split | 80% train / 20% val（seed=42）|
| Xenium C2（Add-on FFPE, ~300 基因）| 39334 patches / 70178 cells | 单次随机 random_split | 80/20 |
| Xenium P1（Prime FFPE, ~5000 基因）| 13832 patches / 137927 cells | 单次随机 random_split | 80/20 |

> ⚠️ 重要：GeneFlow 论文宣称 3-fold 交叉验证，但代码（`rectified_main.py:280-285`）实为**单次 80/20 random_split，且评估集 = 验证集，无独立 holdout test**。本研究如实采用 80/20，但补充 ≥3 个随机种子取均值±标准差以提升严谨性。

**实验设计**：

| 实验 | 目的 | 对比模型 | 评估指标 |
|------|------|---------|---------|
| 主实验（单数据集生成）| 验证整流流生成质量 | GeneFlow vs Diffusion 同构 vs Conditional UNet | FID, SSIM, Feature Distance |
| 跨数据集泛化 | 验证跨 panel 泛化 | 在一样本训、另一样本测 | FID, SSIM |
| 消融（组件） | 验证编码器各组件 | -Gene Att., -Gene Rel., -Multi Att., simple encoder | FID, Feature Distance |
| 数据维度 | all/melanoma/non-melanoma | 同模型不同细胞子集 | FID |
| 生物学验证 | 病理可信度 | UNI2-h FID, 核形态, 空间, 病理学家盲评 | UNI2-h 指标 + 人工评分 |

**关键超参数**：batch size = 16（train.sh；argparse 默认 6），lr = 1e-4，epochs = 50（train.sh；论文 100），weight_decay = 0.01，optimizer = AdamW，scheduler = CosineAnnealingLR（eta_min = lr×0.01），patience = 5，AMP 可选，grad clip = 1.0（仅 AMP 路径），L1 正则 = 0.001（编码器首层），gen_steps = 100（DOPRI5 自适应步长上限），seed = 42，model_type = single + img_channels = 4。

> 值得借鉴/注意：(1) 评估指标 SSIM/PSNR 仅用 RGB 前 3 通道、逐样本算均值；FID 分 batch-wise 与 overall 两版。(2) 生物学指标在 batch 级计算，依赖 UNI2-h 与 HE2RNA 外部模型。(3) 随机性控制不完整（无 cudnn deterministic、生成噪声未固定种子），本研究将补 set_seed 以增强可复现性。(4) RNA round-trip 比较的是"真实图→RNA"与"生成图→RNA"，非与原始 RNA 直接比。

#### 0.2 MUPAD（arXiv 2026）[2]（受控复现，非独立运行）

**论文**：A Generative Foundation Model for Multimodal Histopathology | **代码**：未开源

**核心 idea**：固定 ssGSEA 式通路打分（331 维，复用 SurvPath 签名）作为条件注入扩散模型。本研究**不直接运行 MUPAD**（其为 bulk RNA 预训练的基础模型，与单细胞 Xenium 设置不可比），而是在本方法框架内以 **PathPrior 变体**（真实通路 + 固定 ssGSEA 打分）受控复现其"固定通路压缩"思想，隔离"可学习 vs 固定"这一单一变量。

> 这样设计的理由：直接对比 MUPAD 会混入主干（SiT 扩散 vs 整流流）、数据（bulk vs 单细胞）、预训练规模等多重混杂因素，无法干净归因。PathPrior 在完全相同的主干、数据、训练协议下仅替换权重为固定打分，是对 MUPAD 核心设计的公平消融式复现。

---

### 0.3 领域惯例归纳

> 本节综合 GeneFlow [1] 及同领域工作（RNA-CDM [6]、MUPAD [2]）的实验设计，提炼共识。

**标准 Benchmark**：GeneFlow 的三个 Xenium 黑色素瘤样本（C1/C2/P1）是本任务唯一公开、预处理完毕、配对单细胞分辨率的 benchmark，已成为该子方向的事实标准。

**标准评估指标**：FID（越低越好，图像分布保真度）为该领域第一指标；SSIM/PSNR（越高越好）衡量像素级相似度；UNI2-h FID（病理专用，越低越好）衡量病理语义保真；核形态/空间特征相似度衡量生物学合理性。RNA round-trip 相关性衡量基因-形态一致性。

**消融设计惯例**：GeneFlow 逐组件移除编码器模块（-Gene Att. / -Gene Rel. / -Multi Att. / simple encoder）。本领域消融以"移除/替换单一模块"为命名惯例（w/o X 或 X→baseline）。

**结果汇报规范**：GeneFlow 原代码未做多种子重复，仅报单次结果。本研究**超越基线**，对所有变体报告 ≥3 随机种子的均值 ± 标准差，分数据集单独列表。

### 可行性核实摘要

| 项目 | 状态 | 备注 |
|------|------|------|
| 数据集 Xenium C1/C2/P1 | ✅ | Zenodo records/17429142 公开可下载，wget 链接见 README |
| Baseline GeneFlow 代码 | ✅ | 本地已 clone 于 `code/`，PyTorch 2.2.2，可直接训练 |
| 通路数据库 Hallmark/Reactome | ✅ | gseapy 可获取 MSigDB Hallmark 50 / Reactome |
| 外部评估模型 UNI2-h | ⚠️ | HuggingFace MahmoodLab/UNI2-h 需申请权限；基础指标(FID/SSIM/PSNR)不依赖它 |
| 外部评估模型 HE2RNA/Sequoia | ⚠️ | RNA round-trip 需 Sequoia 包 + HE2RNA 权重；为可选指标 |
| GPU 显存 | ✅ | 估算 ≤80GB（GeneFlow 峰值 78GB），用户单卡 A100/H100 满足 |

---

### 1 数据集

#### 1.1 可用数据集

| 数据集 | 类型 | 规模 | 下载路径 | 用途 |
|-------|------|------|---------|------|
| Xenium C1 (Base FFPE) | 单细胞 ST + H&E/DAPI | 9394 patches, ~300 基因 | Zenodo 17429142 / Xenium_V1_hSkin_Melanoma_Base_FFPE | 主实验 |
| Xenium C2 (Add-on FFPE) | 单细胞 ST + H&E/DAPI | 39334 patches, ~300 基因 | Zenodo 17429142 / Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE | 主实验 + 跨数据集 |
| Xenium P1 (Prime FFPE) | 单细胞 ST + H&E/DAPI | 13832 patches, ~5000 基因 | Zenodo 17429142 / Xenium_Prime_Human_Skin_FFPE | 主实验 + 通路扩展消融 |

> 选择理由：这三个样本是 GeneFlow [1] 的原始 benchmark，预处理完毕、配对单细胞分辨率、覆盖从 ~300 到 ~5000 基因的异质 panel，是本任务唯一公开标准数据集，复用它保证与基线完全可比。

#### 1.2 备用数据集

| 数据集 | 类型 | 规模 | 下载路径 | 备用原因 |
|-------|------|------|---------|---------|
| HEST-1k Xenium 子集 [11] | 单细胞 ST + H&E | 59 样本/12 器官 | HuggingFace MahmoodLab/hest | 若需更大规模跨器官泛化验证（可选） |

#### 1.3 数据预处理

直接复用 GeneFlow 预处理产物（adata.h5ad + cell_image_paths.json + .h5 patch），无需自行预处理。基因表达已 log1p 归一化、过滤低质量细胞；图像为 256×256、4 通道（H&E RGB + 1 aux/DAPI）。**新增的唯一预处理是通路掩码构造**：用 gseapy 取 MSigDB Hallmark 50 条通路，与各数据集 `gene_names` 取交集，去除命中基因 < 3 的通路，生成每数据集独立的二值掩码 A。

> 通路掩码按数据集独立构造的理由：C1/C2（~300 基因）与 P1（~5000 基因）panel 差异极大，统一掩码会在小 panel 上产生大量空通路。每数据集独立取交集保证掩码紧致有效。

### 2 实验设计

**工作量基准**：本研究满足顶会标准——3 个数据集、6 个模型变体、4 类实验（主实验 + 消融 + 跨数据集泛化 + 通路可解释性）、≥3 随机种子均值±标准差。

**实验分工原则（避免冗余）**：主实验只回答"Gene2Image 能否超越 SOTA 基线"，对手聚焦为 GeneFlow；消融以 Gene2Image 为满配上界、逐一削弱为各变体，系统回答"每个设计为何必要"。GeneFlow 在两表均出现，但角色不同——主实验中是擂台对手，消融中是"移除整个通路编码器"的极端下界锚点，二者不构成冗余。

#### 2.0 模型变体定义

> 本研究的所有变体由三个正交的二元开关组合而成：**通路掩码**（真实 Hallmark / 随机同密度 / 全 1 无稀疏）、**(通路,基因) 权重**（端到端可学习 / 固定 ssGSEA 打分）、**Pathway Transformer**（保留 / 移除）。每个消融变体相对主方法 Gene2Image 只翻转一个开关，保证差异可干净归因。

| 变体 | 通路掩码 | (通路,基因)权重 | Pathway Transformer | 相对 Gene2Image 翻转的开关 | 角色 |
|------|------|------|------|------|------|
| **Gene2Image**（本文主方法）| 真实 Hallmark | 可学习 | ✅ | —（满配）| 主方法 |
| **GeneFlow** [1] | 无（无通路编码器）| 无 | 无 | 移除整个通路编码器 | SOTA 基线 / 下界锚点 |
| **randPath** | 随机同密度 | 可学习 | ✅ | 掩码：真实→随机 | RQ2 机制归因 |
| **PathPrior** | 真实 Hallmark | **固定 ssGSEA** | ✅ | 权重：可学习→固定 | RQ3 击穿 MUPAD |
| **noTrans** | 真实 Hallmark | 可学习 | ❌ | 移除 Pathway Transformer | 组件：通路协同 |
| **noMask** | 全 1（无稀疏）| 可学习 | ✅ | 掩码：稀疏→全连接 | 组件：结构化稀疏 |

> **PathPrior 的精确定义（关键）**：PathPrior 与 Gene2Image **架构完全相同**（48 维通路 token + Pathway Transformer + CLS 聚合），唯一区别是模块 A 的权重 $W_{p,g}$ 与偏置 $b_p$ 用 ssGSEA 派生值初始化后冻结（`requires_grad=False`）。如此只翻转"可学习性"一个变量。**不采用"每通路单标量"版本**，因为那会同时丢失 token 化与 Pathway Transformer，与 noTrans 混杂，无法干净隔离 RQ3。[2]

> **每个消融对核心方法的支撑逻辑**：randPath 证明"结构化稀疏+可学习"机制本身即带来收益（机制下界，回应 Sparsity [4]）；PathPrior 证明"端到端可学习"优于"固定打分"（击穿 MUPAD [2] 的核心设计）；noTrans 证明通路间协同建模的必要性；noMask 证明生物掩码的稀疏约束优于无约束全连接。四者合围，逐一排除"收益另有来源"的替代解释，使 Gene2Image 的每个设计都有独立证据支撑。

#### 2.1 主实验：与 SOTA 的生成性能对比

**实验目的**：验证 Gene2Image 相对当前 SOTA 基线 GeneFlow 的生成质量优势，回答 RQ1 的核心断言——可学习结构化通路瓶颈优于无结构编码器。

**数据集与划分**：

| 数据集 | 训练集 | 验证集 | 测试集 | 划分方式 | 划分理由 |
|-------|-------|-------|-------|---------|---------|
| C1 / C2 / P1（各自独立）| 80% | 20% | =验证集 | 单次 random_split（seed∈{42,43,44}）| 对齐 GeneFlow [1] 代码实现（其 test≡val）；补 3 种子取均值±std 提升严谨性 |

**评估指标**：

| 指标 | 含义 | 计算方式 | 选择依据 |
|-----|------|---------|---------|
| FID↓ | 生成-真实图像分布距离 | inception_v3 特征，299×299，overall+batch-wise | GeneFlow [1] 主指标 |
| SSIM↑ | 结构相似性 | skimage，逐样本，RGB 前 3 通道 | GeneFlow [1] |
| PSNR↑ | 峰值信噪比 | skimage，逐样本 | GeneFlow [1] |
| UNI2-h FID↓ | 病理语义保真 | UNI2-h ViT-giant 1536 维特征 FID | GeneFlow [1] 病理指标 |

> 这组指标沿用 GeneFlow 评估协议，覆盖像素级（SSIM/PSNR）、分布级（FID）与病理语义级（UNI2-h FID），保证与基线完全可比。

**预期效果**：Gene2Image 在 FID 上相对 GeneFlow 有可测提升（结构化稀疏减少高维噪声干扰），P1（5000 基因）上提升预期最明显（高维 panel 最受益于通路降噪）。

**参与评测的模型**：

| 模型 | 来源 [n] | 类型 | 代码链接 | 简介 |
|-----|---------|------|---------|------|
| **Gene2Image (Ours)** | — | 本文方法 | — | 可学习结构化通路瓶颈编码器 |
| GeneFlow | Wang et al. [1] | 当前 SOTA / 基线 | github.com/wangmengbo/GeneFlow | 无通路先验注意力编码器 |

> 模型选择逻辑：主实验聚焦"能否超越 SOTA"，故只对 GeneFlow（与本方法共享主干/数据/协议、仅编码器不同，最公平的直接对照）。固定打分对照 PathPrior 留待 2.2 消融中以受控方式检验，避免主实验模型堆叠。

**预期结果表（占位）**：

| 模型 | C1 FID↓ | C1 SSIM↑ | C2 FID↓ | C2 SSIM↑ | P1 FID↓ | P1 SSIM↑ | UNI2-h FID↓ |
|-----|--------|---------|--------|---------|--------|---------|-----------|
| GeneFlow | — | — | — | — | — | — | — |
| **Gene2Image (Ours)** | **?** | **?** | **?** | **?** | **?** | **?** | **?** |

#### 2.2 消融实验：通路瓶颈各设计的有效性

**实验目的**：以 Gene2Image 为满配上界，逐一翻转单个开关，系统拆解收益来源——回答 RQ2（机制 vs 语义）、RQ3（可学习 vs 固定），并验证 Pathway Transformer 与结构化稀疏各自的必要性。

**数据集与划分**：同 2.1（C1/C2/P1，80/20，seed∈{42,43,44}）。

**评估指标**：FID↓（主）、SSIM↑、UNI2-h FID↓。

> **三组关键读法**：
> ① RQ2（机制 vs 语义）：randPath vs GeneFlow 检验"结构化稀疏+可学习"机制本身（预期 randPath > GeneFlow，即机制下界成立）；Gene2Image vs randPath 检验真实通路语义的额外价值（若 Gene2Image > randPath 则语义有用，正面回应 Sparsity [4]；若 ≈ 则收益主要来自机制，主线"可学习结构化瓶颈"仍成立）。
> ② RQ3（可学习 vs 固定）：Gene2Image vs PathPrior 隔离"可学习性"单一变量，击穿 MUPAD [2] 的固定打分设计。
> ③ 组件必要性：Gene2Image vs noTrans 检验通路间协同；Gene2Image vs noMask 检验生物稀疏约束。

**预期效果**：预期排序 Gene2Image ≥ randPath > GeneFlow（机制下界），Gene2Image > PathPrior（可学习优势），Gene2Image > noTrans（通路协同有用），Gene2Image > noMask（稀疏约束有用）。

**参与评测的模型**：Gene2Image / randPath / PathPrior / noTrans / noMask / GeneFlow（六变体）。

**预期结果表（占位）**：

| 变体 | C1 FID↓ | P1 FID↓ | UNI2-h FID↓ | 翻转的开关 | 验证的 RQ/组件 |
|-----|--------|--------|-----------|----------|----------|
| GeneFlow | — | — | — | 无通路编码器 | 下界锚点 |
| randPath | — | — | — | 真实→随机掩码 | RQ2 机制 |
| PathPrior | — | — | — | 可学习→固定权重 | RQ3 固定打分 |
| noTrans | — | — | — | 移除 Pathway Trans | 通路协同 |
| noMask | — | — | — | 稀疏→全连接 | 结构化稀疏 |
| **Gene2Image** | **?** | **?** | **?** | 满配 | RQ1 主方法 |

> 附加消融（可选，时间充裕时跑）：**通路数据库规模**——Hallmark(50) vs Hallmark+Reactome(P≈600)，在 P1（5000 基因）上检验通路粒度/数量对生成质量的影响，对应用户描述的 P≈600 设定。

#### 2.3 跨数据集泛化实验

**实验目的**：验证通路语义空间提供 panel 无关的条件表示，使 Gene2Image 在跨数据集/跨 panel 迁移上优于 GeneFlow，回答"通路是否提升泛化"——这是 GeneFlow future work 明确点出的痛点。

**为什么这个实验对本方法特别有利（设计动机）**：GeneFlow 的编码器输入维度等于基因数 G，**强依赖固定基因顺序**；C1/C2(~300 基因) 与 P1(~5000 基因) 的 panel 差异巨大，三样本仅共享约 126 个基因。GeneFlow 跨 panel 迁移时输入空间直接错位。而 Gene2Image 的条件信号是 **P 维通路 token 序列**——只要两数据集的 Hallmark 通路集合重叠（50 条 Hallmark 在任何人类样本上都成立），通路就充当了 panel 无关的"语义中间层"。这是通路结构相对原始基因最直接的泛化优势。

**数据集与划分**：

| 设置 | 训练 | 测试 | 基因/通路对齐方式 | 设计理由 |
|------|------|------|------|---------|
| C1→C2 | C1 全 80% train | C2 验证集(20%) | 各自 gene_names∩Hallmark 构造掩码，**通路名对齐** | 同器官、相近 panel 量级，测分布迁移 |
| C2→C1 | C2 全 80% train | C1 验证集(20%) | 同上 | 反向迁移，排除单向偶然 |
| C1→P1（跨 panel）| C1 全 80% train | P1 验证集(20%) | 通路名对齐（~300基因 vs ~5000基因）| **最严苛**：小 panel→大 panel，最能体现通路语义锚点价值 |

> **对齐机制说明**：跨数据集时，源域与目标域各自用自身 `gene_names∩Hallmark` 构造掩码 A（基因 panel 不同，但产出的通路 token 维度都是 P×D_token，通路名一一对应）。模型在源域学到的"通路→形态"映射，通过共享的通路语义空间迁移到目标域。GeneFlow 无此中间层，跨 panel 时基因输入维度直接不匹配（需补零/截断），泛化必然更差。

**评估指标**：FID↓、SSIM↑、UNI2-h FID↓；并报告**泛化退化率** = (跨数据集 FID − 同数据集 FID) / 同数据集 FID，越小越好。

**预期效果**：Gene2Image 的泛化退化率显著小于 GeneFlow，尤其在 C1→P1 跨 panel 设置上差距最大。

**参与评测的模型**：Gene2Image vs GeneFlow（核心对照，各跑 3 种子）。

> 此实验验证主实验/消融无法覆盖的"跨 panel 泛化"性质，直接对应通路结构的核心卖点，是 Gene2Image 区别于 GeneFlow 的独立增量贡献。

#### 2.4 通路可解释性分析（RQ4 核心实验）

**实验目的**：检验 Gene2Image 的通路注意力是否提供**模型内生、生物学一致、且因果有效**的通路→形态映射，回答 RQ4。这是"生成质量 + 可解释性双赢"定位的另一翼，也是相对 MUPAD（可解释性借自外部固定先验）的关键差异化贡献。

> 设计原则：可解释性不能只停留在"画注意力热图"。要构成独立的科学贡献，必须同时证明三件事——可解释性是**模型内生的**（非外部赋予）、与**已知生物学一致的**（非随机巧合）、且是**因果有效的**（干预通路真能改变形态）。下面三个子分析分别对应这三点。

**子分析 A：通路注意力提取与主导通路识别（内生性）**

- 方法：对每个细胞，提取 Pathway Transformer 中 CLS token 对各通路 token 的注意力权重 $\alpha_{cls \to p}$（**末层、多头取平均**——代码 `get_pathway_attention` 只取最后一层 Transformer 的 CLS 注意力，非跨层平均），作为该细胞的"通路重要性谱"。按细胞类型（melanoma / fibroblast / macrophage / T cell / endothelial 等，由数据集标注或 UNI2-h 聚类得到）聚合，识别每类细胞的 top-k 主导通路。
- 评估指标：

| 指标 | 含义 | 计算方式 |
|-----|------|---------|
| 通路注意力熵 | 注意力是否聚焦（非均匀）| $-\sum_p \alpha_p \log \alpha_p$，越低越聚焦 |
| 细胞类型-通路特异性 | 不同细胞类型主导通路是否可区分 | top-k 通路集合的跨细胞类型 Jaccard 距离 |

> 内生性论证：若注意力高度均匀（熵接近 $\log P$），说明模型未真正利用通路结构；若呈现细胞类型特异的聚焦模式，则证明通路-形态映射是模型训练中自发学到的。

**子分析 B：与已知生物学的一致性（生物合理性）**

- 方法：将 Gene2Image 识别的主导通路，与两个外部参照对照——(1) GeneFlow [1] 通过基因重要性 GSEA 在同数据集上命中的通路（EMT、ECM organization 等黑色素瘤关键通路）；(2) MSigDB 中黑色素瘤相关的已知通路注释。
- 评估指标：

| 指标 | 含义 | 计算方式 | 选择依据 |
|-----|------|---------|---------|
| 通路-GSEA top-k 重合率 | 模型通路与 GSEA 富集通路重合 | $|top_k^{ours} \cap top_k^{GSEA}| / k$ | 对照 GeneFlow [1] |
| 通路排序相关 | 重要性排序一致性 | Spearman 相关（共有通路上）| 标准排序一致性度量 |

> 生物合理性论证：若 Gene2Image 在**无任何通路监督**（仅图像重建损失）的情况下，其注意力主导通路与独立的 GSEA 分析高度重合，则证明模型学到的通路重要性具有真实生物学意义，而非过拟合伪影。这正是 randPath 无法做到的（随机通路无生物语义），构成可解释性相对 randPath 的额外价值。

**子分析 C：通路干预的因果验证（因果有效性）**

- 方法：对训练好的 Gene2Image，在推理时干预特定通路 token——(1) **通路消融**：将某条主导通路的 token 置零（代码只实现置零，未做"置均值"），其余不变，重新生成图像；(2) **通路增强**：把某通路 token 范数放大 ×3（代码已实现，但默认流程只跑消融，增强需手动指定）。**基线与各干预共享同一初始噪声（固定种子）**，再观测生成图像的形态变化。
- 评估指标：

| 指标 | 含义 | 计算方式 |
|-----|------|---------|
| 通路干预形态偏移 | 干预某通路对形态的因果效应 | 干预前后生成图的**像素级 L2 距离**（代码实际实现；UNI2-h 嵌入距离 / 核形态变化为待实现的增强版，勿写成现指标）|
| 干预特异性 | 主导通路 vs 非相关通路的效应差异 | 主导通路干预偏移 / 随机通路干预偏移（比值越大越特异）|

> 因果有效性论证：若干预**主导**通路（如 EMT）引起显著且生物学合理的形态偏移（如细胞间质形态改变），而干预**无关**通路几乎无影响，则证明通路-形态映射是因果的、可控的——这支撑了 Introduction 中"可控生成与扰动分析"的应用价值，是 MUPAD 的固定标量打分无法提供的（其通路分数不可微地嵌入、无 token 级干预接口）。

**参与评测的模型**：Gene2Image（三个子分析）；子分析 B 定性对照 GeneFlow 的基因级重要性；子分析 C 可加 randPath 作对照（验证随机通路干预无生物学合理的特异性偏移）。

> 三个子分析层层递进（内生→合理→因果），共同将"可解释性"从软性卖点升级为可量化、可证伪的独立贡献，是本研究区别于纯性能刷分工作的核心科学价值。


