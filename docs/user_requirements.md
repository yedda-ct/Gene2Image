# 用户需求记录（Gene2Image）

> 本文件由 Claude 通过对话收集维护，随流程推进更新。内容不复制进主文档。

## 阶段 A：方向探索

### 研究方向（待用户最终确认）
从单细胞/空间转录组基因表达生成 H&E 病理图像，改进 GeneFlow 的 RNA 编码器：
以"先验通路-基因归属"为固定掩码，端到端学习每个 (通路, 基因) 对的权重向量，
将原始基因表达映射为通路级 token 序列，再经通路间自注意力 + 细胞聚合，
得到 patch 级条件信号驱动整流流生成。

### 核心动机
现有方法将基因视为相互独立的输入特征，忽视基因通过功能通路协同作用的结构。
GeneFlow 的 RNA 编码器是注意力机制但未显式注入通路生物学先验。

### 明确参考论文
- **GeneFlow**（NeurIPS 2025，arXiv:2511.00119）——核心基线，项目基于其代码库改进

### 用户约束
- 方向约束：必须基于 GeneFlow 代码库（`code/` 目录，PyTorch）改进，保持 RF+UNet 生成主干
- RQ 约束：待确认
- 硬件约束：待确认（C 阶段收集）
- 通路数据库：用户描述提及 MSigDB Hallmark + Reactome 并集（P≈600）；
  历史内存记录曾用 Hallmark 50 条（gseapy）——需与用户确认最终选择

### 文献精读关键洞察（A-5）
**GeneFlow 基线（精读 PDF）**：RNA 编码器 = Gene-Gene Relation(per-cell 低秩分解)
→ Global Gene Attention(全局软门控,单向量 α∈R^G) → Cell Encoder(残差 512→256)
→ Multi-Head Cell Attention(朴素跨head平均) → Final Proj+Gating。
**完全无通路/基因集先验**，作者原文强调 "without any explicit biological knowledge
encoded in its architecture"，且 future work 点名"引入基因表达基础模型"解高维 panel。
→ Gene2Image 的通路掩码嵌入正好补此空白。数据集：Xenium C1/C2/P1(黑色素瘤,
G=126/300/5000,仅126共享)；H100单卡,batch96,100ep,~12h,峰值78GB。
指标：FID/SSIM/Feature Distance + UNI2-h病理FID/核形态/空间GLCM + 病理学家盲评 + 基因重要性GSEA。

**MUPAD 竞品（精读 PDF, arXiv:2604.03635）**：用固定 ssGSEA 式打分把基因压成
**331维 pathway scores**（复用 SurvPath/CVPR2024 的固定通路签名），是**冻结的、
非端到端预处理**。在 RNA→H&E 任务上 FID 超 GeneFlow ~23%。**关键空白：MUPAD 对
"固定通路打分 vs 端到端基因编码"完全没有消融** → Gene2Image 可正面击穿。
Gene2Image 相对 MUPAD 的优势：①(通路,基因)权重端到端可学习,无信息瓶颈
②保留基因级分辨率(非标量打分) ③可解释性是模型内生而非借来的先验。

**Sparsity is All You Need（威胁, arXiv:2505.04300）**：实证发现随机通路掩码与
真实通路性能相当 → 直接挑战"生物先验有价值"。应对：用 randPath 消融正面回应。

**TOSICA（思想同构, Nat Comm 2023）**：通路token+多头自注意力+CLS,但用于分类。
区别：Gene2Image 用于生成 + (通路,基因)可学习权重 + 细胞间聚合。

### 数据集（已确认：完全复用 GeneFlow，Zenodo 17429142）
三个预处理 Xenium 黑色素瘤样本（皮肤）：
- **C1**: Xenium_V1_hSkin_Melanoma_Base_FFPE（标准 panel ~300 基因）
- **C2**: Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE（add-on ~300 基因）
- **P1**: Xenium_Prime_Human_Skin_FFPE（Prime panel ~5000 基因）
下载：wget Zenodo records/17429142。格式：adata.h5ad + cell_image_paths.json + .h5 patch。
img_size=256, img_channels=4（H&E + DAPI aux）。

### 代码接入点（已分析，最小侵入方案 A）
- 关键硬约束：**PathwayEncoder.forward 必须输出 [B, 512]**（=model_channels128×4），
  UNet 的 rna_proj/rna_embed_dim 一行不用改。
- 替换点：multi_model.py:308 / single_model.py:246 的 `self.rna_encoder=...`
- multi 输入 gene_expr=[B, C_max, G]，C_max 是 batch 内动态 max(num_cells)，
  **0-padding 无显式 mask，靠 num_cells 屏蔽**（参考 multi_model.py:207-221）。
- 数据流不做 log1p（FastSeparatePatchDataset 读已预处理的 .h5）；
  基因数 G 运行时从 unified_genes_cache.json 动态确定，**通路掩码列顺序必须与
  dataset.gene_names 严格对齐**。
- 需新增 CLI：--encoder_type {rna,pathway}，--pathway_mask 路径（utils.py setup_parser）。
- rectified_main.py 默认 --model_type=multi。

### 核心定位（已确认）
- **主线叙事 = 端到端可学习的结构化通路瓶颈**（介于 GeneFlow无结构 与 MUPAD固定打分 之间）
- **randPath 定位 = 机制下界证据 + 安全垫**（不作唯一卖点，避免与通路叙事自相矛盾）
  - randPath > GeneFlow → 证明"结构化稀疏+可学习"机制本身有效（下界）
  - PathLearn > randPath → 证明真实通路语义有额外价值（顺带回应 Sparsity 那篇）
  - PathLearn > PathPrior(固定ssGSEA) → 击穿 MUPAD 的固定打分设计
- **提点目标 = 生成质量(FID/SSIM) + 可解释性 双赢**
  - 底线：即使 FID 提点有限，"哪些通路驱动哪些形态"的可解释性作为独立贡献撑起半篇

### 研究问题（已确认全部）
- **RQ1（主）**：将"基因→条件信号"建模为端到端可学习的结构化通路瓶颈
  （固定通路-基因二值掩码 + 每个(通路,基因)对可学习权重向量），
  是否比 GeneFlow 无结构编码器、固定 ssGSEA 打分(MUPAD式) 生成质量更高？
- **RQ2（机制归因/回应Sparsity）**：生成收益多大来自"结构化稀疏+可学习"机制本身，
  多大来自真实通路语义？变体 randPath vs PathLearn。
- **RQ3（可学习vs固定/击穿MUPAD）**：端到端可学习(通路,基因)权重是否优于固定ssGSEA打分？
  变体 PathPrior vs PathLearn。
- **RQ4（可解释性/双赢另一半）**：可学习通路瓶颈能否提供通路→形态可解释映射，
  且与真实生物学一致？CLS注意力+通路重要性 vs GeneFlow基因重要性GSEA。

### 通路数据库（已确认）
- **主实验：MSigDB Hallmark 50 条**（gseapy 获取，轻量稳妥，覆盖C1/C2的~300基因足够）
- **扩展消融：+Reactome（P≈600）** 作为"通路粒度/数量"维度，主要在 P1(5000基因)发挥
- 去除覆盖基因<3的通路

### 技术框架（已确认 B-1）
保持 GeneFlow 整流流+UNet 主干不变，仅替换 RNA 编码器前段为通路结构化编码器：
- 模块A 基因-通路掩码嵌入（固定掩码A∈{0,1}^{P×G} + 可学习 W_{p,g}∈R^D_token + 偏置 b_p）
- 模块B Pathway Transformer（2层8头，通路间自注意力，无位置编码）
- 模块C CLS 聚合为细胞嵌入（CLS token + 线性 D_token→D_cell=256）
- 模块D Multi-Head Cell Attention（**复用 GeneFlow 原版不动**，保证对比公平）
- 模块E Rectified Flow+UNet（**复用 GeneFlow 原版不动**）
输出硬对齐 [B,512]=D_patch。

**维度链**：D_token=48(默认,消融{32,48,64}) → D_cell=256 → D_patch=512。
参数量：Hallmark主实验~0.36M，+Reactome扩展~2.88M。

**消融开关点**：掩码随机化→randPath(RQ2)；权重固定ssGSEA→PathPrior(RQ3)；
去掩码→noMask；去Pathway Transformer→noTrans。

### Pipeline 细节（已确认 B-2）
7步：①训练前一次性构造通路掩码(Hallmark∩gene_names,去<3基因通路)
②基因→通路token(每(通路,基因)独立48维可学习向量+通路偏置)
③Pathway Transformer通路间自注意力 ④CLS聚合→细胞嵌入256维
⑤复用GeneFlow Multi-Head Cell Attention→512维 ⑥复用GeneFlow整流流UNet训练 ⑦推理。
- **权重参数化：每(通路,基因)对独立48维向量**（最大表达力，L1正则鼓励稀疏）
- **跨panel掩码：每数据集独立构造**（各自gene_names∩Hallmark，主实验各数据集分别训练）

### 实验约束（已确认 C-1）
- **GPU：单卡 A100/H100（40-80GB）**，显存充裕，可直接用 GeneFlow 原 batch_size
- **训练时间：充裕**，可复现 GeneFlow 完整 100ep（~12h/实验）
- **实验组合：由 Claude 推荐完整组合**（含主实验+消融+跨数据集泛化+可解释性专项）
- 数据集：C1/C2/P1 各自训练评测；变体 6 个（GeneFlow基线/PathLearn主/randPath/PathPrior/noTrans/noMask）
  + Reactome 通路数量扩展消融

### 实验设计重构决策（C-6 用户反馈后，2026-06-06）
- **核心方法正式命名 Gene2Image**（不再用 PathLearn）。
- **消除主实验/消融冗余**：主实验只对 GeneFlow（聚焦"超越SOTA"）；
  消融以 Gene2Image 为满配上界逐一翻转单开关，GeneFlow 作下界锚点。角色不同非冗余。
- **PathPrior 精确定义=冻结权重版**：架构与 Gene2Image 完全相同，仅模块A的 W_{p,g}/b_p
  用 ssGSEA 初始化后 requires_grad=False。只翻转"可学习性"。不用标量版(会与noTrans混杂)。
- **为何用 PathPrior 不直接跑 MUPAD（写进论文）**：MUPAD 混杂主干(扩散SiT)+数据(bulk预训练)
  +标量压缩三个无关变量，无法归因；且 MUPAD 需未公开预训练权重不可公平复现。
  PathPrior 在相同主干/数据/协议下只隔离"固定vs可学习"，是受控代理而非复刻。
- **三开关正交**：通路掩码(真实/随机/全1)×权重(可学习/固定)×Pathway Transformer(有/无)。
  每个消融只翻转一个：randPath(掩码真→随,RQ2)、PathPrior(权重学→固,RQ3)、
  noTrans(去transformer)、noMask(掩码→全连接)。
- **跨数据集泛化**：C1→C2、C2→C1、**C1→P1(跨panel最严苛)**。通路名作panel无关语义锚点
  （三样本仅共享~126基因，但Hallmark 50条通用）。报泛化退化率。
- **可解释性升级为三子分析**(RQ4核心)：A内生性(CLS通路注意力熵+细胞类型特异性)、
  B生物合理性(与GeneFlow GSEA top-k重合+Spearman)、C因果(通路token消融/增强→形态偏移，
  主导vs随机通路特异性比值)。层层递进:内生→合理→因果。

### GeneFlow 实验设计精读（C-2，精确到代码）
- **划分**：代码实为单次 80/20 random_split（seed=42, rectified_main.py:280），
  **test≡val，无独立holdout，无真正k-fold**（论文宣称3-fold CV在代码中不存在）。
  → 我们的策略：如实采用80/20，但**补多随机种子(≥3)取均值±std**提升严谨性，超越原基线。
- **超参默认**：batch_size train.sh=16(argparse 6), epochs train.sh=50(论文100),
  lr=1e-4, weight_decay=0.01, AdamW, CosineAnnealingLR(eta_min=lr*0.01),
  patience=5, 无LR warmup, AMP可选, grad clip=1.0(仅AMP路径), L1正则=0.001(编码器首层)。
- **指标实现**：SSIM/PSNR(skimage,逐样本,仅RGB前3通道,data_range=1)；
  FID(inception_v3,299×299,batch-wise+overall两版)；UNI2-h FID(ViT-giant,1536维)；
  核形态(Otsu分割+regionprops circularity/eccentricity/solidity, KS相似度=1-KS_stat)；
  空间GLCM(contrast/dissimilarity/homogeneity/energy)；
  RNA round-trip(HE2RNA: real图→RNA vs gen图→RNA, gene-wise corr)。
- **外部模型**：UNI2-h(HF MahmoodLab)、HE2RNA+ResNet50+Sequoia包。基础指标(FID/SSIM/PSNR)无需外部。
- **生成**：gen_steps默认100(eval.sh=50,为步数上限), DOPRI5自适应步长。
- **随机性**：无统一set_seed/无cudnn deterministic，种子默认42，无内置多次重复。
- **single vs multi**：默认multi但shell跑single+img_channels=4(含1 aux通道)。
  gene importance分析仅single；spatial graph loss仅multi。
  → 我们主实验跟随 shell 默认 **single + img_channels=4**（与论文主表可比）。
