# Implementation Guide — Gene2Image：可学习结构化通路瓶颈编码器
> 生成时间：2026-06-07 | 策略：强 Baseline 改进 | 状态：PENDING_REVIEW
> 原始项目：https://github.com/wangmengbo/GeneFlow （本地已 clone 于 `code/`）
> 关联实验设计：docs/idea_report.md Part 3

---

## 0 实现关键决策（阶段 D 对话确认）

| 决策项 | 选择 | 影响 |
|-------|------|------|
| 主干模型 | **single 为主，multi 为附** | single 是主实验/消融/可解释性主线（对齐 GeneFlow 论文主表 + gene importance 可用）；multi 作为附加扩展验证 |
| 多卡 | **单卡即可，可简化** | 改动以单卡为准，DDP 分支保留不主动删除但不保证维护；新增模块需兼容 `isinstance(model, DDP)` 访问以免原分支崩溃 |
| 代码组织 | **新建 `src/pathway_encoder.py`** | 通路编码器独立成文件，single_model.py / multi_model.py 仅在编码器构造处按 `--encoder_type` 分支导入 |

> 这三条决策决定了下文所有改写方案的形态。single 主干意味着通路编码器的核心入口处理 `[B, G]`（每细胞→token 序列→CLS→[B,512]，**无细胞聚合**）；multi 入口在此基础上多一层「复用 GeneFlow 多头细胞注意力」聚合。

---

## 1 原始项目信息

### 1.1 项目概况

- **名称**：GeneFlow
- **链接**：https://github.com/wangmengbo/GeneFlow
- **框架**：PyTorch 2.2.2 + cu121（见 `code/requirements.txt` 注释行）
- **原始功能**：从单细胞/多细胞基因表达，经注意力式 RNA 编码器 + 整流流条件 UNet，生成 256×256、4 通道（H&E RGB + 1 DAPI/aux）病理图像。

### 1.2 核心数据流与张量形状（精读结论）

```text
single 模式：
  CellImageGeneDataset.__getitem__ → gene_expr [G], image [4,256,256]
  → DataLoader 默认 collate → batch['gene_expr'] [B, G]
  → RNAtoHnEModel.forward(x, t, gene_expr, gene_mask)
      → RNAEncoder(gene_expr) [B, G] → [B, 512]   (512 = model_channels128 × 4)
      → RNAConditionedUNet(x, t, {rna_embedding:[B,512]}) → v_pred [B,4,256,256]

multi 模式：
  PatchImageGeneDataset.__getitem__ → gene_expr [n_cells, G], image [4,256,256]
  → patch_collate_fn → batch['gene_expr'] [B, C_max, G]（0-padding）, num_cells [B]
  → MultiCellRNAtoHnEModel.forward(x, t, gene_expr, num_cells, gene_mask)
      → MultiCellRNAEncoder(gene_expr, num_cells) [B,C_max,G] → [B, 512]
      → UNet → v_pred [B,4,256,256]
```

> **硬约束（不可破坏）**：编码器输出必须是 `[B, 512]`，UNet 的 `rna_embed_dim=model_channels*4=512` 与 `rna_proj` 一行不改。本研究所有改动都在「编码器输入 → [B,512]」这一段内部完成。

### 1.3 训练/评估关键事实（精读结论，直接影响改写）

- **整流流非线性路径**（`rectified_flow.py:53-98`）：`x_t = sin(t·π/2)·x_1 + (1-sin)·noise + 微扰`，`velocity = (x_1-noise)·(π/2)·cos(t·π/2)`。损失 = MSE(v_pred, velocity)。**本研究完全不改此文件。**
- **L1 正则硬编码**（`rectified_train.py:323-331` 训练 / `:457-466` 验证）：
  - multi 取 `model.rna_encoder.cell_encoder[0].weight`
  - single 取 `model.rna_encoder.encoder[0].weight`
  - ⚠️ 通路编码器没有 `cell_encoder`/`encoder` 这两个 `nn.Sequential`，**直接跑会 AttributeError**。必须改为调用编码器自带的 `l1_penalty()` 方法（见 3.4）。
- **数据划分**（`rectified_main.py:280-285`）：单次 `random_split(seed)` 80/20，**test ≡ val**。本研究如实保留，靠 `--seed ∈ {42,43,44}` 跑 3 种子。
- **gene_names 来源**：single 模式 = `expr_df.columns`（来自 adata.var_names）；multi fast 模式 = `dataset.gene_names`（unified_genes_cache.json）。**通路掩码列顺序必须严格对齐此 gene_names。**
- **评估脚本损坏**（`rectified_evaluate.py:30-31`）：`from src.single_model_deprecation import ...` 与 `src.multi_model_deprecation` 文件不存在 → import 即崩。必须修复（见 3.7）。
- **生成 round-trip / UNI2-h / HE2RNA**：依赖外部模型权重，基础指标（FID/SSIM/PSNR）不依赖。本研究主指标用基础指标 + 自建 UNI2-h FID（需申请权限，缺失时降级）。

### 1.4 改写范围总览

| 文件/目录 | 操作 | 改写原因 |
|----------|------|---------|
| `src/pathway_encoder.py` | `[NEW]` | 通路结构化编码器（模块 A 掩码嵌入 + B Pathway Transformer + C CLS 聚合），single/multi 两个入口类 + ssGSEA 固定权重路径 |
| `scripts/build_pathway_mask.py` | `[NEW]` | 训练前一次性构造每数据集的通路掩码 A（Hallmark∩gene_names，去<3基因通路），含 random/noMask/ssGSEA 变体产物 |
| `src/single_model.py` | `[MODIFIED]` | `RNAtoHnEModel.__init__` 按 `encoder_type` 选择 RNAEncoder（原版）或 PathwaySingleEncoder（新） |
| `src/multi_model.py` | `[MODIFIED]` | `MultiCellRNAtoHnEModel.__init__` 同上，附加 multi 入口 |
| `src/utils.py` | `[MODIFIED]` | `setup_parser` 新增 `--encoder_type / --pathway_mask / --pathway_db / --d_token / --pt_layers / --pt_heads / --l1_weight` 等参数 |
| `rectified/rectified_train.py` | `[MODIFIED]` | L1 正则改为 `encoder.l1_penalty()`，兼容新旧编码器（4 处：训练/验证 × DDP/非DDP） |
| `rectified/rectified_main.py` | `[MODIFIED]` | `model_constructor_args` 透传通路相关参数；构造掩码张量并传入模型 |
| `rectified/rectified_evaluate.py` | `[MODIFIED]` | 修复损坏的 deprecation import；透传 encoder_type 与通路参数以正确重建模型 |
| `scripts/run_experiments.sh` | `[NEW]` | 6 变体 × 3 数据集 × 3 种子主实验+消融批量脚本 |
| `scripts/run_cross_dataset.sh` | `[NEW]` | 跨数据集泛化（C1→C2 / C2→C1 / C1→P1）脚本 |
| `analysis/pathway_interpret.py` | `[NEW]` | RQ4 三子分析：CLS 通路注意力提取、与 GSEA 一致性、通路干预因果验证 |
| `rectified/rectified_flow.py` | `[KEEP]` | 整流流主干，绝不改动（隔离编码器贡献） |
| `src/unet.py` | `[KEEP]` | UNet 主干，绝不改动 |
| `baseline/*` | `[KEEP]` | 扩散对照，本研究不动 |

> 划分原则：尽量复用 GeneFlow 已验证的生成主干与训练循环，新增代码集中在「通路编码器」与「掩码构造」两块；对原文件的修改限于「编码器选择分支」与「L1 正则解耦」两类最小侵入点，确保任何性能差异可干净归因到通路编码器。

---

## 2 数据流

### 2.1 训练数据流（single 主线）

```text
原始文件（data/processed_data/{C1,C2,P1}/）
  adata.h5ad（基因表达，已 log1p + 过滤；var_names=基因名）
  cell_patch_256_aux/input/cell_image_paths.json（cell_id → tif 路径，⚠️需路径重映射）
  → parse_adata（src/utils.py，复用）
      读取 adata → expr_df [N_cells, G]，gene_names = expr_df.columns
  → CellImageGeneDataset（复用）
      gene_expr [G]（float32），image [4,256,256]
  → random_split 80/20（seed∈{42,43,44}）
  → DataLoader 默认 collate → batch['gene_expr'] [B, G]
  → 整流流采样路径（复用）→ x_t [B,4,256,256], target_velocity
  → PathwaySingleEncoder(gene_expr [B,G], mask_A [P,G])
      A 掩码嵌入: [B,G] → 通路 token [B, P, D_token]
      Pathway Transformer (+CLS): [B, P+1, D_token] → CLS 输出 h_cls [B, D_token]
      投影: h_cls → [B, D_cell=256] → [B, D_patch=512]
  → UNet(x_t, t, {rna_embedding:[B,512]}) → v_pred [B,4,256,256]
  → loss = MSE(v_pred, target_velocity) + l1_weight · encoder.l1_penalty()
```

### 2.2 训练数据流（multi 附线）

```text
  PatchImageGeneDataset → patch_collate_fn → gene_expr [B, C_max, G], num_cells [B]
  → PathwayMultiEncoder(gene_expr [B,C_max,G], num_cells, mask_A)
      reshape → [B·C_max, G]
      A 掩码嵌入 → [B·C_max, P, D_token]
      Pathway Transformer (+CLS) → 每细胞 [B·C_max, D_token] → 投影 [B,C_max,D_cell=256]
      复用 GeneFlow 多头细胞注意力（屏蔽 padding，靠 num_cells）→ [B, 512]
  → UNet → v_pred
```

### 2.3 通路掩码构造数据流（训练前一次性，离线）

```text
gene_names（来自目标数据集 adata.var_names，顺序固定）
  + MSigDB Hallmark 50（gseapy.get_library('MSigDB_Hallmark_2020')）
  → 对每条通路取 gene_set ∩ gene_names
  → 去除命中基因 < 3 的通路
  → A_real ∈ {0,1}^{P×G}（P=保留通路数，列顺序=gene_names）
  变体产物（同一 P、同一每行非零计数分布）：
    A_rand：每行随机置位，行非零数 = A_real 对应行（同密度随机，RQ2/randPath）
    A_none：全 1 矩阵 [P×G]（noMask，去稀疏）
    W_ssgsea：ssGSEA 派生的固定权重（PathPrior/RQ3，见 3.2 与 5.x）
  → 存为 data/pathway_masks/{dataset}_{db}_{variant}.npz
      含 keys: A [P,G] (int8), pathway_names [P], gene_names [G]
```

> **跨 panel 关键决策**：每数据集独立构造 A（各自 `gene_names ∩ Hallmark`）。C1/C2(~300基因) 与 P1(~5000基因) panel 差异极大，统一掩码会在小 panel 上产生大量空通路。跨数据集实验时通路名一一对应（Hallmark 50 在任何人类样本上都成立），通路 token 维度都是 P×D_token，构成 panel 无关语义中间层。

---

## 3 现有文件改写方案

### 3.1 `src/pathway_encoder.py` [NEW]

**文件职责**：实现通路结构化编码器。包含掩码嵌入层、Pathway Transformer、CLS 聚合，封装为 single/multi 两个入口类，并内置 L1 正则接口与「可学习/固定 ssGSEA」「真实/随机/无掩码」三开关。

#### `PathwayMaskEmbedding(nn.Module)` —— 模块 A

- 职责：将基因表达按固定通路掩码映射为通路 token，每个非零 (通路,基因) 对有独立的 `D_token` 维可学习权重向量。对应 idea_report Part 2 §3.3 公式 $t_{c,p}[r]=\sum_{g:A_{p,g}=1} W_{p,g}[r]\cdot X_{g}+b_p[r]$。
- 初始化参数：
  - `mask`（Tensor `[P,G]`，int8/bool）：固定二值掩码，`register_buffer`（不训练、随 checkpoint 保存）
  - `d_token`（int，默认 48）：每通路 token 维度
  - `learnable`（bool，默认 True）：True=权重可学习（Gene2Image）；False=权重 `requires_grad=False`（PathPrior）
  - `init_weight`（Tensor `[P,G]` 或 None）：ssGSEA 派生的初始化权重（PathPrior 用），None 时用默认初始化
- 初始化逻辑：
  1. 由 `mask` 求非零索引 `(p_idx, g_idx)`（`mask.nonzero()`），存为 buffer `edge_p`、`edge_g`，记 `E=len(p_idx)`（边数）
  2. 可学习权重 `self.W = nn.Parameter(torch.empty(E, d_token))`；偏置 `self.bias = nn.Parameter(torch.zeros(P, d_token))`
  3. 初始化：`learnable=True` 时 `W` 用 `kaiming_uniform_`（或 `xavier`）；`init_weight` 给定时按 `(p,g)` 对应值广播到 `d_token` 维并 `requires_grad_(False)`（连同 bias）
  4. 记录 `self.P, self.G, self.d_token, self.E`
- `forward(x: Tensor) -> Tensor`：
  - 输入 `x`：`[N, G]`（N = B 或 B·C_max，已展平到二维）
  - 实现逻辑（用 `scatter_add` 避免显式构造稠密 `P×G×D` 张量）：
    1. 取边上的表达值：`x_edge = x[:, edge_g]` → `[N, E]`
    2. 加权：`contrib = x_edge.unsqueeze(-1) * self.W.unsqueeze(0)` → `[N, E, d_token]`
    3. 按通路索引散射累加：初始化 `T = zeros(N, P, d_token)`；`T.index_add_(1, edge_p, contrib)`（或 `scatter_add`）→ `[N, P, d_token]`
    4. 加通路偏置：`T = T + self.bias.unsqueeze(0)`
  - 输出：`[N, P, d_token]`
  - > index_add_ 沿 dim=1（通路维）累加，等价于带掩码约束的稀疏线性变换；显存峰值 O(N·E·d_token) 远小于 O(N·P·G·d_token)。
- `l1_penalty() -> Tensor`：返回 `self.W.abs().sum()`（标量）。`learnable=False` 时 W 无梯度，penalty 仍可计算但不影响优化——上层按 `l1_weight` 调用。
  - > 这是把 GeneFlow 原来「编码器首层 L1」的语义迁移到通路权重上，鼓励隐式特征选择（无关基因权重压向零）。

#### `PathwayTransformer(nn.Module)` —— 模块 B + C

- 职责：通路 token 间自注意力（建模通路协同）+ CLS 聚合为单一向量。对应 Part 2 §3.4–3.5。
- 初始化参数：
  - `d_token`（int，48）、`n_layers`（int，默认 2）、`n_heads`（int，默认 8）、`dropout`（0.1）
  - `use_transformer`（bool，默认 True）：False=noTrans 消融（跳过自注意力，直接对通路 token 做 mean+CLS 等价聚合）
  - `d_cell`（int，默认 256）：CLS 输出投影后的细胞嵌入维度
- 初始化逻辑：
  1. `self.cls = nn.Parameter(torch.zeros(1,1,d_token))`（CLS token，截断正态初始化）
  2. `use_transformer=True`：`self.encoder = nn.TransformerEncoder(TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=4*d_token, dropout, batch_first=True, norm_first=True), num_layers=n_layers)`；**无位置编码**（通路无序）
  3. `self.proj = nn.Linear(d_token, d_cell)`
  4. 为可解释性保留注意力：`self.last_attn = None`（forward 中可选填充，见 analysis）
- `forward(T: Tensor) -> Tensor`：
  - 输入 `T`：`[N, P, d_token]`
  - 逻辑：
    1. 拼 CLS：`seq = cat([cls.expand(N,1,d), T], dim=1)` → `[N, P+1, d_token]`
    2. `use_transformer=True`：`h = self.encoder(seq)`；取 `h_cls = h[:,0]` → `[N, d_token]`
       `use_transformer=False`：`h_cls = T.mean(dim=1)`（noTrans：无通路交互，纯均值聚合）
    3. `h_cell = self.proj(h_cls)` → `[N, d_cell]`
  - 输出：`[N, d_cell]`
  - > CLS 对各通路 token 的注意力权重是 RQ4 可解释性素材；提取方式见 analysis/pathway_interpret.py（用 hook 或重写 attention 返回权重）。

#### `PathwaySingleEncoder(nn.Module)` —— single 主入口

- 职责：组装 A→B→C，输出 `[B, output_dim=512]`，签名与原 `RNAEncoder` 对齐（`forward(x, mask=None)`），可直接替换。
- 初始化参数：`mask`、`output_dim`(=512)、`d_token`(48)、`n_layers`(2)、`n_heads`(8)、`d_cell`(256)、`dropout`、`learnable`(True)、`use_transformer`(True)、`init_weight`(None)
- 初始化逻辑：
  1. `self.embed = PathwayMaskEmbedding(mask, d_token, learnable, init_weight)`
  2. `self.transformer = PathwayTransformer(d_token, n_layers, n_heads, dropout, use_transformer, d_cell)`
  3. `self.head = nn.Sequential(nn.LayerNorm(d_cell), nn.Linear(d_cell, output_dim), nn.LayerNorm(output_dim))`（投影到 512，对齐 UNet rna_embed_dim）
- `forward(x: Tensor, mask=None) -> Tensor`：
  - 输入 `x`：`[B, G]`（mask 形参为兼容原签名，忽略——通路掩码在 __init__ 注入）
  - 逻辑：`T = self.embed(x)` `[B,P,d]` → `h_cell = self.transformer(T)` `[B,d_cell]` → `z = self.head(h_cell)` `[B,512]`
  - 输出：`[B, 512]`
- `l1_penalty() -> Tensor`：转发 `self.embed.l1_penalty()`
- `get_pathway_attention(x) -> Tensor`：（可解释性用）返回 CLS→通路注意力 `[B, P]`，供 analysis 调用

#### `PathwayMultiEncoder(nn.Module)` —— multi 附入口

- 职责：在 single 编码器基础上，对 patch 内多细胞复用 GeneFlow 的多头细胞注意力聚合，输出 `[B,512]`，签名对齐原 `MultiCellRNAEncoder`（`forward(x, mask=None, num_cells=None)`）。
- 初始化参数：同 single + `num_aggregation_heads`(4)、`use_layer_norm`(True)
- 初始化逻辑：
  1. `self.embed`、`self.transformer`（输出 d_cell=256）同 single
  2. **复用 GeneFlow 细胞聚合**：从 `src.multi_model` 复制多头细胞注意力子模块（`cell_aggregation_attention` + `aggregation_head_projections` + `final_encoder` + `feature_gate`），输入维度 = d_cell=256，输出 512
     > 直接复用原 MultiCellRNAEncoder 的聚合段保证对比公平；不 import 整个类以免耦合，复制聚合相关子模块到本文件的 `_CellAggregator` 辅助类。
- `forward(x, mask=None, num_cells=None) -> Tensor`：
  1. `B,C_max,G = x.shape`；`x_flat = x.reshape(B*C_max, G)`
  2. `T = self.embed(x_flat)` → `[B·C_max, P, d]` → `h = self.transformer(T)` → `[B·C_max, d_cell]`
  3. `cell_emb = h.reshape(B, C_max, d_cell)`
  4. 多头细胞注意力聚合（按 num_cells 屏蔽 padding，逻辑同 multi_model.py:209-242）→ `[B, 512]`
  - 输出：`[B, 512]`
- `l1_penalty()`：转发 `self.embed.l1_penalty()`

> **三开关如何映射到本文件**：
> - 掩码：真实 `A_real` / 随机 `A_rand` / 全1 `A_none` —— 由 `--pathway_mask` 指向不同 .npz，在 main 中加载后传入 `mask` 参数。
> - 权重：可学习（`learnable=True`）/ 固定 ssGSEA（`learnable=False` + `init_weight=W_ssgsea`）。
> - Pathway Transformer：保留（`use_transformer=True`）/ 移除（`use_transformer=False`，noTrans）。

---

### 3.2 `scripts/build_pathway_mask.py` [NEW]

**文件职责**：训练前离线构造通路掩码及其变体，存为 .npz。每数据集运行一次。

**`build_mask(gene_names: list[str], db: str, min_genes: int = 3) -> tuple[np.ndarray, list[str]]`**
- 功能：取 Hallmark（或 Hallmark+Reactome）通路库，对每通路求与 gene_names 交集，去除命中 < min_genes 的通路。
- 参数：`gene_names`（数据集基因名，顺序固定，来自 adata.var_names）；`db`∈{'hallmark','hallmark_reactome'}；`min_genes`
- 返回：`A_real [P,G]`（int8），`pathway_names [P]`
- 实现逻辑：
  1. `gseapy.get_library(name)`：hallmark='MSigDB_Hallmark_2020'，reactome='Reactome_2022'（并集时合并两字典）
  2. 建 `gene2col = {gene:idx}`（来自 gene_names）
  3. 逐通路：`hits = [gene2col[g] for g in geneset if g in gene2col]`；`len(hits)>=min_genes` 才保留
  4. 填 `A_real[p, hits]=1`
  > db 选择对应 Part 3 §2.2 附加消融（Hallmark 50 vs Hallmark+Reactome P≈600），主实验用 hallmark。

**`make_random_mask(A_real: np.ndarray, seed: int) -> np.ndarray`**
- 功能：生成同密度随机掩码（randPath，RQ2）。
- 实现逻辑：对每行 p，令 `k = A_real[p].sum()`，在 G 列中随机选 k 列置 1。逐行保持非零计数 → 与真实掩码「结构化稀疏程度」完全一致，仅打乱基因归属。
  > 这是 Sparsity is All You Need [4] 的「随机化但保结构」设定的精确复现，使 randPath 成为「机制下界」证据。

**`make_none_mask(P: int, G: int) -> np.ndarray`**
- 返回全 1 矩阵 `[P,G]`（noMask，去稀疏：每通路 token 看到全部基因）。

**`build_ssgsea_weights(adata_path, gene_names, A_real, pathway_names) -> np.ndarray`**
- 功能：为 PathPrior（RQ3）构造固定权重 `W_ssgsea [P,G]`，模拟 MUPAD 的固定 ssGSEA 富集打分思想。
- 实现逻辑：
  1. 用 `gseapy.ssgsea` 或基于通路内基因表达均值的固定打分，导出每个 (通路,基因) 的固定贡献权重（最简稳妥版：通路内基因取等权 `1/k_p`，或按训练集平均表达排序的固定权重）
  2. 仅在 `A_real==1` 处有值，其余 0
  > ⚠️ [低置信度：ssGSEA 权重的精确派生方式]：MUPAD 原文是「每通路压成一个标量分数」，本研究 PathPrior 为保证只翻转「可学习性」单变量、不丢 token 化（见 idea_report §2.0 PathPrior 精确定义），采用「固定 (通路,基因) 权重」而非标量。最简实现 = 通路内等权 + 冻结；若需更贴近 ssGSEA，可用训练集表达统计派生固定权重。编码阶段（E）先用等权冻结版跑通，后续按需精化。

**`main()`**：CLI 接 `--adata`、`--gene_names_from`(single 用 adata / multi 用 unified_genes_cache.json)、`--db`、`--out_dir`、`--seed`，输出 `{dataset}_{db}_{variant}.npz`（variant∈real/rand/none，附 ssgsea 权重存同一 real npz 的 `W_ssgsea` key）。

---

### 3.3 `src/single_model.py` [MODIFIED]

**文件职责**：single 模型组装。**现有核心逻辑**：`RNAtoHnEModel.__init__` 固定实例化 `RNAEncoder`（line 246），forward 调 `rna_encoder(gene_expr, mask=gene_mask)` 得 `[B,512]` 喂 UNet。

**需要改写的函数：**

**`RNAtoHnEModel.__init__(..., encoder_type='rna', pathway_mask=None, d_token=48, pt_layers=2, pt_heads=8, learnable_pathway=True, use_pathway_transformer=True, pathway_init_weight=None)`**（新增参数）
- 原来做什么：无条件 `self.rna_encoder = RNAEncoder(...)`。
- 改为做什么：
  1. `if encoder_type == 'rna':` 保持原 `RNAEncoder(...)` 不变（GeneFlow 基线/下界锚点）
  2. `elif encoder_type == 'pathway':` `from src.pathway_encoder import PathwaySingleEncoder`；`self.rna_encoder = PathwaySingleEncoder(mask=pathway_mask, output_dim=model_channels*4, d_token=d_token, n_layers=pt_layers, n_heads=pt_heads, learnable=learnable_pathway, use_transformer=use_pathway_transformer, init_weight=pathway_init_weight)`
- 参数变化：新增上列通路参数，均有默认值（默认 'rna' → 行为与原版完全一致，向后兼容）。
- 返回值变化：无。`forward` 完全不变（编码器输出都是 `[B,512]`）。

> 为什么这样改：编码器输出维度硬对齐 512，UNet 零改动，使「GeneFlow vs Gene2Image」的差异严格隔离在编码器。[1]

---

### 3.4 `rectified/rectified_train.py` [MODIFIED]

**文件职责**：训练/验证循环。**现有核心逻辑**：L1 正则在 4 处（训练 DDP/非DDP、验证 DDP/非DDP）硬取 `rna_encoder.cell_encoder[0].weight`（multi）/`encoder[0].weight`（single）。

**需要改写：L1 正则解耦（4 处，line 322-331 与 455-466）**
- 原来做什么：`l1_penalty = torch.sum(torch.abs(model.rna_encoder.{cell_encoder|encoder}[0].weight)) * 0.001`
- 改为做什么：统一改为调用编码器的 `l1_penalty()` 方法，并兼容原 RNAEncoder（原版无此方法 → 加兜底）：
  1. 取 `enc = model.module.rna_encoder if isinstance(model, DDP) else model.rna_encoder`
  2. `if hasattr(enc, 'l1_penalty'): l1_penalty = enc.l1_penalty() * l1_weight`
  3. `else:`（原 RNAEncoder）保持原逻辑：取 `enc.cell_encoder[0].weight`（multi）/ `enc.encoder[0].weight`（single）
  - 用函数封装 `def _compute_l1(model, is_multi_cell, l1_weight): ...` 放文件顶部，4 处调用它，消除重复。
- 参数变化：`train_with_rectified_flow` 新增 `l1_weight=0.001` 形参（默认与原一致）。
- 返回值变化：无。

> 为什么这样改：通路编码器的 L1 应作用在通路权重 W 上而非首层线性层；硬编码属性名会让新编码器崩溃。封装为方法调用是最干净的解耦。这是阶段 D 识别的**头号崩溃点**。

> 同时给原 `RNAEncoder` / `MultiCellRNAEncoder`（src/single_model.py、multi_model.py）**也补一个 `l1_penalty()` 方法**（返回原首层权重的 L1），这样 4 处分支可统一为「调用 l1_penalty()」，彻底消除 hasattr 兜底。推荐此做法。

---

### 3.5 `rectified/rectified_main.py` [MODIFIED]

**文件职责**：训练入口。**现有核心逻辑**：构建 `model_constructor_args`（line 383-405），按 model_type 实例化模型。

**需要改写：**

**A. 加载通路掩码（新增，在模型构造前）**
- 逻辑：
  1. `if args.encoder_type == 'pathway':` 加载 `np.load(args.pathway_mask)`，取 `A`（按 variant 已是 real/rand/none），转 `torch.tensor` → `pathway_mask`
  2. 校验 `A.shape[1] == gene_dim`（列数必须等于当前 gene_names 长度），不等则报错（提示掩码与数据集 panel 不匹配）
  3. PathPrior 时额外加载 `W_ssgsea` → `pathway_init_weight`，并置 `learnable_pathway=False`

**B. 透传参数到 `model_constructor_args`（修改 line 383-405）**
- 新增 keys：`encoder_type=args.encoder_type, pathway_mask=pathway_mask, d_token=args.d_token, pt_layers=args.pt_layers, pt_heads=args.pt_heads, learnable_pathway=args.learnable_pathway, use_pathway_transformer=args.use_pathway_transformer, pathway_init_weight=pathway_init_weight`
- 这些 key 对 RNAtoHnEModel / MultiCellRNAtoHnEModel 均新增同名形参（默认值保证 encoder_type='rna' 时行为不变）。

**C. 透传 l1_weight 到训练（修改 train_with_rectified_flow 调用，line 576）**
- 新增 `l1_weight=args.l1_weight`

> 校验掩码列数 = gene_dim 是防止「掩码 gene 顺序与 dataset.gene_names 错位」这一隐性致命 bug（user_requirements line 66 明确点名）。

---

### 3.6 `src/utils.py` [MODIFIED]

**文件职责**：CLI 参数。**现有核心逻辑**：`setup_parser`（line 198）集中定义所有公共参数。

**需要改写：在 arch_group 新增通路参数组**

```
pathway_group = parser.add_argument_group('Pathway Encoder')
--encoder_type {rna,pathway}  default='rna'   # rna=GeneFlow基线; pathway=Gene2Image
--pathway_mask <path>         default=None    # .npz 掩码文件（real/rand/none 变体）
--pathway_db {hallmark,hallmark_reactome} default='hallmark'
--d_token int                 default=48
--pt_layers int               default=2
--pt_heads int                default=8
--learnable_pathway           action=store_true default=True   # 配 --no_learnable_pathway → PathPrior
--no_learnable_pathway        dest=learnable_pathway store_false
--use_pathway_transformer     action=store_true default=True    # 配 --no_pathway_transformer → noTrans
--no_pathway_transformer      dest=use_pathway_transformer store_false
--l1_weight float             default=0.001
```

> 6 变体的 CLI 组合（single，img_channels=4）：
> - **Gene2Image**：`--encoder_type pathway --pathway_mask {ds}_hallmark_real.npz`
> - **GeneFlow**：`--encoder_type rna`（默认，不传 pathway）
> - **randPath**：`--encoder_type pathway --pathway_mask {ds}_hallmark_rand.npz`
> - **PathPrior**：`--encoder_type pathway --pathway_mask {ds}_hallmark_real.npz --no_learnable_pathway`（main 自动载 W_ssgsea）
> - **noTrans**：`--encoder_type pathway --pathway_mask {ds}_hallmark_real.npz --no_pathway_transformer`
> - **noMask**：`--encoder_type pathway --pathway_mask {ds}_hallmark_none.npz`

---

### 3.7 `rectified/rectified_evaluate.py` [MODIFIED]

**文件职责**：评估（FID/SSIM/PSNR + UNI2-h）。**现有核心逻辑**：import 段（line 30-31）引用不存在的 `single_model_deprecation`/`multi_model_deprecation` → 崩溃；用 setup_parser，构造模型后载 checkpoint 评估。

**需要改写：**
- **修复 import（line 30-31）**：删除/注释两行 `*_deprecation` import（确认全文件无使用，仅历史遗留），或改为 try/except 兜底。
- **重建模型时透传 encoder_type 与通路参数**：评估时需用与训练相同的编码器结构重建模型，否则 load_state_dict 失败。逻辑：从 checkpoint 或 args 读取 encoder_type/pathway_mask 等，按 3.3/3.5 同样方式构造模型。建议在 checkpoint 中**额外保存 `model_config`**（在 rectified_train.py 保存 checkpoint 时附 `args` 关键字段），评估时优先从 checkpoint 读取，保证训练-评估配置一致。
- **基础指标不依赖外部模型**：FID（inception_v3）、SSIM/PSNR（skimage）保留可用；UNI2-h FID 在权重缺失时 try/except 降级跳过并 log 警告。

> 这是阶段 D 识别的**第二崩溃点**：原始 eval 脚本本地直接 import 即报错，必须先修复才能产出任何评估指标。

---

### 3.8 `analysis/pathway_interpret.py` [NEW]（RQ4 三子分析）

**文件职责**：对训练好的 Gene2Image（single）做通路可解释性分析，对应 Part 3 §2.4。

**子分析 A — `extract_pathway_attention(model, data_loader, device) -> pd.DataFrame`**
- 提取 PathwayTransformer 中 CLS→各通路 token 注意力 `α_{cls→p}`（多层多头平均），逐细胞 `[N, P]`。
- 实现：给 TransformerEncoderLayer 的 self-attn 注册 hook 取 `attn_output_weights`（需 `need_weights=True, average_attn_weights=True`），或自定义 attention 子层返回权重。取 CLS 行对其余 P 列。
- 输出指标：通路注意力熵 `-Σ α_p log α_p`（越低越聚焦）；按细胞类型聚合的 top-k 主导通路；细胞类型-通路 Jaccard 特异性。
- 结果存 `results/interpret/{ds}_attention.csv`（cell_id, cell_type, pathway, attention）。

**子分析 B — `consistency_with_gsea(attn_df, geneflow_importance_csv) -> dict`**
- 将 Gene2Image top-k 主导通路 vs GeneFlow `gene_importance_scores.csv` 经 GSEA 富集的通路对照。
- 指标：top-k 重合率 `|topk_ours ∩ topk_gsea|/k`；Spearman 排序相关。
- > GeneFlow gene importance 由 `analyze_gene_importance`（src/utils.py:449）产出，仅 single 可用——这正是选 single 为主干的价值之一。

**子分析 C — `pathway_intervention(model, ...) -> pd.DataFrame`**
- 推理时干预通路 token：消融（置零/置均值）或增强（放大范数），重新生成，测形态偏移。
- 指标：干预前后生成图 UNI2-h 嵌入距离 / 核形态变化；主导通路 vs 随机通路偏移比值（干预特异性）。
- 实现：在 PathwayMaskEmbedding 输出 T 处插入干预钩子（指定 pathway idx 操作），其余前向不变。

---

## 4 数据下载与准备

### 4.1 数据集

| 数据集 | 类型 | 来源 | 存放路径 | 获取方式 |
|-------|------|------|---------|---------|
| Xenium C1 (Base FFPE, ~300基因) | 单细胞 ST+H&E/DAPI | Zenodo 17429142 | `data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/` | 从 Zenodo 下载并解压 |
| Xenium C2 (Add-on FFPE, ~300基因) | 同上 | Zenodo 17429142 | `data/processed_data/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/` | 从 Zenodo 下载并解压 |
| Xenium P1 (Prime FFPE, ~5000基因) | 同上 | Zenodo 17429142 | `data/processed_data/Xenium_Prime_Human_Skin_FFPE/` | 从 Zenodo 下载并解压 |

> 数据已由作者处理好并归档：**Zenodo 17429142**（https://zenodo.org/records/17429142，开放下载）。
> 每个样本目录含 `adata.h5ad` + `cell_patch_256_aux/input/{cell_image_paths.json, cell_images/}`。
> 下载、放置与自检的完整步骤见根目录 **`DATA_SETUP.md`**。**无需从 10x 下原始数据自行预处理。**

### 4.2 必做预处理步骤（编码前）

**步骤 1：修复 cell_image_paths.json 的失效绝对路径** ⚠️
- 问题：json 中路径为 `/depot/natallah/data/Mengbo/HnE_RNA/processed_data/...`（原作者集群路径），本地实际文件在 `data/processed_data/.../cell_images/`。直接训练会 0 张图通过过滤（rectified_main.py:170 `os.path.exists` 全 False）。
- 解决：写 `scripts/fix_image_paths.py`，将 json 中每条路径的前缀替换为本地实际前缀，输出 `cell_image_paths_local.json`。训练用 `--image_paths` 指向修复后文件。
- 校验：修复后 `os.path.exists` 命中数应 = cell 数（C1≈106980）。

**步骤 2：构造通路掩码**
```bash
mkdir -p data/pathway_masks
# 对每个数据集（gene_names 从 adata 读）
python scripts/build_pathway_mask.py --adata data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/adata.h5ad --db hallmark --out_dir data/pathway_masks --seed 42
# → 产出 c1_hallmark_real.npz / c1_hallmark_rand.npz / c1_hallmark_none.npz（含 W_ssgsea）
# C2、P1 同理；P1 额外跑 --db hallmark_reactome（附加消融）
```

**步骤 3：环境修复** ⚠️
- `zw@Gene2Image` 环境的 torch 报 `iJIT_NotifyEvent`（MKL 符号冲突）。需修复：`pip install mkl==2024.0`（或降级 mkl）/ 重装匹配的 torch 2.2.2+cu121。编码阶段先确认 `python -c "import torch"` 通过。
- 安装 `gseapy`（掩码构造依赖，不在原 requirements）。

### 4.3 数据字段说明

| 字段 | 类型 | 含义 | 备注 |
|-----|------|------|------|
| adata.X | float32 | 基因表达（已 log1p 归一化）| FastSeparatePatchDataset/CellImageGeneDataset 不再额外 log1p |
| adata.var_names | str | 基因名（掩码列对齐基准）| C1/C2≈300, P1≈5000 |
| cell_images/*_original.tif | uint8 | 256×256×4（RGB+DAPI aux）| img_channels=4 |
| pathway_masks/*.npz: A | int8 [P,G] | 二值通路掩码 | 列序 = adata.var_names |
| pathway_masks/*.npz: W_ssgsea | float32 [P,G] | PathPrior 固定权重 | 仅 real npz |

---

## 5 results 文件格式规范

### 5.1 模型权重 `results/{exp_name}/checkpoints/best_checkpoint.pt`
- 格式：PyTorch dict，含 `model_state_dict`、`epoch`、`best_val_loss`、`val_loss`、**`model_config`**（新增：encoder_type/pathway_mask/d_token/...，供 eval 重建模型）。
- 读取：`torch.load(path, map_location='cpu')`。

### 5.2 训练曲线 `results/{exp_name}/training_losses.csv`（复用原 GeneFlow 输出）
| 字段 | 类型 | 含义 |
|-----|------|------|
| epoch | int | epoch 编号，从 1 |
| train_loss | float | 训练集平均损失（含 L1）|
| val_loss | float | 验证集平均损失 |

### 5.3 评估结果 `results/{exp_name}/eval_{seed}.json`
| 字段 | 类型 | 单位 | 含义 | 方向 |
|-----|------|------|------|------|
| variant | str | — | 模型变体名（gene2image/geneflow/randpath/pathprior/notrans/nomask）| — |
| dataset | str | — | C1/C2/P1 | — |
| seed | int | — | 随机种子 | — |
| fid_overall | float | — | inception_v3 整体 FID | 越低越好 |
| fid_batchwise | float | — | batch 级 FID 均值 | 越低越好 |
| ssim | float | — | 结构相似性（RGB前3通道逐样本均值）| 越高越好 |
| psnr | float | dB | 峰值信噪比 | 越高越好 |
| uni2h_fid | float/null | — | UNI2-h 特征 FID（缺权重时 null）| 越低越好 |
| num_samples | int | — | 评估样本数 | — |
| checkpoint | str | — | 权重路径 | — |

### 5.4 多种子汇总 `results/summary_main.csv`（主实验 2.1）
| 字段 | 含义 |
|-----|------|
| variant, dataset | 变体与数据集 |
| fid_mean, fid_std | 3 种子 FID 均值±标准差 |
| ssim_mean, ssim_std / psnr_* / uni2h_fid_* | 同上各指标 |

### 5.5 消融汇总 `results/ablation/summary.csv`（消融 2.2）
| 字段 | 含义 |
|-----|------|
| variant | gene2image/randpath/pathprior/notrans/nomask/geneflow |
| dataset | C1/P1（消融主看 C1、P1）|
| fid_mean±std, uni2h_fid_mean±std | 指标 |
| flipped_switch | 翻转的开关（真实→随机/可学习→固定/去transformer/稀疏→全连接/无编码器）|
| target_rq | RQ2/RQ3/组件 |

### 5.6 跨数据集 `results/cross_dataset/summary.csv`（2.3）
| 字段 | 含义 |
|-----|------|
| model | gene2image / geneflow |
| setting | C1→C2 / C2→C1 / C1→P1 |
| fid_cross, fid_same | 跨/同数据集 FID |
| degradation_rate | (fid_cross − fid_same)/fid_same，越小越好 |

### 5.7 可解释性 `results/interpret/`（2.4，RQ4）
- `{ds}_attention.csv`：cell_id, cell_type, pathway, attention, entropy（子分析 A）
- `{ds}_gsea_consistency.json`：topk_overlap, spearman（子分析 B）
- `{ds}_intervention.csv`：pathway, intervention_type, morph_shift, specificity_ratio（子分析 C）

---

## 6 实现顺序

```
1. 环境修复：torch 可 import + 安装 gseapy（4.2 步骤3）→ 跑通原始 GeneFlow single 训练（小 epoch 烟测，记录基线 FID）
   - 同步修复 cell_image_paths 路径（4.2 步骤1）与 eval 损坏 import（3.7）——否则原项目本地跑不通
2. scripts/build_pathway_mask.py（含 real/rand/none + W_ssgsea）→ 对 C1 产出 .npz，校验列数=gene_dim
3. src/pathway_encoder.py：先 PathwayMaskEmbedding（单测 shape [B,G]→[B,P,d] + l1_penalty）
   → PathwayTransformer（[B,P,d]→[B,d_cell]，含 use_transformer 开关）
   → PathwaySingleEncoder（→[B,512]，替换 RNAEncoder 单测前向）
   → PathwayMultiEncoder（multi 附线）
4. 改 src/single_model.py + multi_model.py（encoder_type 分支）+ 补原编码器 l1_penalty()
5. 改 src/utils.py（CLI）+ rectified_main.py（载掩码/透传/校验列数）
6. 改 rectified_train.py（L1 解耦，_compute_l1 封装）
7. 端到端最小训练：Gene2Image on C1（--debug 小样本）跑通 1 epoch，确认无 shape/属性错误
8. 改 rectified_evaluate.py（修 import + model_config 重建）→ 跑通 eval 产出 eval json
9. scripts/run_experiments.sh（6变体×3数据集×3种子）+ run_cross_dataset.sh
10. analysis/pathway_interpret.py（RQ4 三子分析）
11. 结果汇总脚本（summary_main / ablation / cross_dataset csv）
```

每完成一个文件，立即在 `docs/dev_log.md` 更新进度并添加日志条目。`✅ Done` 只能在文件写完且本地运行验证无报错后标记。

> 实现顺序的关键依赖：第 1 步「跑通原始项目」是基准——必须先确认原 GeneFlow 在本地能训能评（含修两个崩溃点），才能保证后续「Gene2Image vs GeneFlow」对比公平、差异可归因。
