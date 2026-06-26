# 开发日志 — Gene2Image：可学习结构化通路瓶颈编码器
> 创建时间：2026-06-07 | 最后更新：2026-06-07
> 关联实现指南：docs/implementation.md

## 项目概览
| 项目 | 内容 |
|------|------|
| 研究方向 | 从单细胞基因表达生成 H&E 病理图像，将 RNA 编码器替换为可学习结构化通路瓶颈 |
| 实现策略 | 强 baseline 改写：GeneFlow（code/ 已 clone）|
| 框架 | PyTorch 2.2.2 + cu121 |
| 环境 | conda `zw@Gene2Image`（mkl 已降级至 2024.0.0 修复 torch import）|
| 主干 | single 为主、multi 为附 |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|------|------|------|---------|------|
| 环境修复 | mkl/gseapy | ✅ Done | 2026-06-07 | mkl 2025→2024.0.0；gseapy 1.2.1；torch+cuda 可用 |
| 路径修复 | scripts/fix_image_paths.py | ✅ Done | 2026-06-07 | C1 106980/106980 路径命中；C2/P1 待同样处理 |
| eval 修复 | rectified/rectified_evaluate.py | ✅ Done | 2026-06-07 | deprecation import 改 try/except + 守卫；加 pathway 重建分支 |
| 基线烟测 | — | ✅ Done | 2026-06-07 | 原 single 模型 forward+backward 通过，51.5M 参数，输出[B,4,256,256] |
| 通路掩码 | scripts/build_pathway_mask.py | ✅ Done | 2026-06-07 | C1/C2/P1 real/rand/none + W_ssgsea + P1 hallmark_reactome；全部已生成验证 |
| 通路编码器 | src/pathway_encoder.py | ✅ Done | 2026-06-07 | A→B→C，single/multi 入口；6变体 forward+backward 通过 |
| 模型集成 | src/single_model.py, multi_model.py | ✅ Done | 2026-06-07 | encoder_type 分支 + 原编码器补 l1_penalty() |
| CLI/入口 | src/utils.py, rectified_main.py | ✅ Done | 2026-06-07 | 通路参数组 + 载掩码透传 + 列数校验 + model_config 入 checkpoint |
| 训练 L1 | rectified/rectified_train.py | ✅ Done | 2026-06-07 | compute_l1_penalty 封装，4处统一；l1_weight/model_config 形参 |
| 评估修复 | rectified/rectified_evaluate.py | ✅ Done | 2026-06-07 | UNI2-h/sequoia/embeddings 全部优雅降级；pathway 重建；基础指标跑通 |
| 实验脚本 | scripts/run_*.sh + build_cross_masks.py | ✅ Done | 2026-06-07 | 6变体×3数据集×3种子 + 跨数据集(通路名对齐掩码) + summarize_results |
| 可解释性 | analysis/pathway_interpret.py | ✅ Done | 2026-06-07 | RQ4 三子分析 A/B/C 端到端跑通 |
| notebook | notebooks/01_*.ipynb | ✅ Done | 2026-06-07 | 掩码/token/注意力/生成 可视化；JSON 校验通过 |
| 文档 | README.md, .gitignore, requirements | ✅ Done | 2026-06-07 | 根目录运行说明；忽略 results/logs 保留 masks；新依赖记录 |

状态：⬜ TODO / 🔄 WIP / ✅ Done（已运行验证）/ ❌ Blocked

## 开发日志

### 2026-06-07 — 环境修复
- **完成内容**：诊断并修复 `zw@Gene2Image` 环境。根因 = mkl 2025.0.0 与 torch 2.2.2 不兼容（`undefined symbol: iJIT_NotifyEvent`）。`pip install mkl==2024.0.0` 后 `import torch` + `torch.cuda.is_available()=True` 通过。安装 gseapy 1.2.1（通路掩码依赖，原 requirements 缺）。
- **遇到的问题**：conda main channel 无 mkl<2025；anndata/scanpy/numpy 本就可用。
- **解决方案**：改用 pip 安装 mkl 2024.0.0（连带 intel-openmp 2024.2.2、tbb）。
- **数据现状**：C1/C2/P1 adata.h5ad 已在本地（2.4/4.7/5.2 GB），cell_patch_256_aux 图像齐全；但 cell_image_paths.json 内为原作者集群绝对路径，需重映射。

### 2026-06-07 — 路径/eval 修复 + 基线烟测
- **完成内容**：(1) `scripts/fix_image_paths.py` 重映射 C1 cell_image_paths，106980/106980 全命中本地 .tif。(2) 修复 `rectified_evaluate.py`：损坏的 `*_deprecation` import 改 try/except + 两处 append 守卫；新增 pathway 编码器重建分支（从 checkpoint config 还原掩码）。(3) 原始 single 模型 forward+backward 烟测通过（51.5M 参数，输出 [B,4,256,256]，L1 属性路径完好）。
- **遇到的问题**：GPU 实为 4× **V100 32GB**，非 idea_report 估算的 H100 80GB（峰值 78GB）。
- **解决方案**：真实训练需减小 batch_size（如 single 4→8，按显存调）/ 开 --use_amp；完整 100ep 训练不在本地代跑（按用户指示）。
- **未代跑**：原始 GeneFlow 完整训练（~12h），仅做模型级烟测确认代码路径无误。

### 2026-06-07 — 通路编码器 + 全链路集成
- **完成内容**：
  1. `scripts/build_pathway_mask.py`：构造通路掩码 + 三变体 + ssGSEA 权重。已对 C1(G=282,P=33)/C2(G=382,P=40)/P1(G=5006,P=50) 生成 real/rand/none；P1 额外 hallmark_reactome(P=1666)。校验列数=gene_dim、rand 保留每行计数、none 全1、W_ssgsea 行和=1。
  2. `src/pathway_encoder.py`：PathwayMaskEmbedding(edge-list scatter_add)→PathwayTransformer(自定义层可导出CLS注意力)→PathwaySingleEncoder/PathwayMultiEncoder。
  3. 集成到 single_model/multi_model(encoder_type 分支)、utils.py(CLI 参数组)、rectified_main.py(载掩码+列数校验+model_config)、rectified_train.py(compute_l1_penalty 解耦)。
  4. **集成测试通过**：6变体(Gene2Image/randPath/PathPrior/noTrans/noMask/GeneFlow)×(single+multi) forward+backward 全过；输出统一 [B,4,256,256]；PathPrior W 冻结确认；CLS→通路注意力 [B,P] 行和≈1。Gene2Image single 42.3M 可训练参数(< 基线 51.5M)。
- **遇到的问题**：torchmetrics 未安装(原 requirements 注释掉)→训练循环 import 失败。
- **解决方案**：`pip install torchmetrics==1.7.1`。
- **注意**：P1+Reactome 实际 P=1666(idea_report 估 P≈600)，通路编码器会更大；附加消融时留意显存。

### 2026-06-07 — 评估修复 + 实验脚本 + 可解释性 + 文档
- **完成内容**：
  1. **rectified_evaluate.py 全链路修复**：原脚本本地无法运行。逐一修复 import 与可选依赖：`*_deprecation`(try/except)、UNI2-h(load 返回 None 而非 raise + 4 处计算块守卫)、sequoia/HE2RNA(utils_he2rna 惰性 import)、embeddings 保存(空数组守卫)。结果：基础 FID/SSIM/PSNR 完整跑通，UNI2-h/round-trip 缺权重时自动跳过(N/A)。pathway checkpoint 自动重建编码器。
  2. **实验脚本**：run_experiments.sh(6变体×3数据集×3种子)、build_cross_masks.py(通路名对齐的跨数据集掩码)、run_cross_dataset.sh(c1→c2/c2→c1/c1→p1，训练源+评估目标+源参考)、summarize_results.py(多种子均值±std → summary_main/ablation csv)。
  3. **analysis/pathway_interpret.py**：RQ4 三子分析(A 注意力熵+主导通路 / B 与参照重合 / C 通路干预形态偏移)，端到端跑通产出 csv/json。
  4. **文档**：根 README.md(环境/数据/变体/运行全流程)、.gitignore(忽略 results/logs，保留 pathway_masks)、requirements.txt(记录新依赖)、notebooks/01 可视化。
- **端到端验证(真实 C1 数据)**：train(debug,1ep) → checkpoint(weights_only 可载) → 生成+gene importance → eval(FID/SSIM/PSNR,UNI2-h降级) → interpret(A/B/C) → summarize。全链路打通(指标因 toy 训练无意义，仅验证管线)。
  - 修复 torch 2.2.2 AMP 兼容(GradScaler)、checkpoint 存 tensor 而非 ndarray(weights_only 兼容)。
- **额外安装依赖**：torchmetrics, scikit-image, timm, einops, safetensors, opencv-python-headless==4.10.0.84；**numpy 锁回 1.26.4**(scikit-image 曾拉到 2.4 破坏 numba)。
- **遇到的问题**：评估脚本依赖链很长且多为原作者集群环境遗留(失效绝对路径/缺包/硬 raise)；numpy 版本冲突。
- **解决方案**：可选依赖一律惰性化+优雅降级；numpy 显式锁版本 + opencv 降级到 numpy<2 兼容版。

### 2026-06-18 — 代码补全：规格对齐 + 修复（按 implementation.md/idea_report.md 核查）
- **背景**：对全仓做 claim 级核查，发现若干处「能跑通烟测但不符合文档规格」或「破损」的代码，逐一补全/修复并在 `envs/Gene2Image`（torch 2.x + gseapy）本地单元验证。生成主干 `rectified_flow.py`/`unet.py`、扩散 `baseline/*`、单次 80/20 `test≡val` 等文档 `[KEEP]`/明示选择项**未改动**。
- **完成内容**：
  1. **`analysis/pathway_interpret.py` 子分析 B 重写为真正的 GSEA 一致性（§3.8 B/§2.4 B）**：读 GeneFlow `gene_importance_scores.csv`，对模型自身 Hallmark 通路集做 `gseapy.prerank` 富集，输出 **top-k 重合率 + Spearman 排序相关** → `gsea_consistency.json`。`analysis_A` 改为累计每通路平均注意力（不再只取 argmax）。**删除**原「无参照时与自身比 → overlap 恒=1.0」的自证逻辑；缺 `--geneflow_importance` 时明确跳过。单测：一致→overlap 1.0/ρ+1，反序→overlap 0/ρ−1。
  2. **`rectified_generate.py` 导入修复**：`*_deprecation` 改可选 try/except（与 eval 一致）、`rectified.rectified_utils`→`rectified.utils`；导入通过。
  3. **跨数据集迁移评估（§2.3）修复**：原 eval 优先用 checkpoint 内**源**掩码、静默忽略目标 `--pathway_mask` → 跨 panel 用源基因索引目标基因（越界）。新增 `--cross_dataset_eval`：在**目标**掩码上重建编码器，按 (通路,基因) 名把源训练的 `W`、按通路名把 `bias` 移植，transformer/head/UNet 1:1 载入；`rectified_main.py` 的 `model_config` 增存 `gene_names`；`run_cross_dataset.sh` 目标评估加该标志。单测：目标面板前向无越界、共享边权重正确迁移、UNet 1:1。
  4. **`build_pathway_mask.py` 增表达派生 ssGSEA 权重（§3.2 待办）**：`--ssgsea_mode {equal,expression}`，expression（默认，需 `--adata`）按通路内训练集平均表达加权，弱化 PathPrior 等权代理对 RQ3 公平性的影响；equal 仍可回退。
  5. **`summarize_results.py` 补跨数据集汇总（§5.6）**：新增 `cross_dataset/summary.csv`（model×setting 的 fid_cross/fid_same/**degradation_rate**）；主表过滤 `eval_on_*` 子目录，避免跨数据集结果污染主表。
  6. **数据准备脚本明显运行时 bug**（非主训练链路）：`prepare_xenium_data.py` `normalize_image(convert_to_uint8=)`→`convert_to=np.uint8`（2 处）、`adata.to_df`→`adata.to_df()`；`add_coordinates_to_patch.py` 单样本分支传 list→`args.sample_id[0]`。
  7. **统一实验编排 `scripts/run_all.sh`（全部实验一个脚本 + GPU 任务队列）**：覆盖 2.1 主实验 / 2.2 消融（+可选 Reactome）/ 2.3 跨数据集 / 2.4 RQ4，含前置（fix_image_paths、build_pathway_mask、build_cross_masks，一次性串行做以避免并行任务争抢同一 .npz）、末尾 `summarize_results`、并生成自述目录 `results/EXPERIMENTS_CATALOG.md`。第一个参数 = 最大并行任务数（=GPU 数，一任务一卡 `CUDA_VISIBLE_DEVICES`，`wait -n` 队列，做完一个补一个）；阶段化（训练→RQ4→汇总）。配套把 `run_experiments.sh` 改为「训练+评估」一体（每个 run 产出 `evaluation_summary.json`，eval 用同 `--seed` 保证同一 80/20 val 划分），并修 `run_cross_dataset.sh` 两处 eval 漏传 `--seed`（seed 43/44 会评在错划分上）的隐患。`DRY_RUN=1` 可预览全计划。
  8. **并发默认值与对抗性审查修复**：`run_all.sh` 最大并行默认改为 10（`MAX_PARALLEL=${1:-10}`）。对全部改动跑了多智能体对抗性审查（4 维并行 + 逐条核验），确认并修复 6 处：① 调度器在 GPU 池为空（`MAX_PARALLEL=0` / `GPUS=" "`）时会忙等死循环 → 加 GPU 槽位/整数校验，空池快速报错退出、空白 GPUS 回退到 `MAX_PARALLEL`；② 跨数据集产物落盘位置与 catalog 不一致 → cross 任务显式 `OUT_ROOT=$OUT_ROOT/cross_dataset`；③ 掩码前置守卫只查 real.npz → 改为 real/rand/none 三件齐备才跳过；④ `analysis_C` 干预与基线用不同随机噪声致 specificity_ratio 被噪声主导无意义 → base 与每次干预共享同一初始噪声（固定 seed）；⑤ `load_model` `list(pathway_names)` 在值为 None 时 TypeError → `or []` 归一；⑥ `load_model` 缺 model_type 守卫 → multi checkpoint 现给清晰报错。另两条被核验为**误报**仍做零风险加固：跨设备索引赋值（实测 PyTorch 隐式拷贝不报错，仍加 `.to(device)`）、expression ssGSEA 全零通路行（加等权回退保证行和=1）。
- **验证**：全部编辑文件 `py_compile` 通过、模块 import 通过、`bash -n` 三个脚本通过；上述功能均有本地单元/集成测试通过（含真实 `RNAtoHnEModel` 跨面板移植+前向、GSEA 一致性、调度器并发+空卡复用、空池快速报错、ssGSEA 零行回退、跨数据集移植回归）；`DRY_RUN` 全计划（63 train+eval + 3 interpret + 9 前置；含 Reactome 时 66 + 10）正确。**仍未代跑**：真实 100ep 多种子完整训练（数据/算力在用户外部服务器）。

## 已知问题
- [ ] 真实多种子完整训练(100ep)未代跑(按用户指示，~12h/run)；脚本就绪可手动启动
- [ ] UNI2-h FID / RNA round-trip 需 gated 权重，当前降级为 N/A；需要时申请权重并放对路径
- [x] ssGSEA 权重派生：已加 `--ssgsea_mode expression`（默认，训练集平均表达加权）；`equal`(1/k) 可回退（2026-06-18）
- [ ] P1+Reactome P=1666 远大于估算 600，附加消融显存需留意
- [ ] 真实训练显存：V100 32GB < 估算 78GB，需调小 batch / 开 AMP
- [ ] C2/P1 的 cell_image_paths 仍需各跑一次 fix_image_paths.py（训练前）
- [x] cell_image_paths.json 路径失效（`/depot/natallah/...`）→ 已修复脚本，C1 已处理
- [x] rectified_evaluate.py 导入不存在的 `*_deprecation` → 已 try/except 修复
- [x] rectified_generate.py 导入 `*_deprecation`/`rectified_utils` → 已修复（2026-06-18）
- [x] rectified_train.py L1 正则 → 已 `compute_l1_penalty` 解耦（通路编码器走 `l1_penalty()`）
- [x] 子分析 B 仅做通路名集合交集且默认 overlap=1.0 → 已重写为 GSEA 富集 + top-k 重合 + Spearman（2026-06-18）
- [x] 跨数据集评估忽略目标掩码（用源基因索引目标基因，越界）→ 已加 `--cross_dataset_eval` 按名移植权重（2026-06-18）
- [x] 掩码列对齐运行期仅校验列数 → 已加 `rectified_main.py` 逐基因名顺序比对（npz gene_names vs dataset gene_names 不一致即硬报错，杜绝静默 token 污染）（2026-06-18，验收审计）
- 验收审计（2026-06-18，多智能体逐条核对用户要求+项目文档要求）：25/25 需求满足、0 阻断/破损；上述列对齐为审计中唯一稳健性提示，已闭环。
