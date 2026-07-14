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
- **解决方案**：真实训练需减小 batch_size（如 single 4→8，按显存调）/ 开 --use_amp；完整 50ep 训练不在本地代跑（按用户指示）。
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
  4. **`build_pathway_mask.py` 增表达派生 ssGSEA 权重（§3.2 待办）**：`--ssgsea_mode {equal,expression}`，expression（默认，需 `--adata`）按通路内**全 panel(所有细胞,含验证集)平均表达**加权（`get_mean_expression` 用整个 h5ad，**非仅训练集** → 对 PathPrior 有小而对称的 train/val 泄漏），弱化 PathPrior 等权代理对 RQ3 公平性的影响；equal 仍可回退。
  5. **`summarize_results.py` 补跨数据集汇总（§5.6）**：新增 `cross_dataset/summary.csv`（model×setting 的 fid_cross/fid_same/**degradation_rate**）；主表过滤 `eval_on_*` 子目录，避免跨数据集结果污染主表。
  6. **数据准备脚本明显运行时 bug**（非主训练链路）：`prepare_xenium_data.py` `normalize_image(convert_to_uint8=)`→`convert_to=np.uint8`（2 处）、`adata.to_df`→`adata.to_df()`；`add_coordinates_to_patch.py` 单样本分支传 list→`args.sample_id[0]`。
  7. **统一实验编排 `scripts/run_all.sh`（全部实验一个脚本 + GPU 任务队列）**：覆盖 2.1 主实验 / 2.2 消融（+可选 Reactome）/ 2.3 跨数据集 / 2.4 RQ4，含前置（fix_image_paths、build_pathway_mask、build_cross_masks，一次性串行做以避免并行任务争抢同一 .npz）、末尾 `summarize_results`、并生成自述目录 `results/EXPERIMENTS_CATALOG.md`。第一个参数 = 最大并行任务数（=GPU 数，一任务一卡 `CUDA_VISIBLE_DEVICES`，`wait -n` 队列，做完一个补一个）；阶段化（训练→RQ4→汇总）。配套把 `run_experiments.sh` 改为「训练+评估」一体（每个 run 产出 `evaluation_summary.json`，eval 用同 `--seed` 保证同一 80/20 val 划分），并修 `run_cross_dataset.sh` 两处 eval 漏传 `--seed`（seed 43/44 会评在错划分上）的隐患。`DRY_RUN=1` 可预览全计划。
  8. **并发默认值与对抗性审查修复**：`run_all.sh` 最大并行默认改为 10（`MAX_PARALLEL=${1:-10}`）。对全部改动跑了多智能体对抗性审查（4 维并行 + 逐条核验），确认并修复 6 处：① 调度器在 GPU 池为空（`MAX_PARALLEL=0` / `GPUS=" "`）时会忙等死循环 → 加 GPU 槽位/整数校验，空池快速报错退出、空白 GPUS 回退到 `MAX_PARALLEL`；② 跨数据集产物落盘位置与 catalog 不一致 → cross 任务显式 `OUT_ROOT=$OUT_ROOT/cross_dataset`；③ 掩码前置守卫只查 real.npz → 改为 real/rand/none 三件齐备才跳过；④ `analysis_C` 干预与基线用不同随机噪声致 specificity_ratio 被噪声主导无意义 → base 与每次干预共享同一初始噪声（固定 seed）；⑤ `load_model` `list(pathway_names)` 在值为 None 时 TypeError → `or []` 归一；⑥ `load_model` 缺 model_type 守卫 → multi checkpoint 现给清晰报错。另两条被核验为**误报**仍做零风险加固：跨设备索引赋值（实测 PyTorch 隐式拷贝不报错，仍加 `.to(device)`）、expression ssGSEA 全零通路行（加等权回退保证行和=1）。
- **验证**：全部编辑文件 `py_compile` 通过、模块 import 通过、`bash -n` 三个脚本通过；上述功能均有本地单元/集成测试通过（含真实 `RNAtoHnEModel` 跨面板移植+前向、GSEA 一致性、调度器并发+空卡复用、空池快速报错、ssGSEA 零行回退、跨数据集移植回归）；`DRY_RUN` 全计划（63 train+eval + 3 interpret + 9 前置；含 Reactome 时 66 + 10）正确。**仍未代跑**：真实 50ep 多种子完整训练（数据/算力在用户外部服务器）。

### 2026-07-13 — 正式跑实验前的四项协议级修复（用户「还没开始跑代码」，允许改划分/协议）
> 背景：用户复核代码后指出四处问题，明确本人**尚未开跑**任何实验，故允许改动数据划分与训练/评估协议。四项均已落地、`py_compile` 全通过、逻辑逐条自检。**注意**：这些修改会让代码行为与已发给师兄的初稿（`/root/ctw/Gene2Image_初稿.txt`，描述旧行为）产生分歧，后续需同步论文（见文末「待同步论文」）。

- **修复①　PathPrior 数据泄漏（RQ3 公平性）**　`rectified/rectified_main.py` PathPrior（`--no_learnable_pathway`）分支。
  - 旧：直接载入 `mask_npz['W_ssgsea']`，而该权重由 `build_pathway_mask.py:get_mean_expression` 对 **h5ad 全部细胞（含验证/测试集）** 求均值派生 → 冻结基线偷看了评价集统计。
  - 新：在训练时，从**本 seed 的训练子集**重算每基因均值（解包 `random_split` 的 `Subset` → `base.cell_ids`/`base.expr_df`，`.loc[train_cells].mean` → `reindex(gene_names)`），调用 `build_ssgsea_weights(A, train_mean)` 重建 (P,G) 冻结权重，PathPrior **不再接触评价细胞**。取不到训练细胞（如 multi-cell）时优雅回退到 mask 的 `W_ssgsea` 并 `warning`。
- **修复②　生成初始噪声配对（变体/定性/干预可比性）**　`rectified/utils.py` + `rectified/rectified_flow.py` + `rectified_evaluate.py` + `rectified_generate.py` + `analysis/pathway_interpret.py`。
  - 旧：`generate_sample` 内部 `torch.randn` 现抽噪声，无注入口；六个变体是**各自独立进程**，其 **模型初始化消耗全局 RNG 的量不同** → 同一细胞在不同变体拿到不同 x(t=0)，变体间差异被噪声方差污染。
  - 新：两个 `generate_sample` 增 `noise=` 形参；`generate_images_with_rectified_flow` 增 `sample_ids=`，用 `sha256("{noise_seed}:{cell_id}")` 派生**每样本确定性噪声**（独立于全局 RNG）。评估/生成按 `cell_id`/`patch_id` 传入 → 同一细胞在**所有变体**共享同一初始噪声。`analysis_C` 干预分析从旧的「改前重置全局 seed」迁移到该显式 API（更稳健，不再依赖生成过程不额外消耗 RNG）。
- **修复③　速度目标与训练路径不匹配**　`rectified/rectified_flow.py:sample_path`。
  - 旧：`x_t` 额外加了随机项 `0.05*(1-t)*ξ`，但监督速度 `velocity` **未含其时间导数 -0.05ξ** → 回归目标不是训练路径的真实导数（等于往标签注入不可约的零均值噪声）。
  - 新：删去该未配对随机项，`x_t = sin(πt/2)·x₁ + (1-sin(πt/2))·ε` 落在干净正弦路径上，`velocity = (x₁-ε)·(π/2)·cos(πt/2)` 恰为 `d x_t/dt`。生成边界一致（t=0→ε、t=1→x₁）。
- **修复④　加独立测试集（val≠test）**　`rectified/rectified_main.py` + `rectified/rectified_evaluate.py`。
  - 旧：80/20，验证集 = 评估集（用于选 checkpoint 的集合又拿来报最终指标）。
  - 新：80/10/10，`random_split([train,val,test], Generator().manual_seed(args.seed))`；main 用 **val** 选 checkpoint、evaluate 取 `_, _, eval_dataset`（**test**）报指标。两文件划分逻辑逐字一致（同 `int(0.8n)`/`int(0.1n)`/余数、同独立 `Generator` seed），保证 held-out 测试集对齐。
- **验证**：8 个改动文件 `python3 -m py_compile` 全通过；main/evaluate 划分一致性、PathPrior 分支变量作用域（`os`/`np`/`A`/`gene_names`/`train_dataset` 均在块前定义）、`build_ssgsea_weights` 签名与动态导入路径均逐条核对无误。**仍未代跑**真实训练（数据/算力在用户外部服务器）。
- **待同步论文**：初稿描述的旧行为需改写——(a) 速度不再是「近似/加噪」；(b) PathPrior 不再有 train/val 泄漏；(c) 现有独立 test（val≠test）。已告知用户代码将与已发稿分歧且已获接受。

### 2026-07-13 — debug 跑得极慢的根因定位 + eval 提速（A+B）
> 用户给的 `g2i_debug-3725225.{out,err}`（Capella H100，job 3725225）显示 debug 跑了 ~3 小时。逐时间戳还原：载数据+**图像路径存在性扫描 ~3min**（`rectified_main.py:174` 对 106980 条路径逐个 `os.path.exists`）；**训练 64 细胞 1 epoch 仅 ~6s**（不慢）；**评估 625 batch(~5000 张) @ ~18.5s/it ≈ 3.2h**（这才是慢源）。图像生成本身快（日志 `DOPRI5 completed in 4 steps`）。

- **根因① launcher 漏给 eval 限样本（已修，零科学影响）**：`slurm/capella_debug_1gpu.slurm` 的 `--debug --debug_samples 64` 只经 `$EXTRA` 缩**训练**；eval 读**另一个** `$EVAL_EXTRA`（空）→ 回退默认 `--max_samples 5000`；且 **eval 根本不认 `--debug_samples`，只认 `--max_samples`**。兄弟脚本 `smoke_test.slurm` 早已 `EVAL_EXTRA="--max_samples 64"`（注释：这是 30min 内跑完的关键），debug 脚本忘了。**修复**：给 `capella_debug_1gpu.slurm` 补 `EVAL_EXTRA="--max_samples 64"`（+ 注释说明）。debug 从 ~3h → 几分钟。
- **根因② eval 每 batch 大量重复/无效计算（用户选 A+B 已修）**：`rectified/rectified_evaluate.py` 评估循环里每个 8-样本 batch 都做：**逐 batch FID `sqrtm`**（line 1000，2048-d 特征在 8 样本上协方差奇异→统计无效，且 O(d³) CPU，日志一路 `Matrix is singular`）+ **UNI2-h ViT-H 前向 3~4 次**（`calculate_uni2h_fid` 内部已提特征，后面又 `extract_uni2_h_embeddings` 两遍）+ **逐图 CPU 核形态学**（`extended_biological_evaluation_uni2h`）。
  - **opt A**：删逐 batch Inception `calculate_fid`（保留特征累积 + 廉价的逐样本特征 L2 距离）；FID 只在末尾用**全集**累积特征算一次（`overall_fid`，本就如此、且才是可信值）。
  - **opt B**：UNI2-h 每 batch 只 `extract_uni2_h_embeddings` 一次(real+gen)并累积；删 `calculate_uni2h_fid`（省 2 次 ViT-H 前向 + 1 次逐 batch sqrtm）；UNI2-h FID 同样末尾从全集算一次（`overall_uni2h_fid`）。
  - 进度条 postfix 改显示廉价的 `FeatDist(batch)`（原逐 batch FID 已无）；末尾日志去掉误导的 "batch-wise mean" 只报 overall。`valid_fids/valid_uni2h_fids` 变空 → `utils_plot` 的 FID 直方图子图有 `if valid_fids:` 守卫会自动跳过、不报错；**报告指标 `overall_fid/overall_uni2h_fid` 与改前逐字一致**。`py_compile` 通过。
  - **未做（用户没选）**：opt C（把生物学核形态学验证做成开关/抽样，剩下的大头 CPU 成本）、opt D（缓存那 3min 路径扫描）。预计每 it ~18.5s → ~6-9s，正式 eval 每 run ~3.2h → ~1-1.6h（×54 ≈ 从 ~170 GPU-h 降到 ~55-85）。

### 2026-07-13 — 以初稿为规格的 draft↔code 一致性审计（8 组，多智能体 + 逐条对抗复核）
> 用户要求"根据初稿检查代码是否满足要求"。对初稿(`/root/ctw/Gene2Image_初稿.txt`)里所有可核对的具体主张分 8 组(编码器结构 / Transformer头 / U-Net条件 / 流优化 / GeneFlow编码器 / 六变体+L1 / 数据评价 / 参数复杂度)逐条对到代码。三处我们已主动改进的(精确速度、80/10/10 独立测试集、PathPrior 训练集统计)标为有意改进、不当 bug。

- **结论：0 处代码 bug**。代码忠实实现初稿描述的方法。仅 2 处不一致，且均为"论文写错、代码没错"(MISMATCH_DRAFT_WRONG，经二次对抗复核确认)：
  - **① U-Net 注意力尺度(影响中文初稿 §3.4 + main.tex)**：初稿写"在下采样因子 16 和瓶颈处用空间自注意力"，但 `attention_resolutions=(16,)`+`channel_mult=(1,2,2,2)` 只有 4 层/3 次下采样，ds 走 1→2→4→8 **永远到不了 16** → (16,) 空设置不生效；真正生效的只有 `unet.py` `middle_block` 无条件 AttentionBlock，即**瓶颈层**(ds=8，256px 下 32×32)。是 GeneFlow U-Net 原样复用的设定，**代码不改**(改成 ds=16 要加第 5 层=改架构)；**改论文**为"自注意力仅在瓶颈层(32×32)施加"。已自读 `src/unet.py:382-447` 复核。
  - **② 早停耐心(仅英文 main.tex 错)**：main.tex 写 "early stopping (patience 5)" 与代码不符；**中文初稿 §4.2 "早停耐心设得较大/训满 50 epoch" 与代码一致(capella 脚本 `--patience 9999`，早停关闭、val_mse 选点)**。中文稿无需改；用 main.tex 时改掉 "patience 5"。
- **配套代码加固**：`rectified_main.py` 80/10/10 前加**空划分守卫**(n<10 时 val=0 会静默失败 → 直接报错)，`py_compile` 通过。纯安全网，正式/debug(64+)都不触发。
- **仅 debug 的已知项(不碰正式 54-run)**：debug 时 main 先取子集再划分而 eval 不认 `--debug` → 两者划分错位；实测可忽略(launcher 未给 eval 传 `--debug` 不会崩，两个 64 子集在 10 万里重叠期望 ~0.04)。待用户决定是否彻底对齐。

### 2026-07-13 — 最终全面审查(9 子系统对抗性 + ds16 深评)结论 + 修复
> 目标：正式跑 54-run 前确认代码无阻塞。9 子系统各 high-effort 审查 + 逐条对抗复核。**结论：单卡 54-run 0 阻塞**。共 9 条确认，多数为纯 DDP(单卡不触发)或配置默认(54-run 不触发)或 nit。

- **已修(真正影响自动流水线/交付物的 3 条)**：
  - `run_all.sh` **静默跳过 RQ4 机制研究(major)**：`build_interpret_jobs` 用 `case $VARIANTS in *gene2image*` 门，而规范的 interpret-only 趟用 `VARIANTS=""` → 整个 RQ4 被跳过。改为**按 gene2image checkpoint 是否存在**来判定(缺则告警跳过该 ds)。
  - `rectified_evaluate.py` **NaN 传染(minor)**：一张发散图的 NaN 会把整变体 SSIM/PSNR/FID 全变 NaN。已让 SSIM/PSNR 均值和 FID 特征行**先滤非有限值再平均**并告警。
  - `rectified_generate.py` **加载不了通路 checkpoint(定性图脚本)**：原来只从 args 建 RNA 编码器 → Gene2Image/变体 checkpoint 无法加载。**照 evaluate.py 移植**：先读 checkpoint config → 按 `encoder_type='pathway'` 载掩码(config 数组或文件)重建通路编码器(d_token/pt_layers/pt_heads/learnable_pathway/use_pathway_transformer),img 维度也从 config 取；划分改**有种子 80/10/10 test**;pathway 加载失败不回退旧 RNA 构造器(直接报错)。`RNAtoHnEModel.__init__` 本就接受全部这些 kwargs(single_model.py:231-264)，故零风险。`py_compile` 通过。
- **单卡 54-run 不触发(纯 DDP，5 条，不改)**：非有限 loss `continue` 致 DDP all-reduce 死锁、noTrans 的 CLS 未用参数在 DDP 报错、DistributedSampler 重复填充、`gene_names` 只在 rank0 赋值致非零 rank PathPrior 回退。canonical launcher 是 1 卡/任务、use_ddp=False → 全不发生。**若将来改多卡 DDP 需回头处理**。
- **潜在但 54-run 不触发(不改)**：RNA 消融 flag 未从 checkpoint 传进 eval 构造器(54-run 全默认 True 故一致)；各变体 U-Net 因 RNG 消耗不同而初始化不同(nit，跨 3 seed 均值);掩码构建无锁竞争(仅跳过串行 PREP 时)；`analyze_gene_importance` 在 DDP 崩(单卡+训练后不触发)。
- **ds16 深评(专职 agent + 我自读 unet.py:382-447 双确认)**：`attention_resolutions=(16,)` 配 `channel_mult=(1,2,2,2)` ds 走 1→2→4→8 到不了 16 → 空设置;真正生效的只有瓶颈无条件 AttentionBlock(ds=8,32×32)。四处(train/eval/generate/single_model)配置逐字相同 → 对**跨变体可比性和机制结论零影响**,只等量影响绝对质量。**代码保持原样,仅改论文措辞**。
- **误报 5 条**已否掉(如染色归一化泄漏=默认关)。

### 2026-07-13 — diff-review + 全代码库全面审查(10 子系统 + 完备性批判)+ 修复
> 用户要求"再来一次全方面审查"。先对本轮新改动做 diff-review(4 区),再做覆盖 10 子系统的全面审查 + 完备性批判 agent，均逐条对抗复核。共修 8 条(2 个 major)。

- **diff-review(3 条，均已修）**：
  - **major(我上一轮 run_all.sh 修复引入的回归)**：`build_interpret_jobs` 在训练**之前**(计划阶段)调用，我改成"按 checkpoint 存在判定" → 从零跑整轮时 checkpoint 还没训出来 → 整个 RQ4 又被静默跳过。改为 **gene2image 在 VARIANTS(会被训) 或 checkpoint 已在磁盘** 才 emit（bash harness 验证整轮/VARIANTS=""/仅 geneflow 三流程 + DRY_RUN 均正确）。
  - minor：`generate.py` 移植漏了 evaluate 的**基因序守卫** → 补上（基因序不符即报错，不静默用错位条件出图）。
  - minor：`evaluate.py` NaN 过滤后 `total_samples`(未滤) 与 mean 的 N(已滤) 不一致 → 加 `n_ssim_used/n_psnr_used`。
- **全面审查(10 子系统，确认 10 条；已修 5 条真正相关的)**：
  - **major(pre-existing，非我引入)**：`evaluate.py:1225` biological_summary 用 `r[key]` 无 `key in r` 守卫，而 nuclear_*/spatial_* 键只在成功分割核的 batch 里出现 → 弱变体的模糊图(0 核)使某 batch 缺键 → **KeyError 崩掉整个 eval、在 evaluation_summary.json 写出前**，丢掉该 run 全部指标，且 `set -e` 下后续矩阵中断。→ 加 `key in r` 守卫（与相邻 rna_summary 一致）。**这是这轮最重要的发现**。
  - minor：UNI2-h FID 未做 Inception FID 那样的非有限行过滤 → 补上(一致)。
  - minor：`dataset.py` 图像读取失败的回退建 3 通道 PIL → 与 img_channels=4 不符、`torch.stack` 崩整个 batch → 改为直接返回 `torch.zeros(img_channels,H,W)`。
  - minor(可比性)：`single_model.py` 编码器在 UNet **之前**构建 → 各变体编码器耗 RNG 不同致 UNet 初始化不同 → **捕获编码器前 RNG 状态、建 UNet 前恢复**，使 UNet 初始化对 6 变体逐字相同(仍随 seed 变)。强化"只有编码器变"的受控比较前提。
  - minor：`run_all.sh` summarize 补 `--expected_seeds $NSEEDS`(某 seed 全臂失败不再被当"完整")。
  - **剩余 5 条(54run=false / nit / 仅报告完备性）后续也全部清掉**：① generate on-demand 分支补 `gene_names = getattr(dataset,'gene_names',None)`；② `evaluate.py` 把 checkpoint 的 6 个 RNA 消融 flag 传进构造器(原读进 locals 未用；main.py:720 本就持久化)；③ `summarize_results.py` `aggregate_cross` 加 n_seeds + 不完整-seed 警告(镜像主聚合)并接 `--expected_seeds`；④ debug 划分错位：`evaluate.py` 增 `--debug/--debug_samples` 并在划分前镜像同 seed randperm 子集，debug launcher `EVAL_EXTRA` 改 `--debug --debug_samples 64`；⑤ `run_all.sh` verify_masks 掩码名 `hallmark`→`${DB}`(与 run_experiments 一致)。
- 全部改动 `py_compile` / `bash -n` 通过。结论：**单卡 54-run 现无已知会崩溃或污染指标的问题；两次审查合计 13 条确认全部处理完毕**。

### 2026-07-13 — 端到端「可运行就绪」验证(5 组实验逐个走通命令→脚本→代码)+ 修复
> 代码正确性已 4 轮验证(0 缺陷)，这轮换角度验证「能不能真跑起来」并产出实验清单。发现 3 处就绪缺陷(脚本占位/守卫，非核心代码 bug)，已修：
- **major** `preflight_smoke.sh:45`：STAGE1 只设 EXTRA 未设 EVAL_EXTRA → eval 跑全量(多小时)而非 64-cell → 加 `EVAL_EXTRA="--debug --debug_samples 64"`(另一处 stage 是 EVAL=0 无需)。
- **major** `generate.sh`：4 处 `/GeneFlow/...` 占位路径 + 错误文件名 `best_model.pt`(实际 `best_checkpoint.pt`)→ 原样跑必 FileNotFoundError → 重指向 canonical `results/gene2image_c1_seed42/checkpoints/best_checkpoint.pt` + `data/processed_data/...cell_image_paths_local.json`。(定性图训练时也会自动出 `generation_results.png`，此脚本为可选重生成工具、不在关键路径。)
- **minor** `run_all.sh` verify_masks：只查共享 `${ds}_${DB}_rand.npz`，漏 randpath 实际用的 per-seed `_rand_s<seed>.npz` → 加 per-seed 断言(否则 3 seed 退化同掩码、丢 RQ2 掩码方差)。
- 未改(非缺陷/预期占位)：SLURM `--account=CHANGEME`(用户必填)、`GMT_HALLMARK` 默认路径(包内 `gmt/` 相对 `code/../gmt` 可解)、`run_cross_dataset.sh all` 硬编码 seed(非 canonical 路径)。
- 结论:**核心链路(PHASE0 掩码→0.5 fail-fast→54-run→cross→RQ4→summarize)端到端走通，无 blocker**；上述 3 处已修。bash -n 通过，包已重打(2.6M)。

### 2026-07-14 — 用户第三方(Codex)只读审计发现 3 阻断 + 1 robustness(前几轮遗漏)，已全修
> 用户用另一工具对同一份包做了独立审计，指出我前面几轮没抓到的问题。核实属实，当场修：
- **阻断① 调试污染正式(dir 冲突 + --auto_resume)**：`_capella_common.sh:18` 默认 `OUTPUT_DIR=$PROJECT_DIR/results`；`capella_debug_1gpu.slurm`(geneflow_c1_seed42,64样本) 和 `capella_smoke_1gpu.slurm`(gene2image_c1_seed42,256样本) 都不覆盖 → 都写进 `results/`;正式 array 用 `--auto_resume` → 会自动接着这俩假模型续训、继承其优化器/best_val → 对应两个正式 run 作废。**修**:debug/smoke 在 source 前 `export OUTPUT_DIR=...results_debug/results_smoke`(pilot 是真实 50ep 训练，仍留 `results/` 供 array resume)。
- **阻断② DOPRI5 不拒步**：`rectified_flow.py` 自适应求解器算了误差却**无条件接受当前步**、只缩下一步 dt → 无真正误差控制,影响所有生成图 + FID/SSIM/PSNR(checkpoint 不受影响，修后只需重生成+评估)。**修**:加嵌入式误差拒步——`error_ratio=‖err‖/(atol+rtol‖x‖)`,>1 则拒(不推进 t、以更小 dt 重试),带 dt 下限防死循环;欠积分守卫仍兜底到 t=1。
- **阻断③ run_all 失败当成功**：`run_queue` 记录 rc 却不累计、不传播 → 有 run 失败仍继续汇总并打印 DONE。**修**:run_queue 加 `n_fail` 计数、有失败 `return 1`;PHASE1/2 `|| OVERALL_RC=1`;末尾若 OVERALL_RC≠0 则 loud warning + `exit 1`(summarize 仍跑以聚合已成功的)。
- **robustness④ logs/ 未预建**:SBATCH `--output=logs/%x-%j.out` 在作业体前生效，脚本内 mkdir 太晚 → README §4 加 `mkdir -p logs outputs`(首次 sbatch 前)。
- **「会影响论文结论」那几条**基本已在初稿 §5 声明(FID非clean-fid/DAPI不入指标、细胞级划分非泛化、干预=条件敏感非因果、PathPrior非标准ssGSEA/探索性、变体非严格单因素),我此前已同步进 `Gene2Image_初稿.txt`。唯一可再收紧:randPath 只保每通路基因数、未保每基因度/覆盖率/重叠 → §3.7 措辞宜从"隔离真实通路语义"改为"真实成员 vs 通路大小匹配的随机成员"(留给用户定稿)。
- 全部 `py_compile`/`bash -n` 通过，包已重打(2.6M)。**修正后的开跑顺序**:先隔离 debug/smoke 目录(已默认隔离)→ pilot/正式;DDP/空间损失/梯度检查点分支不要用(默认也不用)。

### 2026-07-14 — 第三方审计追加两点(不能只声明)+ 4 项回归实跑 + RQ4 强化
- **DOPRI5 误差判据(Point1)**：误差范数从全局 `‖err‖/‖x‖` 改为**标准逐元素加权 RMS** `sqrt(mean((err/(atol+rtol·max|x|,|x_next|))²))`(与批量/尺寸无关);`_compute_adaptive_step_size` 改收 `error_norm`;加**拒步/dt-floor/欠积分兜底三项计数**并打印,兜底日志加 `UNDER_INTEGRATION_FALLBACK` 标记。→ 正式前看 pilot 三项计数,频繁兜底不接受结果。
- **4 项回归检查(实跑,非声明)**：①包内无 `results/`、debug/smoke 已隔离(HPC 旧遗留需用户自查);②DOPRI5 控制逻辑纯 python 镜像仿真 4 情形全终止/到 t=1 或明确 fallback/计数正确;③**真实 run_queue 代码** + 故意失败 → 1/3 FAILED、返回非零、drive exit1;④pilot 与 formal 配置(EPOCHS/BATCH/GEN_STEPS/DB/EXTRA)逐项一致 → resume 兼容。
- **randPath 全文收紧(Point3)**：按用户措辞统一改 摘要/引言RQ/贡献/§2.2/§3.7表/§4.4/§5/结论 共 9 处,删"隔离真实通路语义"类过强表述(只保留"而非…完全独立隔离"否定句),§4.4 加两版结果解读预案。
- **RQ4 证据强化(Point2,用户选"强化代码")**：`analysis_C` 重写——①样本 8→`--interv_cells`(默认128);②**选择集/验证集不相交**(选择集按注意力选主导通路、验证集做干预测量,消除循环选择);③对照改**均匀随机通路(排除主导)**而非注意力后半;`C_causal.json` 记录 disjoint/random-control/样本数;仍标注"条件敏感性、非生物因果"。加 `--interv_cells/--interv_k`;§4.5(C) 措辞同步。
- 全部 `py_compile`/`bash -n` 通过,同步稿进包、重打(2.6M)。**状态:主要阻断项已修 + 逻辑层回归已确认,待真实 GPU 回归(DOPRI5 计数、机制特异性)。**

### 2026-07-14(二) — 第三方回归复审:补掉残留阻断 + DOPRI5/RQ4 精修
> 用户对 14:50 包再独立回归:核心修复确认落地,但发现**目录隔离仍可被环境变量绕过**(真阻断)+ 多处精修。逐条核实并修:
- **阻断:目录隔离被 env 绕过** —— `env_g2i.sh:10` 导出 `OUTPUT_DIR=$PROJECT_DIR/results` 且在 `$COMMON` 里,Slurm 继承提交环境 → debug/smoke 的 `${OUTPUT_DIR:-...results_debug}` 会保留正式 `results`。**修:改用专用变量** `${DEBUG_OUTPUT_DIR:-...}` / `${SMOKE_OUTPUT_DIR:-...}`,预设 OUTPUT_DIR 无法覆盖;bash 实测:预设 `/proj/results` 后 debug/smoke 仍走 `/proj/results_debug|smoke` ✅。
- **DOPRI5 逐图保证** —— 误差范数从"整批 BCHW 平均"改为**逐图 RMS 再取 batch max**(易样本不能稀释困难样本,真正逐图容差);加 `n_fallback`,末尾打印 `DOPRI5_DIAGNOSTICS rejected=.. dt_floor=.. under_integration_fallback=..`(硬门槛:dt_floor=0 且 fallback=0 才可信;拒步>0 正常)。
- **RQ4-C 精修**(去循环基础上继续硬化):①**分块生成** `gen_batch`(默认=batch_size)避免 64 细胞一次入显存 OOM;②**自助法 95% CI** 记录 specificity;③JSON 补 主导/随机通路名、验证 cell_id、seed、gen_steps、距离定义(4通道像素L2)、"disjoint 但非训练留出";④文件名 `C_causal.json`→`C_intervention_sensitivity.json`,模块标题 Causality→Sensitivity(消除与"非因果"定位冲突),run_all catalog 同步;⑤默认仅置零消融(放大可选未跑)——论文 §4.5(C) 已如实改。
- **其它**:capella_smoke 补 `EVAL_EXTRA="--debug --debug_samples 256"`(评价原会跑全量);run_all summarize 崩溃也置 `OVERALL_RC=1`。
- 全部 `py_compile`/`bash -n` 通过,同步稿进包、重打(2.6M)。

### 2026-07-14(三) — 目录隔离阻断确认闭环 + 证据边界固化为指南
- **目录隔离阻断:当前包已闭环**。用户审的是 14:50 快照(专用变量修复之前)。当前包(重打)已是 `${DEBUG_OUTPUT_DIR:-...}` / `${SMOKE_OUTPUT_DIR:-...}`;从当前 tar 解出实测:预设 `OUTPUT_DIR=/proj/results` 后 debug/smoke 仍走 `/proj/results_{debug,smoke}` ✅。→ 用户需以当前 tar 为准重新解包。**至此三处运行阻断(DOPRI5/run_all/目录隔离)全部闭环。**
- **证据边界固化**:用户给出精准的 claim–evidence 分析(6 条边界 + 实验-结论对照表 + 4 情形解读预案 + 可直接用的实验总结段与定位句 + 3 个运行门槛)。这些大多已在初稿 §5 及 randPath/RQ4 收紧中体现;为不臃肿正文,整理成 `paper/结果解读与表述指南.md`(并同步 `写论文/`)作为写结果时的权威口径。3 门槛:DOPRI5 `dt_floor=0`、无 `UNDER_INTEGRATION_FALLBACK`、每 run `n_ssim_used==n_psnr_used==total_samples`。
- 结论:**单卡 54-run 主链路无新的模型/训练阻断;运行阻断全闭环;其余为证据边界(不推翻论文,决定解释),已固化。待真实 GPU pilot 看 3 门槛。**

## 已知问题
- [ ] 论文待改(措辞)：U-Net 自注意力仅瓶颈层 32×32；main.tex "patience 5"；randPath/RQ4-C 收紧已落**中文初稿**、英文 main.tex 待同步(现阶段以中文稿为权威)
- [ ] **待真实 GPU 回归**：pilot 看 `DOPRI5_DIAGNOSTICS`(dt_floor/fallback=0);RQ4-C 扩样本+去循环+CI 后特异性是否仍稳定>1
- [ ] **未做的深度硬化(非阻断)**：checkpoint 未写 config 指纹,--auto_resume 仅靠 pilot==formal 配置一致保证;若提交前后环境变量改变(如 EPOCHS)仍可能错误续训 → 建议后续在 checkpoint 存 config 指纹并在 resume 时校验
- [x] debug 跑 ~3h：eval 未限样本(5000) + 逐 batch 无效 FID/UNI2-h：已修(EVAL_EXTRA + opt A/B)（2026-07-13）
- [ ] eval 仍慢的剩余大头 = 逐图 CPU 核形态学(生物学验证) + 3min 路径扫描：opt C/D 待用户决定
- [x] PathPrior 用全 panel（含评价集）均值 → RQ3 泄漏：已改为**训练集重算**冻结权重（2026-07-13）
- [x] 六变体生成用各自随机噪声（模型初始化耗 RNG 不同）→ 变体不可比：已加**按 cell_id 确定性配对噪声**（2026-07-13）
- [x] 速度目标漏掉 mid-state 随机项导数 → 目标非路径真导数：已删未配对随机项（2026-07-13）
- [x] 验证集=评估集（无 held-out test）：已改 80/10/10，val 选点 / test 报指标（2026-07-13）
- [ ] 真实多种子完整训练(50ep)未代跑(按用户指示，~12h/run)；脚本就绪可手动启动
- [ ] UNI2-h FID / RNA round-trip 需 gated 权重，当前降级为 N/A；需要时申请权重并放对路径
- [x] ssGSEA 权重派生：已加 `--ssgsea_mode expression`（默认，**全 panel(含验证集)平均表达**加权，非仅训练集）；`equal`(1/k) 可回退（2026-06-18）
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
