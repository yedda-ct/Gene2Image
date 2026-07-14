# Gene2Image — 跑代码指南

Xenium 空间转录组 → H&E 图像生成。命令里 `$CODE` = 本目录下的 `code/`。

矩阵：6 变体（`gene2image` / `geneflow` / `randpath` / `pathprior` / `notrans` / `nomask`）× 3 数据集（`c1`/`c2`/`p1`）× 3 seed（42/43/44）= **54 主 run** + cross-dataset + RQ4 interpret。
数据集名（写死，勿改）：`c1`=`Xenium_V1_hSkin_Melanoma_Base_FFPE`，`c2`=`Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE`，`p1`=`Xenium_Prime_Human_Skin_FFPE`。

**Slurm 脚本速查**（全部单卡；`_capella_common.sh` 集中站点路径 + GPU fail-fast；短 walltime + `--auto_resume` 拆短）：

| 脚本 | 用途 | GPU/CPU/MEM/TIME |
|---|---|---|
| `slurm/capella_debug_1gpu.slurm` | 环境/路径/数据/模型自检（写 `results_debug/`） | 1 / 14 / 120G / 1h |
| `slurm/capella_smoke_1gpu.slurm` | 少量 epoch + resume + 指标（写 `results_smoke/`） | 1 / 14 / 150G / 4h |
| `slurm/capella_train_1gpu.slurm` | 单卡训练（pilot / 单实验补跑，可恢复） | 1 / 14 / 180G / 12h |
| `slurm/capella_array_1gpu.slurm` | **【默认】主 54-run job array**（一实验一任务，`%N` 限流） | 1 / 14 / 180G / 2d·任务 |
| `slurm/capella_train_4gpu.slurm` | （可选）整节点 4 并行单卡 + `BATCH`（含 cross+RQ4+汇总） | 4 / 56 / 700G / 12h |

---

## 1. 必须先外部准备（代码不自动拉取）
| # | 资源 | 从哪来 | 放哪 / 怎么用 |
|---|---|---|---|
| 1 | **3 个 Xenium 样本数据** | Zenodo **17429142**（现成 `wget` 见 `code/GeneFlowREADME.md`） | 解压到 `$CODE/data/processed_data/<三个固定目录名>/`（见 §3） |
| 2 | **Hallmark GMT** | 仓库**已打包** `gmt/msigdb_2023.2_Hs/h.all.v2023.2.Hs.symbols.gmt` | 无需另下；`_capella_common.sh` 的 `GMT_HALLMARK` 已指向 |
| 3 | **Inception-v3 权重**（FID 用） | PyTorch hub（~100MB，见 §2c） | 登录节点预取到 `~/.cache/torch`；离线否则 FID=NaN（已守卫**不崩**） |
| 4 | **UNI2-h 权重**（生物学 FID，可选） | 门控 `MahmoodLab/UNI2-h`（HuggingFace） | 含 `pytorch_model.bin` 的目录，`export UNI2H_MODEL_PATH=该目录` |
| 5 | **Capella 账号** | 你的 ZIH 项目号 | 填各 `slurm/capella_*.slurm` 的 `--account`（占位 `CHANGEME_capella_account`） |

## 2. 建环境 + 装依赖（登录节点，一次）
```bash
cd $CODE
# (a) 建 venv 并装 torch/torchvision（cu121 wheel）
bash slurm/setup_env_pip_alpha.sh                    # 可用 VENV_DIR=... 覆盖
source /data/horse/ws/<你的>/venv_piptorch/bin/activate
# (b) 其余依赖（含 opencv/gseapy/scanpy/timm/matplotlib 等）
pip install -r requirements.no_torch.txt
# (c) 预取 Inception-v3 权重（FID 需要；离线计算节点首次会联网下载→崩）
python -c "import torchvision.models as m; m.inception_v3(weights='IMAGENET1K_V1')"
```
Python 3.11 + CUDA 12.1；`numpy` 保持 1.26.4。

## 3. 放数据（目录名严格如下）
```
code/data/processed_data/
├── Xenium_V1_hSkin_Melanoma_Base_FFPE/            (c1)
│   ├── adata.h5ad
│   └── cell_patch_256_aux/input/{cell_image_paths.json, cell_images/*_original.tif}
├── Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/    (c2，同上)
└── Xenium_Prime_Human_Skin_FFPE/                  (p1，同上)
```
`cell_image_paths.json` 的绝对路径**无需手改**——`run_all.sh` PHASE 0 会自动调 `fix_image_paths.py` 重映射。自检见 `DATA_SETUP.md` 第 3 步。

## 4. 提交
先改各 `slurm/capella_*.slurm` 的 `--account`，并确认 `slurm/_capella_common.sh` 顶部站点路径（`PROJECT_DIR`/`VENV_DIR`/`DATA_DIR`/`OUTPUT_DIR`/module）。
```bash
cd $CODE
mkdir -p logs outputs                              # SLURM 作业日志写到 logs/%x-%j.out；首次 sbatch 前必须建好
export UNI2H_MODEL_PATH=/path/to/UNI2-h            # 要 UNI2-h FID 就设，否则可留空

# 1) 自检 / 冒烟（写独立目录，绝不污染正式 results/）
sbatch slurm/capella_debug_1gpu.slurm                                                 # 环境/路径调试(1h → results_debug/)
sbatch slurm/capella_smoke_1gpu.slurm                                                 # 冒烟+resume+指标(4h → results_smoke/)
sbatch --export=ALL,VARIANT=gene2image,DS=c1,SEED=42 slurm/capella_train_1gpu.slurm   # c1 pilot(12h → 正式 results/，array 会 resume 它)

# 2)【默认】主 54-run = 1GPU job array（先建一次掩码，再投 array）
PREP=$(sbatch --parsable --export=ALL,MODE=prep --gres=gpu:1 --time=00:30:00 slurm/capella.slurm)
sbatch --array=0-53%8 --dependency=afterok:$PREP slurm/capella_array_1gpu.slurm        # 54 任务，最多 8 并发
sbatch --export=ALL,BATCH=4 slurm/capella_train_4gpu.slurm                             # 之后：cross + RQ4 + 汇总

# 补跑：先 §6 validate 查缺，再 sbatch --array=<idx列表> slurm/capella_array_1gpu.slurm
```
- **`%N` 限流**：`--array=0-53%8` 同时最多 8 个（=8 卡）；卡多就调大 `%16`/`%32`。总量 ≈ 2100 GPU-小时（8 卡≈11 天，16≈5.5，32≈2.7）。
- **单实验补跑**：`sbatch --export=ALL,VARIANT=..,DS=..,SEED=.. slurm/capella_train_1gpu.slurm`。
- 单实验 ~30-41h，2 天 walltime 整段跑完；评估不可 resume 但能整段跑完。

## 5. checkpoint / resume（把长训练拆成多个短 job）
- **`best_checkpoint.pt`**：只在 `val_mse`（纯速度 MSE）改善时更新，仅用于 eval / model selection。
- **`latest_checkpoint.pt`**：每 epoch 无条件、原子保存，供 `--auto_resume` 从上次完成的 epoch 续。
- 训练脚本带 `--auto_resume` + `#SBATCH --requeue`：抢占 → 回队 → 从 latest 续；短 walltime + 多次提交即可覆盖长训练。

## 6. 跑后门禁
```bash
python scripts/validate_runs.py --results_root results --expected_epochs 50 --patience 9999
```
核对 54 run 齐全、没被截断在低 epoch、按 `val_mse` 选点、eval 指标非 NaN 并重聚合主对比。**人工再看** `val_mse` 是否在 50 epoch 平台化，不够则统一上调 EPOCHS。

**正式结果可信前，还须过 3 个硬门槛（过不了别用该结果）：**
```bash
# ①&② DOPRI5：期望「无输出」= 无 dt-floor 强接受、无欠积分兜底
grep -h DOPRI5_DIAGNOSTICS logs/*.out | grep -E 'dt_floor=[^0]|under_integration_fallback=[^0]'
# ③ 无评价样本因 NaN 被过滤：期望「无输出」
grep -l 'Dropped .* non-finite' logs/*.out
```
`rejected>0` 属正常（自适应求解器健康表现）。另需确认每个 run 的 `evaluation_summary.json` 里 `n_ssim_used == n_psnr_used == total_samples`。

## 7. 输出物（`results/` 下）
| 文件 | 内容 |
|---|---|
| `summary_main.csv` | 每(变体×数据集) FID/SSIM/PSNR/UNI2h-FID，跨 seed 均值±std ← 主结果（在留出 test 集上算） |
| `ablation/summary.csv` / `cross_dataset/summary.csv` | 消融 / 跨数据集 degradation |
| `<variant>_<ds>_seed<seed>/` | `checkpoints/best_checkpoint.pt`、`evaluation_summary.json`、`training_losses.csv`、`gene_importance_scores.csv` |
| `interpret/<ds>/` | RQ4：`attention.csv`、`attention_by_celltype.csv`、`gsea_consistency.json`、`A_endogeneity.json`、`intervention.csv`、`C_intervention_sensitivity.json`（含特异性比 + 95% CI） |
| `qualitative_<ds>_seed<seed>/` | 定性对比图 `main_comparison.{png,pdf}`、`ablation_comparison.{png,pdf}`（§8 生成） |

### 7.1 导出清单（从服务器导出用；√ 必导 / ○ 出图选导 / ✗ 留服务器）

**√ 必导 —— 几 MB，写论文直接用**
- [ ] `results/summary_main.csv` —— 主表
- [ ] `results/ablation/summary.csv`、`results/cross_dataset/summary.csv`
- [ ] `results/<每个 run>/evaluation_summary.json` —— 精确数字 + `n_ssim_used`（门槛③）
- [ ] `results/interpret/<ds>/*.{json,csv}` —— RQ4 全部
- [ ] `results/qualitative_<ds>_seed<seed>/{main,ablation}_comparison.{png,pdf}` —— 两张对比图
- [ ] `results/EXPERIMENTS_CATALOG.md`
- [ ] **门槛证据**：`logs/*.out`（或 grep 后的结论，见 §6）+ `validate_runs.py` 输出

**○ 出图选导 —— 只导你要展示的**
- [ ] `results/<run>/training_losses.csv` / `training_curves.png` —— 收敛图
- [ ] `results/<run>/embeddings/uni2h_embeddings.npy` + `embeddings_metadata.csv` —— 仅做 UMAP 时

**✗ 留服务器 —— 大，勿导**
- [ ] `checkpoints/*.pt`、全部 `generated_images/`（体量巨大）、`{original,predicted}_rna_expressions.npy`

**一键打包 √ 档（在服务器上）：**
```bash
cd $CODE/results
tar -czf paper_bundle.tgz \
  summary_main.csv ablation/summary.csv cross_dataset/summary.csv EXPERIMENTS_CATALOG.md \
  */evaluation_summary.json interpret/*/*.json interpret/*/*.csv \
  qualitative_*/main_comparison.* qualitative_*/ablation_comparison.*
```
> 建议：定性图/UMAP 在服务器上生成、只导成品图（checkpoint 大，别拉回本地）。导结果前先过 §6 的 3 门槛。

## 8. 定性对比图（54-run 跑完后，可选）
出**两张对比图**：主实验（Real｜GeneFlow｜Gene2Image）+ 消融（Real｜Gene2Image｜randPath｜PathPrior｜noTrans｜noMask）。一条命令（各模型同 DS+同 seed → 同一批 test 细胞 + 同一配对噪声 → 逐格对齐，差异只来自编码器）：
```bash
cd $CODE
DS=c1 SEED=42 NCELLS=8 bash scripts/make_comparison_figures.sh
#  -> results/qualitative_c1_seed42/{main,ablation}_comparison.{png,pdf}   ← 只导这两张
```
单模型"真实 vs 生成"也可直接跑 `python rectified/rectified_generate.py --model_path <ckpt> --adata <..> --image_paths <..cell_image_paths_local.json> --num_samples 100 --gen_steps 100 --seed 42 --output_dir <out>`。
> `--enable_stain_normalization` 只用于出图美化，**评估时勿开**（会把真实图染色分布泄漏进生成图、虚高指标）。

## 9. 仍需人工确认（代码替不了）
各 `slurm/capella_*.slurm` 的 `--account`；`_capella_common.sh` 顶部 `PROJECT_DIR`/`VENV_DIR`/`DATA_DIR`/`OUTPUT_DIR`/module 名；`UNI2H_MODEL_PATH`；`WANDB_MODE`（默认 offline）；`EPOCHS=50` 是否够收敛（c1 pilot 确认）。
