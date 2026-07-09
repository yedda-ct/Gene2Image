# Gene2Image — 代码修改 + 跑代码

Xenium 空间转录组 → H&E 图像生成。命令里 `$CODE` = 本目录下的 `code/`。

矩阵：6 变体（`gene2image` / `geneflow` / `randpath` / `pathprior` / `notrans` / `nomask`）× 3 数据集（`c1`/`c2`/`p1`）× 3 seed（42/43/44）= **54 主 run** + cross-dataset + RQ4 interpret。
数据集名（写死，勿改）：`c1`=`Xenium_V1_hSkin_Melanoma_Base_FFPE`，`c2`=`Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE`，`p1`=`Xenium_Prime_Human_Skin_FFPE`。

---

# 一、代码修改

## 代码修复
| 文件 | 改动 | 原因 |
|---|---|---|
| `rectified/rectified_flow.py:171,387` | `clamp(x,-1,1)` → `clamp(x,0.0,1.0)` | 图像全程 `[0,1]`（`ToTensor`，无 `[-1,1]` 归一化），eval `data_range=1.0`；采样须一致，否则放行 `[-1,0)` 非法暗像素、污染 SSIM/PSNR/FID |
| `rectified/rectified_train.py` | **`latest_checkpoint.pt` 每 epoch 无条件、原子保存**（写 `.tmp` 再 `os.replace`），与 `best` 解耦 | `--auto_resume` 从**上次完成的 epoch** 续（不是上次改善）；`best_checkpoint.pt` 仍**只在 val_mse 改善时保存、仅用于 eval / model selection**；`manage_checkpoints` 不删 latest |
| `rectified/rectified_evaluate.py` | Inception-v3(FID) 实例化 + 3 处使用点加 **try/except 守卫**；UNI2-h/HE2RNA/HEST 路径改 env 可配 | 离线计算节点缺 Inception 权重时**不再崩整个评估**（FID=NaN，SSIM/PSNR/UNI2-h 照常）；UNI2-h FID 死路径修复 |
| `rectified/rectified_main.py`、`rectified_generate.py` | HEST csv 路径改 `$HEST_METADATA_CSV` | 同类写死死路径 |
| `code/requirements.no_torch.txt` | 启用 `opencv-python-headless==4.10.0.84` | UNI2-h 的 `utils_uni2h.py` 顶层 `import cv2` |
| `rectified/rectified_train.py`（选点）、`src/pathway_encoder.py`、`scripts/build_pathway_mask.py`、`analysis/pathway_interpret.py`、`scripts/run_*.sh`、`summarize_results.py` | 最优检查点/早停判据改 **`val_mse`**（纯速度 MSE，不含 L1）；PathPrior 定种子初始化；randPath 每 seed 独立随机掩码；RQ4 逐细胞类型分析；数据划分泄漏修复（`sorted(set)`+`PYTHONHASHSEED=0`）；`EPOCHS` 默认 **50** | 跨变体公平选点、可复现、RQ2 误差棒含随机掩码方差 |

## 新增 Slurm（按 Capella 资源表，**短 walltime**；全部 `set -euo pipefail` + GPU fail-fast + 打印 host/CUDA/JOB_ID + 建 logs/outputs + `--auto_resume`）
| 脚本 | 用途 | GPU/CPU/MEM/TIME |
|---|---|---|
| `slurm/capella_debug_1gpu.slurm` | 环境/路径/import/数据/模型/单 batch | 1 / 14 / 120G / **1h** |
| `slurm/capella_smoke_1gpu.slurm` | 少量 epoch + resume + 指标 | 1 / 14 / 150G / **4h** |
| `slurm/capella_train_1gpu.slurm` | 单卡短训练（pilot / 单实验补跑，可恢复） | 1 / 14 / 180G / **12h** |
| `slurm/capella_array_1gpu.slurm` | **【默认】主实验 54-run job array**（一实验一任务，`%N` 限流；GPU 够时最快、最好排队） | 1 / 14 / 180G / 2d·任务 |
| `slurm/capella_train_4gpu.slurm` | **（可选）**整节点 4 并行单卡（`run_all.sh`，**非 DDP**）+ `BATCH`；省事自包含但并发封顶 4/节点、更难排队 | 4 / 56 / 700G / **12h** |
| `slurm/capella_train_4gpu_long.slurm` | 同上，**仅当某 BATCH 确需 >12h**（非默认；优先 12h + resume 拆短） | 4 / 56 / 700G / 2d |

`slurm/_capella_common.sh`：集中硬编码站点路径 + 安全检查 + GPU fail-fast + 掩码构建助手。
**默认 walltime 都尽量短**（不写满 7 天）以减少排队；长任务靠 checkpoint resume 拆成多个短 job。

## 删除
删除未验证 DDP 的 `capella_train_2gpu.slurm`（4GPU 改为**并行单卡**，非 DDP）；`slurm/capella.slurm`（分波 array）保留但**非推荐**。所有训练脚本均支持 `--auto_resume`。

---

# 二、跑代码

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
# (b) 其余依赖（含 opencv/gseapy/scanpy/timm 等）
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

## 4. 提交（默认：多个 1GPU job array，卡多时最快；4GPU 为可选自包含方式）
先改各 `slurm/capella_*.slurm` 的 `--account`，并确认 `slurm/_capella_common.sh` 顶部站点路径（`PROJECT_DIR`/`VENV_DIR`/`DATA_DIR`/`OUTPUT_DIR`/module）。
```bash
cd $CODE
export UNI2H_MODEL_PATH=/path/to/UNI2-h            # 要 UNI2-h FID 就设，否则可留空

sbatch slurm/capella_debug_1gpu.slurm                                                 # 1) 环境/路径调试(1h)
sbatch slurm/capella_smoke_1gpu.slurm                                                 # 2) 冒烟+resume+指标(4h)
sbatch --export=ALL,VARIANT=gene2image,DS=c1,SEED=42 slurm/capella_train_1gpu.slurm   # 3) c1 pilot(12h，量准单实验墙钟)

# 4)【默认】主实验（54 run）= 1GPU job array —— GPU 够时最快、最好排队
#    （1GPU 任务能 backfill 任意空卡，并发可超 4、不用等整节点）。先建一次掩码，再投 array（%N 限流）：
PREP=$(sbatch --parsable --export=ALL,MODE=prep --gres=gpu:1 --time=00:30:00 slurm/capella.slurm)
sbatch --array=0-53%8 --dependency=afterok:$PREP slurm/capella_array_1gpu.slurm        # 54 任务，最多 8 并发
#    %8 → %16/%32 提并发更快（看账号配额）。单实验 ~30-41h，每任务 2 天 walltime 整段跑完。
sbatch --export=ALL,BATCH=4 slurm/capella_train_4gpu.slurm                             # 之后：cross + RQ4 + 汇总
# 补跑：先 validate 查缺，再 sbatch --array=<idx列表> slurm/capella_array_1gpu.slurm

# —— 可选（省事、自包含，但并发封顶 4/节点、更难排队）：4GPU 整节点分块 ——
# sbatch --export=ALL,BATCH=1 slurm/capella_train_4gpu_long.slurm   # c1（2天+重投至完成），c2/p1 同理
```
- **为什么默认 1GPU array**：GPU 够时，多个 1GPU 在"并发上限"和"排队难度"两头都赢 4GPU（4GPU 要整节点 4 空卡才起、封顶 4 并发）。总量 ≈ 2100 GPU-小时：8 卡 ≈ 11 天、16 卡 ≈ 5.5 天、32 卡 ≈ 2.7 天。
- **`%N` 限流**：`--array=0-53%8` 同时最多 8 个（=8 卡），防一次投爆队列；想快就调大。
- **单实验补跑**：`sbatch --export=ALL,VARIANT=..,DS=..,SEED=.. slurm/capella_train_1gpu.slurm`。
- **评估不可 resume**：单实验评估 ≈ 12h，2 天 walltime 下单实验（≈41h）能整段跑完、无风险。
- checkpoint / resume 见 §5。

## 5. checkpoint / resume（把长训练拆成多个短 job）
- **`best_checkpoint.pt`**：只在 `val_mse` 改善时更新，仅用于 eval / model selection。
- **`latest_checkpoint.pt`**：每 epoch 无条件、原子保存，供 `--auto_resume` 从上次完成的 epoch 续。
- 训练脚本带 `--auto_resume` + `#SBATCH --requeue`：抢占 → 回队 → 从 latest 续；短 walltime + 多次提交即可覆盖长训练，避免一次失败丢全部进度、也减少排队。
- `run_all.sh` 不幂等：`capella_train_4gpu.slurm` 整作业回队会重跑该 BATCH 里已训完的 run（但 `--auto_resume` 让它们跳过训练、只重评估）；优先按 BATCH 分块。

## 6. 跑后门禁
```bash
python scripts/validate_runs.py --results_root results --expected_epochs 50 --patience 9999
```
核对 54 run 齐全、没被截断在低 epoch、按 `val_mse` 选点、eval 指标非 NaN 并重聚合主对比。**人工再看** `val_mse` 是否在 50 epoch 平台化，不够则对全部 54 run 统一上调 EPOCHS。

## 7. 输出物（`results/` 下）
| 文件 | 内容 |
|---|---|
| `summary_main.csv` | 每(变体×数据集) FID/SSIM/PSNR/UNI2h-FID，跨 seed 均值±std ← 主结果 |
| `ablation/summary.csv` / `cross_dataset/summary.csv` | 消融 / 跨数据集 degradation |
| `<variant>_<ds>_seed<seed>/` | `checkpoints/best_checkpoint.pt`、`evaluation_summary.json`、`training_losses.csv`、`gene_importance_scores.csv` |
| `interpret/<ds>/` | RQ4：`attention.csv`、`attention_by_celltype.csv`、`gsea_consistency.json`、`A_endogeneity.json`、`intervention.csv`、`C_causal.json` |

## 8. 仍需人工确认（代码替不了）
各 `slurm/capella_*.slurm` 的 `--account`；`_capella_common.sh` 顶部 `PROJECT_DIR`/`VENV_DIR`/`DATA_DIR`/`OUTPUT_DIR`/module 名；`UNI2H_MODEL_PATH`；`WANDB_MODE`（默认 offline）；`EPOCHS=50` 是否够收敛（c1 pilot 确认）。
