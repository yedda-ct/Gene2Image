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
| `slurm/capella_cross_1gpu.slurm` | 跨 panel 迁移 array（9 任务，各训 50ep；**非后处理**） | 1 / 14 / 180G / 24h·任务 |
| `slurm/capella_walltime_test.slurm` | 验证 walltime 自请求重排 + 续跑（8 分钟） | 1 / 14 / 120G / 8min |
| `slurm/capella_train_4gpu.slurm` | （可选）整节点 4 并行单卡 + `BATCH`（`BATCH=4` = RQ4+汇总） | 4 / 56 / 700G / 12h |

---

## 0. 部署 / 更新代码（换新版代码时用）

包里**不含数据**（你原来的 `code/data/` 要保留），推荐「解压到旁边、整目录替换、把数据接回来」，
不要用 `tar --strip-components=1 -C .` 直接覆盖——`tar` 解压是叠加式的，旧版删掉的文件会残留，
新旧脚本并存容易跑错。

```bash
cd /data/horse/ws/<你的工作区>/Gene2Image

# 1) 解压到临时目录
mkdir -p /tmp/g2i_new && tar -xzf Gene2Image_给师兄.tar.gz -C /tmp/g2i_new
#   得到 /tmp/g2i_new/Gene2Image/{code, README.md, gmt, ...}

# 2) 备份旧 code（可回退）
mv code code_OLD_$(date +%F)

# 3) 把旧结果移开留存（在 code_OLD 里；不清理旧 checkpoint，正式跑会触发协议守卫 exit 106，见 §4）
mv code_OLD_$(date +%F)/results results_INVALID_$(date +%F) 2>/dev/null || true

# 4) 放入新 code + gmt，并把数据接回来
cp -r /tmp/g2i_new/Gene2Image/code .
cp -r /tmp/g2i_new/Gene2Image/gmt  .
cp -r code_OLD_$(date +%F)/data code/data          # 数据在别处就改成你的真实路径
```

> 新 `code/` 是全新解压、本就没有 `results/`——正式跑第一步（建掩码）会创建干净的空 `results/`，
> 协议守卫不会误触发。第 3 步只是把**旧**结果挪走留存。

**首次部署（没有旧 code）**：跳过第 2、3 步，直接解压 + 放数据即可。

---

## 1. 必须先外部准备（代码不自动拉取）
| # | 资源 | 从哪来 | 放哪 / 怎么用 |
|---|---|---|---|
| 1 | **3 个 Xenium 样本数据** | Zenodo **17429142**（现成 `wget` 见 `code/GeneFlowREADME.md`） | 解压到 `$CODE/data/processed_data/<三个固定目录名>/`（见 §3） |
| 2 | **Hallmark GMT** | 仓库**已打包** `gmt/msigdb_2023.2_Hs/h.all.v2023.2.Hs.symbols.gmt` | 无需另下；解压后在 `Gene2Image/gmt/`，`_capella_common.sh` 的 `GMT_HALLMARK` 默认指向它 |
| 3 | **Inception-v3 权重**（FID 用） | PyTorch hub（~100MB，见 §2c） | 登录节点预取到 `~/.cache/torch`；离线否则 FID=NaN（已守卫**不崩**） |
| 4 | **UNI2-h 权重**（生物学 FID） | 门控 `MahmoodLab/UNI2-h`（HuggingFace） | 含 `pytorch_model.bin`，放默认位置 `Gene2Image/../models/UNI2-h`。**权重不在会 `exit 104` 拒绝启动**（防跑完才发现生物 FID 全 NaN）；确实不带它跑才 `export ALLOW_NO_UNI2H=1` |
| 5 | **Capella 账号** | 你的 ZIH 项目号 | 填**每个** `slurm/*.slurm` 的 `--account`（`grep -l CHANGEME slurm/*.slurm` 列全部） |

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

## 3. 放数据

数据（3 个 Xenium 样本）从 **Zenodo 17429142**（https://zenodo.org/records/17429142 ，开放下载）下。
解压后目录名**必须逐字如下**（训练脚本里写死，写错会 `Unknown dataset`）：

```
code/data/processed_data/
├── Xenium_V1_hSkin_Melanoma_Base_FFPE/            (c1, ~282 基因, ~107k 细胞)
│   ├── adata.h5ad                                  ← 基因表达（模型输入）
│   └── cell_patch_256_aux/input/{cell_image_paths.json, cell_images/*_original.tif}
├── Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/    (c2, ~382 基因，同上)
└── Xenium_Prime_Human_Skin_FFPE/                  (p1, ~5006 基因，同上)
```

每个样本目录下要能找到：① `adata.h5ad`，② `cell_patch_256_aux/input/cell_image_paths.json`，
③ 该 json 指向的 `.tif`（在 `.../input/cell_images/` 下，形如 `*_original.tif`，4 通道 RGB+DAPI）。

`cell_image_paths.json` 里是原作者的集群绝对路径，**无需手改**——`run_all.sh`/建掩码那步会自动调
`scripts/fix_image_paths.py` 按 `processed_data/` 之后的部分重映射到你的本地目录。

**数据自检（强烈建议，1 分钟）**：确认三个样本齐、图像能对上：
```bash
cd $CODE
python - <<'PY'
import os, json, anndata as ad
ROOT="data/processed_data"
DS={"c1":"Xenium_V1_hSkin_Melanoma_Base_FFPE","c2":"Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE","p1":"Xenium_Prime_Human_Skin_FFPE"}
ok=True
for tag,d in DS.items():
    base=os.path.join(ROOT,d); h5=os.path.join(base,"adata.h5ad")
    js=os.path.join(base,"cell_patch_256_aux/input/cell_image_paths.json")
    if not os.path.exists(h5) or not os.path.exists(js):
        print(f"[{tag}] ✗ 缺 adata.h5ad 或 cell_image_paths.json"); ok=False; continue
    a=ad.read_h5ad(h5,backed="r"); N,G=a.n_obs,a.n_vars; a.file.close()
    paths=json.load(open(js))
    def local(p):
        m="processed_data/"; i=p.rfind(m); return os.path.join(ROOT,p[i+len(m):]) if i>=0 else p
    hit=sum(os.path.exists(local(p)) for p in list(paths.values())[:50])
    print(f"[{tag}] ✓ {N} 细胞 × {G} 基因 | 图像 {len(paths)} 条 | 抽查 50 命中 {hit}/50")
    if hit==0: ok=False
print("\n结果:", "全部通过 ✅" if ok else "有问题 ❌")
PY
```
期望三行 `✓`，基因数约 c1≈282 / c2≈382 / p1≈5006，抽查命中 50/50。命中 0 = `.tif` 没解压到位。

## 4. 提交

**提交前三件事**（缺一样都会在第一次 sbatch 时踩坑）：
1. **填 `--account`**：`slurm/` 下**每个** `.slurm` 文件的 `--account=CHANGEME_capella_account` 都要改成你的项目号（`grep -l CHANGEME slurm/*.slurm` 列出全部）。或提交时统一加 `--account=<项目号>` 覆盖。
2. **确认站点路径**：`slurm/_capella_common.sh` 顶部的 `PROJECT_DIR`/`VENV_DIR`/`DATA_DIR`/module。
3. **UNI2-h 权重**：路径已写死默认 `$PROJECT_DIR/../../models/UNI2-h`，**不用手动 export**。但权重不在会直接 `exit 104` 拒绝启动（防止跑完两天才发现生物 FID 全是 NaN——这是上一批作废的原因）。确实要不带生物 FID 跑，才 `export ALLOW_NO_UNI2H=1`。

> ⚠️ **协议守卫（`exit 106`）**：若 `results/<run>/checkpoints/latest_checkpoint.pt` 是**旧代码/旧协议**写的，`--auto_resume` 会拒绝续跑并 `exit 106`。run 目录名与历史批次相同，不清理会把作废批次“洗”成看着跑满 50 的。**换代码后首次跑正式实验前**，先把旧结果移开：`mv results results_INVALID_$(date +%F)`（作业会重建空 `results/`，守卫看的是里面有没有旧 checkpoint，不是目录本身）。

```bash
cd $CODE
mkdir -p logs outputs                              # SLURM 作业日志写到 logs/%x-%j.out；首次 sbatch 前必须建好
# UNI2-h 路径已写死默认值，通常无需下面这行；仅当权重放在非默认位置时才 export
# export UNI2H_MODEL_PATH=/path/to/UNI2-h

# 1) 自检 / 冒烟（写独立目录，绝不污染正式 results/）
sbatch slurm/capella_debug_1gpu.slurm                                                 # 环境/路径调试(1h → results_debug/)
sbatch slurm/capella_smoke_1gpu.slurm                                                 # 冒烟+resume+指标(4h → results_smoke/)
sbatch --export=ALL,VARIANT=gene2image,DS=c1,SEED=42 slurm/capella_train_1gpu.slurm   # c1 pilot(12h → 正式 results/，array 会 resume 它)

# 2)【默认】主 54-run = 1GPU job array（先建一次掩码，再投 array）
PREP=$(sbatch --parsable --export=ALL,MODE=prep --gres=gpu:1 --time=00:30:00 slurm/capella.slurm)
sbatch --array=0-53%8 --dependency=afterok:$PREP slurm/capella_array_1gpu.slurm        # 54 任务，最多 8 并发

# 3) 跨 panel(独立 array —— 它的 9 个设置各要训练 50 epoch，约 300 GPU-h，装不进 12h 的收尾作业)
sbatch --array=0-8 slurm/capella_cross_1gpu.slurm                                      # 需主 54-run 先完成
# 4) 收尾：RQ4 interpret + 汇总（这两步才是真正的后处理，很快）
sbatch --export=ALL,MODE=post --gres=gpu:1 --time=04:00:00 slurm/capella.slurm

# 补跑：先 §6 validate 查缺，再 sbatch --array=<idx列表> slurm/capella_array_1gpu.slurm
```
- **并发 = 速度**：54 个 run 相互独立、一卡一个，所以墙钟 = **最慢的那个 run**，不是总量。按 H100 实测（2.60 it/s @ batch 16）：**c2 ~24h、c1 ~33h、p1 ~44h**（含约 2.6h 评估）。总量 ≈ 1800 GPU-小时。

  | 同时可用卡数 | 提交 | 大致墙钟 |
  |---|---|---|
  | 54 | `--array=0-53` | **~44h（下限）** |
  | 27 | `--array=0-53%27` | ~4 天 |
  | 8 | `--array=0-53%8` | ~11 天 |

  卡再多也压不到 44h 以下——那是单个 p1 run 的长度；再快只能降 `EPOCHS`（预算仍相同、claim 不受损）。

- **walltime 是排队旋钮，不是赌注**：脚本用 `--signal=B:USR1@900` 在被杀前 15 分钟**自己 `scontrol requeue`**，`--auto_resume` 接着跑。（`#SBATCH --requeue` **不覆盖 TIMEOUT**，只覆盖抢占/节点故障——所以这个 trap 是必需的。）
  ```bash
  sbatch --time=24:00:00 --array=0-53 slurm/capella_array_1gpu.slurm   # 默认，好回填
  sbatch --time=12:00:00 --array=0-53 slurm/capella_array_1gpu.slurm   # 队列忙时回填更好，多几个续跑周期
  sbatch --time=3-00:00:00 --array=0-53 slurm/capella_array_1gpu.slurm # 队列空时一把跑完（~44h）
  ```
  **下限约 4h**：评估（~2.6h）不可 resume，窗口太小会每个周期重跑评估。
- **单实验补跑**：`sbatch --export=ALL,VARIANT=..,DS=..,SEED=.. slurm/capella_train_1gpu.slurm`。

## 5. checkpoint / resume（把长训练拆成多个短 job）
- **`best_checkpoint.pt`**：只在 `val_mse`（纯速度 MSE）改善时更新，仅用于 eval / model selection。
- **`latest_checkpoint.pt`**：每 epoch 无条件、原子保存，供 `--auto_resume` 从上次完成的 epoch 续。
- 训练脚本带 `--auto_resume` + `#SBATCH --requeue`：抢占 → 回队 → 从 latest 续；短 walltime + 多次提交即可覆盖长训练。

## 6. 跑后门禁
```bash
python scripts/validate_runs.py --results_root results --expected_epochs 50 --patience 9999
```
`--patience` 必须与训练用的值一致（现为 9999 = 早停关闭）：门禁用它判断早停轮次是否合法，传错会把正常 run 判成 FAIL。

核对 54 run 齐全、没被异常截断、按 `val_mse` 选点、eval 指标非 NaN（含 UNI2-h 生物 FID）并重聚合主对比。早停关闭时每个 run 都应 `reached full budget 50/50`；`summary_main.csv` 的 `stop_epoch_mean` 应恒为 50 —— 它是「各分支训练预算确实相同」的证据，若有分支不是 50，说明它被截断而非早停。

**3 个硬门槛现在由 `validate_runs.py` 自动校验**（上面那条命令已包含），无需手工 grep：
- ①② DOPRI5 `dt_floor` / `under_integration_fallback` —— 从每-run 日志 `results/logs/exp_<v>_<ds>_s<seed>.log` 读（array 会 `tee -a` 到那里）
- ③ `n_ssim_used == n_psnr_used == total_samples` —— 从 `evaluation_summary.json` 读

`rejected>0` 属正常（自适应求解器的健康表现），门禁不视为失败。

要手工复核的话，注意 **python logging 走 stderr**，所以要 grep `.err` 而不是 `.out`：
```bash
grep -h DOPRI5_DIAGNOSTICS logs/*.err | grep -E 'dt_floor=[^0]|under_integration_fallback=[^0]'   # 期望无输出
grep -l 'Dropped .*non-finite' logs/*.err                                                          # 期望无输出
```

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
- [ ] **门槛证据**：`results/logs/exp_*.log`（每-run 日志，含 DOPRI5 诊断）+ `validate_runs.py` 输出

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
  qualitative_*/main_comparison.* qualitative_*/ablation_comparison.* \
  logs/exp_*.log efficiency_table.csv
```
**先生成效率表再打包**（54 份日志手工 grep 不现实）：
```bash
python3 scripts/collect_efficiency.py --results_root results   # -> results/efficiency_table.csv
```

> `logs/exp_*.log` **必须带上**：效率表的全部数据（`MODEL_STATS` / `EFFICIENCY_STATS`）和三条门槛的证据（`DOPRI5_DIAGNOSTICS` / `EVAL_GATES` / `TRAIN_GATES`）**只存在于每-run 日志里**，跑完就没有第二个来源（显存峰值、单 epoch 耗时无法事后重建）。`efficiency_table.csv` 由下面的聚合器生成。
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
各 `slurm/*.slurm` 的 `--account`；`_capella_common.sh` 顶部 `PROJECT_DIR`/`VENV_DIR`/`DATA_DIR`/`OUTPUT_DIR`/module 名；`WANDB_MODE`（默认 offline）；`EPOCHS=50` 是否够收敛（c1 pilot 确认）。

## 10. 常见问题速查
| 现象 | 原因 / 处理 |
|---|---|
| 作业 `exit 104` | UNI2-h 权重不在默认路径。放到 `Gene2Image/../models/UNI2-h/pytorch_model.bin`，或 `export ALLOW_NO_UNI2H=1` 不带生物 FID 跑（§1 第 4 项） |
| 作业 `exit 106` | `results/` 里有旧协议的 checkpoint。换代码后先 `mv results results_INVALID_$(date +%F)`（§0 / §4） |
| PHASE 0.5 报 "required pathway mask missing" | 没备好 Hallmark GMT。仓库已打包在 `Gene2Image/gmt/`，`_capella_common.sh` 默认指向；若被移动，`export GMT_HALLMARK=/绝对路径/h.all.v2023.2.Hs.symbols.gmt` |
| 训练日志 "0 cells with both expression and images" | 图像 `.tif` 没解压到位 / json 路径对不上。跑 §3 自检看抽查命中数 |
| `Unknown dataset` | `processed_data/` 下样本目录名写错，必须与 §3 三个名字逐字一致 |
| `import torch` 报 `iJIT_NotifyEvent` | MKL 符号冲突，`pip install mkl==2024.0`（环境问题，与数据无关） |
| 正常 run 被门禁判 FAIL | `validate_runs.py` 的 `--patience` 没跟训练一致（都应 9999，§6） |
