# Gene2Image — 在 H100 集群（PBS）上跑全部实验

> 面向 **PBS**（PBS Pro / OpenPBS / Torque）调度、**每作业 1×H100** 的运行手册。
> 脚本在 `code/pbs/`：`prereqs.pbs`(建掩码) · `one_exp.pbs`(单 run 训练+评估) · `smoke.pbs`(冒烟测) · `submit_all.sh`(一键提交+依赖链)。
> 与 `docs/PLAN_EXECUTION_GUIDE.md`(规划↔代码合规、产物→论文表)配套。环境/数据细节见根 `README.md`、`DATA_SETUP.md`、`slurm/setup_env.sh`。

---

## 0. 一次性前置（登录节点）

**a. 建环境**（module + venv，别用 conda；见 `slurm/setup_env.sh`）：
```bash
cd <你的>/Gene2Image/code
export PROJECT_DIR=$PWD
export VENV_DIR=/data/ws/<你>/venv
export RELEASE_MODULE=release/24.04            # module spider release 查实际值
export PYTORCH_MODULE=PyTorch/2.1.2-CUDA-12.1.1 # module spider PyTorch，要 torch 2.x + cu12x
bash slurm/setup_env.sh                          # 幂等；装 gseapy/torchmetrics 等，numpy 锁 1.26.4
```

**b. 数据**：Zenodo `records/17429142` 下载解压到 `code/data/processed_data/<folder>/`（三个 Xenium 样本，folder 名见 `pbs/one_exp.pbs` 的 `dataset_dir`）。步骤见 `DATA_SETUP.md`。

**c.（可选）离线通路库**：计算节点无外网时，`export GMT_HALLMARK=/path/h.all.v2023.2.Hs.symbols.gmt`（Reactome 另加 `GMT_REACTOME`）。有外网则留空，自动用 gseapy 联网。

**d.（可选）UNI2-h 权限**：申请 `MahmoodLab/UNI2-h` 放对路径 → UNI2-h FID；缺则该指标 N/A，不影响 FID/SSIM/PSNR。

---

## 1. 站点变量（每次提交前 export 一组，三处保持一致）

```bash
export PROJECT_DIR=$PWD                          # 仓库 code/ 目录
export VENV_DIR=/data/ws/<你>/venv
export RELEASE_MODULE=release/24.04  PYTORCH_MODULE=PyTorch/2.1.2-CUDA-12.1.1
export OUTPUT_DIR=$PROJECT_DIR/results           # ★ 所有作业必须同一个
export DATA_DIR=$PROJECT_DIR/data/processed_data
export MASK_DIR=$PROJECT_DIR/data/pathway_masks
# 可选：export GMT_HALLMARK=... CELL_TYPE_KEY=cell_type BATCH_SIZE=32 AUTO_RESUME=1
```
> **队列/账户**：`pbs/*.pbs` 顶部把 `#PBS -q gpu` / `#PBS -P/-A` 的 `CHANGEME` 改成你的；或提交时 `export QUEUE=<你的GPU队列>`（`submit_all.sh` 会带 `-q`）。

---

## 2. 冒烟测（**必做**，~1-2h，1×H100）

```bash
qsub -v PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,DATA_DIR,MASK_DIR,GMT_HALLMARK pbs/smoke.pbs
```
它会：① 建 c1/p1 掩码 ② **nomask@p1 显存**冒烟(防全量时 OOM) ③ **gene2image@c1 全链路**(1 epoch+eval)。
看 `logs/g2i_smoke-*`：三步都过 = 运行时链路健康。**把 “gene2image@c1 1-epoch+eval 墙钟 ≈ N s” 发给我，我给你精确总时长。**
若第②步 OOM → 降 `BATCH_SIZE`（见 §6 排错）。

---

## 3. 一键提交全部（依赖链 + 幂等）

```bash
# 先预览（默认 DRY_RUN=1，只打印提哪些/跳哪些，不真提）：
DRY_RUN=1 bash pbs/submit_all.sh
# 确认无误后真正提交：
DRY_RUN=0 bash pbs/submit_all.sh
```
自动编排(**train/eval 已拆分**)：**prereqs(建掩码)** → 每个 run 拆成【**train 作业**(12h/36h,`depend=prereqs`)+ **eval 作业**(4h,`depend=afterok:train`)】→ **RQ4 interpret 3**(依赖对应 `gene2image`[+`geneflow`] eval)。
作业总数 **130**：54 train + 54 eval + 9 cross_train + 9 cross_eval + 3 interpret + 1 prereqs。**好处**:eval 短(4h)进短队列;eval 失败/重跑不必重训;有 checkpoint 但缺 eval 时只补 eval。
**幂等**：run 目录已存在则跳过 → 作业被 walltime 打断后，重跑 `submit_all.sh` 只补没跑完的（配合 `AUTO_RESUME=1` 可续训）。

**手动提单个**（补跑/调试）：
```bash
qsub -N g2i_notrans_p1_s44 -l walltime=36:00:00 \
  -v PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,OUTPUT_DIR,DATA_DIR,MASK_DIR,KIND=exp,VARIANT=notrans,DS=p1,SEED=44 \
  pbs/one_exp.pbs
```

---

## 4. 监控 + 汇总

```bash
qstat -u $USER                     # 看排队/运行
tail -f logs/g2i_gene2image_c1_s42-*    # 看某作业日志；grep "final t=" 确认 DOPRI5 积分到 1.0000
# 全部完成后（手动跑一次）：
python scripts/summarize_results.py --results_root $OUTPUT_DIR --out_dir $OUTPUT_DIR --expected_seeds 3
```
`--expected_seeds 3`：任何 (变体,数据集) 组少于 3 个 seed 或指标缺失会**响亮报警**（防把残缺均值当完整）。

**产物 → 论文表**：`summary_main.csv`→主表(RQ1) · `ablation/summary.csv`→消融(RQ2/RQ3) · `cross_dataset/summary.csv`→跨 panel · `interpret/<ds>/`→RQ4(A 熵+按细胞类型 Jaccard / B GSEA一致性 / C 干预)。

---

## 5. H100 资源与时间

**每作业资源**（`pbs/*.pbs` 已设）：`select=1:ncpus=8:ngpus=1:mem=120gb`，`--use_amp`。H100 80G 对 256² UNet 有余量。

| KIND | walltime(安全上限) | 说明 |
|---|---|---|
| exp c1/c2 | 12:00:00 | 单 run 训练+评估 |
| exp p1 / cross | 36:00:00 | P1 最大(~5000 基因) |
| interpret | 04:00:00 | 读已训 ckpt |

**任务量**：**130 作业** = 54 train + 54 eval + 9 cross_train + 9 cross_eval + 3 interpret + 1 prereqs（train/eval 已拆分:train 作业占 12h/36h,eval 作业只 4h）。

**总时长估计（诚实,±2×）**：核心成本是训练 = `epoch数 × cells/batch × 单步`。**EPOCHS 默认已改为 50(对齐 GeneFlow 源码 train.sh)**,早停 patience=5 常先触发,故上限 50 主要压低了「跑满」的保守估计。累计约 **250–450 GPU-小时**(旧 100-epoch 估计的保守端约减半)。
- **每作业 1 卡、集群多卡并发**：墙钟 ≈ 总 GPU-时 ÷ 你能同时占的 H100 数 → 若能并发 8 张,~2-3 天。
- **全局只有 1 张 H100（严格串行）**：**~2–4 周**。
- 参照项目自己的预算(A100 上 c1/c2=12h、p1=36h)；H100 更快,早停(patience 5)通常远不到上限。
- **最靠谱**：冒烟测的 1-epoch 墙钟 × 实际 epoch 数 × 63 run = 真实值 —— 发我算。

**砍时间**：① `BATCH_SIZE=32`（先冒烟测显存，~1.5-2×，属改优化动力学的选择）② 先 `INCLUDE_CROSS=0 INCLUDE_INTERPRET=0` 拿主/消融核心 ③ P1 最贵可最后跑。

---

## 6. 排错

- **nomask@p1 OOM**：`nomask` 建稠密 A(P×G)，P1 最紧。降 `BATCH_SIZE`（16→8）、确保 `--use_amp`(默认带)；仍不行记录并在论文注明该臂受限。
- **作业被 walltime 杀**：`export AUTO_RESUME=1` 后重跑 `submit_all.sh`（判据=`evaluation_summary.json`,半成品会被重提续训）。⚠️ 续训从**最优 checkpoint** 恢复(非严格最新),会重做少数 epoch + AMP scaler 重热身(影响小);故建议给足 walltime 尽量一次跑完。**补提前先 `qstat -u $USER` 确认无同名作业在跑**,以免重复提交。
- **RQ4-A 按细胞类型没出**：自动探测没命中 obs 列名 → `export CELL_TYPE_KEY=<列名>`（先 `python -c "import anndata;print(anndata.read_h5ad('.../adata.h5ad',backed='r').obs.columns.tolist())"` 看列名）。
- **`module load` 报找不到**：`module spider release` / `module spider PyTorch` 查实际版本填 `RELEASE_MODULE`/`PYTORCH_MODULE`。
- **`import torch` 报 `iJIT_NotifyEvent`**：venv 里 `pip install "mkl==2024.0.0"`。
- **PBS 是 Torque(非 PBS Pro)**：把 `-l select=1:ncpus=8:ngpus=1:mem=120gb` 换成 `-l nodes=1:ppn=8:gpus=1 -l mem=120gb`，依赖用 `-W depend=afterok:<id>`（多数 Torque 兼容）。
- **`-v` 传参**：站点变量须在提交前 `export`（`-v VAR` 继承名字）；值带空格的（如 SEEDS）已用下划线编码、脚本内还原。

---

## 7. 最短路径 TL;DR
```bash
# 1) 环境 + 数据（一次性,见 §0）
# 2) export 站点变量（§1）
# 3) 冒烟测：qsub ... pbs/smoke.pbs   →  三步过 + 记 1-epoch 墙钟
# 4) DRY_RUN=1 bash pbs/submit_all.sh  →  预览
# 5) DRY_RUN=0 bash pbs/submit_all.sh  →  提全量（自动依赖链）
# 6) 全完成后: python scripts/summarize_results.py --results_root $OUTPUT_DIR --out_dir $OUTPUT_DIR --expected_seeds 3
```
