# Gene2Image — 运行说明（单卡 H100 + PBS）

> **本文件只讲“怎么把代码跑起来”**。研究方法/结果见论文；更细的合规核对、产物→论文表映射、
> 更细的运行细节（产物→论文表、时间估计、排错）见 `docs/RUN_H100_PBS.md`。
> 脚本在 `code/pbs/`：`prereqs.pbs`(建掩码) · `one_exp.pbs`(单 run) · `smoke.pbs`(冒烟测) · `submit_all.sh`(一键提交)。

---

## 0. 一次性准备（登录节点）

```bash
cd <你的>/Gene2Image-main/code
export PROJECT_DIR=$PWD
export VENV_DIR=/data/ws/<你>/venv
export RELEASE_MODULE=release/24.04             # module spider release 查实际值
export PYTORCH_MODULE=PyTorch/2.1.2-CUDA-12.1.1  # module spider PyTorch，要 torch 2.x + cu12x
bash slurm/setup_env.sh                           # 幂等；建 venv、装 gseapy/torchmetrics 等，numpy 锁 1.26.4
```

- **数据**：Zenodo `records/17429142` 下载解压到 `code/data/processed_data/<folder>/`（三个 Xenium 样本；步骤见 `DATA_SETUP.md`）。
- **（可选）离线通路库**：无外网时 `export GMT_HALLMARK=/path/h.all.v2023.2.Hs.symbols.gmt`（有外网留空，自动 gseapy 联网）。
- **（可选）UNI2-h 权限**：`MahmoodLab/UNI2-h` 放对路径 → UNI2-h FID；缺则该指标 N/A，不影响 FID/SSIM/PSNR。

---

## 1. 站点变量（每次提交前 export 一组，保持一致）

```bash
export PROJECT_DIR=$PWD                          # 仓库 code/ 目录
export VENV_DIR=/data/ws/<你>/venv
export RELEASE_MODULE=release/24.04  PYTORCH_MODULE=PyTorch/2.1.2-CUDA-12.1.1
export OUTPUT_DIR=$PROJECT_DIR/results           # ★ 所有作业必须同一个
export DATA_DIR=$PROJECT_DIR/data/processed_data
export MASK_DIR=$PROJECT_DIR/data/pathway_masks
# 可选：export GMT_HALLMARK=... CELL_TYPE_KEY=cell_type BATCH_SIZE=32 QUEUE=<你的GPU队列>
```
> **队列/账户**：把 `code/pbs/*.pbs` 顶部 `#PBS -q gpu` / `#PBS -P/-A` 的 `CHANGEME` 改成你的，或 `export QUEUE=<你的GPU队列>`。

---

## 2. 冒烟测（**必做**，~1-2h，1×H100）

```bash
qsub -v PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,DATA_DIR,MASK_DIR,GMT_HALLMARK pbs/smoke.pbs
```
它建 c1/p1 掩码 → 测 nomask@p1 显存 → gene2image@c1 全链路(1 epoch+eval)。
看 `logs/g2i_smoke-*`：三步都过 = 链路健康。记下日志里的 **“gene2image@c1 1-epoch+eval 墙钟”**（用来估全量时长）。

---

## 3. 一键提交全部

```bash
DRY_RUN=1 bash pbs/submit_all.sh     # 先预览：打印提哪些/跳哪些，不真提
DRY_RUN=0 bash pbs/submit_all.sh     # 确认后真正提交
```
自动依赖链(**train/eval 已拆分**)：**prereqs** → 每 run 拆成【**train**(12h/36h)+ **eval**(4h,依赖 train)】→ **RQ4 interpret 3**。共 **130 作业**(54 train + 54 eval + 9 cross_train + 9 cross_eval + 3 interpret + 1 prereqs)。
**幂等**：run 完成(有 `evaluation_summary.json`)则跳过 → 被打断后重跑本命令只补没跑完的。

**手动提单个**（调试/补跑）：
```bash
qsub -N g2i_notrans_p1_s44 -l walltime=36:00:00 \
  -v PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,OUTPUT_DIR,DATA_DIR,MASK_DIR,KIND=exp,VARIANT=notrans,DS=p1,SEED=44 \
  pbs/one_exp.pbs
```

---

## 4. 监控 + 汇总

```bash
qstat -u $USER                                   # 看排队/运行
tail -f logs/g2i_gene2image_c1_s42-*             # 看日志；grep "final t=" 应见 1.0000
# 全部完成后跑一次：
python scripts/summarize_results.py --results_root $OUTPUT_DIR --out_dir $OUTPUT_DIR --expected_seeds 3
```

**产物 → 论文表**：`summary_main.csv`→主表 · `ablation/summary.csv`→消融 · `cross_dataset/summary.csv`→跨 panel · `interpret/<ds>/`→RQ4(A/B/C)。

---

## 5. 资源、时间、旋钮

- **每作业**：`select=1:ncpus=8:ngpus=1:mem=120gb` + `--use_amp`；walltime c1/c2=12h、p1/cross=36h、interpret=4h。
- **任务量**：**130 作业**(54 train + 54 eval + 9 cross_train + 9 cross_eval + 3 interpret + 1 prereqs)。train 作业占 12h/36h、eval 只 4h。**单卡串行 ≈ 2 周–2 个月**（早停 vs 跑满差别大）；能并发 N 张卡则 ÷N。**先冒烟测拿 1-epoch 墙钟才是准数。**
- **旋钮**：`BATCH_SIZE=32`(先冒烟测显存，~1.5-2× 提速) · `SEEDS="42 43"` · `INCLUDE_CROSS=0 INCLUDE_INTERPRET=0`(先只跑主/消融) · `INCLUDE_REACTOME=1`(P1 附加消融) · `AUTO_RESUME=1`(被杀作业续训)。

---

## 6. 排错

- **nomask@p1 OOM**：降 `BATCH_SIZE`（16→8），确保 `--use_amp`。
- **作业被 walltime 杀**：`export AUTO_RESUME=1` 后重跑 `submit_all.sh`（判据=`evaluation_summary.json`，半成品会被续训）。补提前先 `qstat` 确认无同名作业在跑。
- **RQ4-A 按细胞类型没出**：`export CELL_TYPE_KEY=<obs列名>`（`python -c "import anndata;print(anndata.read_h5ad('.../adata.h5ad',backed='r').obs.columns.tolist())"` 查列名）。
- **`module load` 找不到**：`module spider release` / `module spider PyTorch` 查版本。
- **`import torch` 报 `iJIT_NotifyEvent`**：venv 里 `pip install "mkl==2024.0.0"`。
- **PBS 是 Torque 非 PBS Pro**：`-l select=1:ncpus=8:ngpus=1:mem=120gb` 换成 `-l nodes=1:ppn=8:gpus=1 -l mem=120gb`。

---

## TL;DR
```bash
# 1) bash slurm/setup_env.sh   +   下数据到 data/processed_data/（见 DATA_SETUP.md）
# 2) export 站点变量（§1）
# 3) qsub ... pbs/smoke.pbs                 → 三步过 + 记 1-epoch 墙钟
# 4) DRY_RUN=1 bash pbs/submit_all.sh       → 预览
# 5) DRY_RUN=0 bash pbs/submit_all.sh       → 提全量（自动依赖链）
# 6) python scripts/summarize_results.py --results_root $OUTPUT_DIR --out_dir $OUTPUT_DIR --expected_seeds 3
```
