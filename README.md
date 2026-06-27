# Gene2Image：可学习结构化通路瓶颈的基因到病理图像生成

从单细胞基因表达生成 H&E 病理图像。本项目把 [GeneFlow](https://github.com/wangmengbo/GeneFlow)（NeurIPS 2025）的 RNA 编码器替换为一个**端到端可学习的结构化通路瓶颈编码器**——用固定的「通路-基因」二值掩码约束稀疏连接、为每个 (通路, 基因) 对赋予可学习的权重向量，介于 GeneFlow 的「无结构编码」与 MUPAD 的「固定 ssGSEA 打分」两个极端之间。**整流流 + UNet 生成主干完全复用 GeneFlow、逐字节不改**，使任何性能差异都能干净归因到编码器。

> 🖥️ **要在 Alpha Centauri（TU Dresden ZIH，8×A100 Slurm 集群）上代跑？**
> 本文 **第一部分（§A）就是完整操作手册**：环境搭建 → 30 分钟冒烟 → 分 8 批的完整 8 条 `sbatch` 命令（含 batch 8 的 `afterok` 依赖）→ 取回结果与排错速查。
> 跑实验只看 §A 即可；**第二部分（§B）是项目/方法总览**，代跑不需要读。

本文是仓库里**唯一**的总说明文档。更深的研究/实现细节在 `docs/`（`idea_report.md` / `implementation.md` / `dev_log.md`）与 `code/GeneFlowREADME.md`（上游基线）。

---
---

# §A 在 Alpha Centauri 上跑 Gene2Image（给代跑同学的操作手册）

> 一句话：**改一处变量 → 装环境 → 冒烟 30 分钟 → 分 8 批 sbatch → 取回 `results/`**。
> 全部脚本在 `code/slurm/`。本节是唯一需要照着敲的清单。全程不需要任何密码 / token，脚本里也没有。

集群：TU Dresden ZIH **Alpha Centauri**（8×A100 / 节点，Slurm，`module` + `venv`，**不用 conda**）。
作者本人无法登录，麻烦你代跑——有任何报错把 `logs/` 下对应 `.log`/`.err` 发回即可。

```
登录节点(有网) ──► 改 ★ 变量 ──► 建环境 ──► 冒烟测试 ──► 8 批正式实验 ──► 回传日志/结果
```

## A0. 一次性：把代码和数据放到 workspace

```bash
# 在 workspace（不要放 $HOME，配额小）。示例路径，按你的实际 workspace 改：
cd /data/horse/ws/<你的ws>-gene2image
# 解压交付包后，仓库 code/ 目录应在：
#   /data/horse/ws/<你的ws>-gene2image/Gene2Image/code
ls Gene2Image/code/scripts/run_all.sh        # 确认存在
ls Gene2Image/code/slurm/                     # 4 个脚本都在
```

三个数据集（c1/c2/p1）需放在 `code/data/processed_data/` 下，每个目录含 `adata.h5ad` 和
`cell_patch_256_aux/input/`。**数据已由作者处理好并归档在 Zenodo，你自己联网下载即可**
（约几十 GB，无需向作者索取）：**Zenodo 17429142** → https://zenodo.org/records/17429142 （开放下载）。
下载解压后最终长这样：

```
code/data/processed_data/
  Xenium_V1_hSkin_Melanoma_Base_FFPE/          (c1)
  Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/  (c2)
  Xenium_Prime_Human_Skin_FFPE/                (p1)
每个目录里都有 adata.h5ad 和 cell_patch_256_aux/
```

> 📄 **数据怎么下、目录怎么摆、怎么自检齐不齐**，详见交付包根目录 **`DATA_SETUP.md`**。
> 第一次放数据务必照它跑一遍自检（1 分钟），避免上来就训练几小时后才发现缺文件。
> json 里的失效集群路径**不用手动改**——正式作业会自动重映射到本地。

> 若脚本报 `\r` / `bad interpreter`（Windows 换行问题），执行一次：
> `sed -i 's/\r$//' slurm/*.sh slurm/*.slurm`

---

## A1. 改一处：所有脚本顶部的 ★ 变量（**最关键，错了全盘错**）

`setup_env.sh`、`smoke_test.slurm`、`alpha_batch.slurm`、`alpha_8gpu_train.slurm`
四个文件顶部都有同一组 `★ TODO` 变量。**最省事的办法是不改文件，改用环境变量传入**（见下方每条命令的 `--export`），或者把四个文件顶部改成一致的真实值。变量清单：

| 变量 | 含义 | 怎么填 |
|---|---|---|
| `PROJECT_DIR` | 仓库 `code/` 绝对路径 | 如 `/data/horse/ws/abc-gene2image/Gene2Image/code` |
| `VENV_DIR` | venv 路径（脚本会建） | 如 `/data/horse/ws/abc-gene2image/venv`，建议放 workspace |
| `RELEASE_MODULE` | release 模块版本 | 登录节点跑 `module spider release` 看可用版本，如 `release/24.04` |
| `PYTORCH_MODULE` | PyTorch 模块版本 | `module spider PyTorch`，选 **2.x + CUDA 12.x**，如 `PyTorch/2.1.2-CUDA-12.1.1` |
| `GMT_HALLMARK` | 离线通路库 `.gmt` | **计算节点若无外网必须填**（见 §A2.1）；有外网留空 |
| `GMT_REACTOME` | Reactome `.gmt` | 仅当跑 Reactome 扩展消融时才需要；默认留空 |
| `DATA_DIR` | 数据父目录 | 默认 `$PROJECT_DIR/data/processed_data`，一般不用改 |
| `OUTPUT_DIR` | 结果输出根 | 默认 `$PROJECT_DIR/results`。**8 个批次必须用同一个值！** |
| `INTERPRET_SEED` | RQ4 用哪个种子的 ckpt | 默认 `42`，不用改 |

> 账号已写死在脚本里：`--partition=alpha --account=swtest`。若你的账号/分区不同，改这两行。

---

## A2. 装环境（在**登录节点** login.alpha，有外网）

```bash
cd /data/horse/ws/<你的ws>-gene2image/Gene2Image/code
mkdir -p logs                                  # ⚠️ 必须先建！否则 sbatch 写不了日志会直接失败
# 若用环境变量传值（推荐），先 export，再跑 setup：
export PROJECT_DIR=$PWD
export VENV_DIR=/data/horse/ws/<你的ws>-gene2image/venv
export RELEASE_MODULE=release/24.04            # ← 换成 module spider 查到的真实值
export PYTORCH_MODULE=PyTorch/2.1.2-CUDA-12.1.1 # ← 同上
bash slurm/setup_env.sh
```

它会：`module load` 你填的模块 → 建 venv → 装依赖 → 打印 `torch / cuda avail / numpy` 自检。
脚本幂等（可重复跑）。**登录节点 `cuda avail` 可能是 False，属正常**——登录节点没 GPU，真假要在 GPU 作业里看。

### A2.1 离线通路库（计算节点没有外网时，**必须**先准备）

掩码构建默认走 `gseapy` 联网拉 Enrichr。Alpha 计算节点通常**无外网**，否则
`gene2image` 全臂会崩。解决：在**登录节点**（有外网）下载一次 Hallmark `.gmt`
（文件名类似 `h.all.v2023.2.Hs.symbols.gmt`，来源 MSigDB / GSEA，也可让作者直接给你），
然后把路径填进 `GMT_HALLMARK`：

```bash
# 放到 workspace，例如：
#   /data/horse/ws/<你的ws>-gene2image/gmt/h.all.v2023.2.Hs.symbols.gmt
export GMT_HALLMARK=/data/horse/ws/<你的ws>-gene2image/gmt/h.all.v2023.2.Hs.symbols.gmt
```

> 若 `GMT_HALLMARK` 没填且节点无外网，`run_all.sh` 会在训练前 **fail-fast 报错退出**
> （这是有意的，避免半个对比静默跑出来）。看到 "required pathway mask file(s) are missing"
> 就是这个原因 → 回来填 `GMT_HALLMARK`。

---

## A3. 冒烟测试（30 分钟，**务必先过再交大作业**）

```bash
cd /data/horse/ws/<你的ws>-gene2image/Gene2Image/code
mkdir -p logs
sbatch --exclude=i8009 --export=ALL,PROJECT_DIR=$PWD,VENV_DIR=$VENV_DIR,\
RELEASE_MODULE=$RELEASE_MODULE,PYTORCH_MODULE=$PYTORCH_MODULE,GMT_HALLMARK=$GMT_HALLMARK \
  slurm/smoke_test.slurm
squeue -u $USER                  # 看状态：PD 排队 / R 运行
tail -f logs/g2i_smoke-*.out     # 实时看输出，Ctrl-C 退出查看（不影响作业）
```

成功标志：`results_smoke/` 下出现 `evaluation_summary.json` 与 `summary_main.csv`
（落在 `results_smoke/`，不污染正式 `results/`）。冒烟会先秒级验证编排，再用很小的样本
真跑一遍 train+eval。**冒烟过了再往下做正式实验。**

---

## A4. 正式实验：分 8 批提交（每批 8 卡，≤72h）

整套实验拆成 8 批，**每批在 8 卡上最多跑 2 轮**，所以即使单个实验最坏跑满
100 epoch（约 36h 上界），每批也不会超 72h。逐批提交即可，互不阻塞——
**只有批 8 依赖批 1 和批 4**（见下表"依赖"）。c1/c2（快）合并、p1（慢）每批单独占 1 轮。

> 下面每条命令都带齐 ★ 变量。先把这几个 export 在 shell 里设好（与 §A2 一致），命令会更短：
> ```bash
> export PROJECT_DIR=$PWD VENV_DIR=... RELEASE_MODULE=... PYTORCH_MODULE=... GMT_HALLMARK=...
> export OUTPUT_DIR=$PROJECT_DIR/results     # ★ 8 批必须一致，别改
> COMMON="ALL,PROJECT_DIR,VENV_DIR,RELEASE_MODULE,PYTORCH_MODULE,GMT_HALLMARK,OUTPUT_DIR"
> ```

### 批次表

| 批 | 内容 | job 数 | 轮数 | 依赖 |
|---|---|---|---|---|
| 1 | c1+c2 主对照 `gene2image`+`geneflow` ×3 种子 | 12 | 2 | — |
| 2 | c1+c2 消融 `randpath`+`pathprior` ×3 | 12 | 2 | — |
| 3 | c1+c2 消融 `notrans`+`nomask` ×3 | 12 | 2 | — |
| 4 | p1 主对照 `gene2image`+`geneflow` ×3 | 6 | 1 | — |
| 5 | p1 消融 `randpath`+`pathprior` ×3 | 6 | 1 | — |
| 6 | p1 消融 `notrans`+`nomask` ×3 | 6 | 1 | — |
| 7 | 跨数据集 c1↔c2 / c1→p1 ×3 | 9 | 2 | — |
| 8 | RQ4 可解释性 + 最终汇总 CSV | 3 | 1 | **批 1 + 批 4** |

### 8 条提交命令

```bash
cd /data/horse/ws/<你的ws>-gene2image/Gene2Image/code

# 批 1～7 互不依赖，可一次性全提（Slurm 自己排队）：
sbatch -J g2i_b1 --export=$COMMON,BATCH=1 slurm/capella.slurm
sbatch -J g2i_b2 --export=$COMMON,BATCH=2 slurm/capella.slurm
sbatch -J g2i_b3 --export=$COMMON,BATCH=3 slurm/capella.slurm
sbatch -J g2i_b4 --export=$COMMON,BATCH=4 slurm/capella.slurm
sbatch -J g2i_b5 --export=$COMMON,BATCH=5 slurm/capella.slurm
sbatch -J g2i_b6 --export=$COMMON,BATCH=6 slurm/capella.slurm
sbatch -J g2i_b7 --export=$COMMON,BATCH=7 slurm/capella.slurm

# 批 8 必须等批 1 和批 4 都跑完再提（它要读那两批的 checkpoint）。
# 办法 A：等批 1、批 4 状态变 COMPLETED（squeue 看不到了）后手动提：
sbatch -J g2i_b8 --export=$COMMON,BATCH=8 slurm/alpha_batch.slurm

# 办法 B（推荐，自动依赖）：记下批 1、批 4 的 jobid，用 afterok 串起来：
JID1=$(sbatch -J g2i_b1 --parsable --export=$COMMON,BATCH=1 slurm/capella.slurm)
JID4=$(sbatch -J g2i_b4 --parsable --export=$COMMON,BATCH=4 slurm/capella.slurm)
sbatch -J g2i_b8 --dependency=afterok:$JID1:$JID4 \
  --export=$COMMON,BATCH=8 slurm/capella.slurm
```

> `-J g2i_bN` 决定日志文件名（`logs/g2i_bN-<jobid>.out`），**别省略**。
> 若 `--export` 太长可读性差，也可把 ★ 变量直接写进 `slurm/alpha_batch.slurm` 顶部，
> 命令就只剩 `sbatch -J g2i_bN --export=ALL,BATCH=N slurm/alpha_batch.slurm`。

### 如果某批 72h 不够 / 被砍了

**先把活儿拆小再提**——把种子对半拆（每次只跑一部分种子）：

```bash
sbatch -J g2i_b4a --export=$COMMON,BATCH=4,SEEDS="42 43" slurm/alpha_batch.slurm
sbatch -J g2i_b4b --export=$COMMON,BATCH=4,SEEDS="44"    slurm/alpha_batch.slurm
```

`OUTPUT_DIR` 不变，不同种子落到不同目录（`<变体>_<数据集>_seed<种子>`），互不覆盖。

> ⚠️ **重要——重跑不会自动跳过已完成的 run。** 编排脚本**没有**"checkpoint 已存在就跳过"
> 的逻辑：默认 `rectified_main.py` **从头训练并覆盖该 run 目录**。所以一批被砍后**别原样重提**，
> 否则已经跑完的 run 会白白重训。两种正确续跑方式：
>
> 1. **缩小范围重提**：只提还没跑完的部分。比如批 4 里 `gene2image` 三个种子已完成、
>    `geneflow` 没跑，就 `--export=$COMMON,BATCH=4` 不行（会连 gene2image 一起重训）——
>    这种情况改用更细的 `run_all.sh` 直调，只点没完成的：
>    ```bash
>    # 在一个 8 卡交互/批作业里，手动只跑缺的那部分：
>    DATASETS="p1" VARIANTS="geneflow" SEEDS="42 43 44" \
>    INCLUDE_CROSS=0 INCLUDE_INTERPRET=0 OUT_ROOT=$OUTPUT_DIR \
>    bash scripts/run_all.sh 8
>    ```
> 2. **断点续训**：给训练加 `--auto_resume`，让**部分训练**的 run 从 `latest_checkpoint.pt`
>    接着练（已训完的 run 仍会从头——所以这招只省"练到一半被砍"的 run，不省"已练完"的）：
>    ```bash
>    sbatch -J g2i_b4 --export=$COMMON,BATCH=4,EXTRA="--use_amp --auto_resume" slurm/alpha_batch.slurm
>    ```
>
> 经验法则：**事前用 `SEEDS`/拆批把每批控制在 72h 内**（批次表已按此设计），就基本不用碰续跑。

### 备选：一把梭（不分批）

若集群空、想一次提全部 63 个 job（同样 8 卡队列，但单个 72h 作业里跑完一切，**风险高**）：

```bash
sbatch -J g2i_train8 --export=$COMMON slurm/alpha_8gpu_train.slurm
```

不推荐——一旦超 72h 被杀，整作业的进度按 checkpoint 保留但需手动续；分批更稳。

---

## A5. 监控与取回结果

```bash
squeue -u $USER                         # 看排队/运行状态
tail -f logs/g2i_b1-*.out               # 跟某批的实时日志
tail -f results/logs/exp_*.log          # 某个具体实验的细日志（看单实验多久）
# 出错先看 .err：
tail -n 100 logs/g2i_b1-*.err
scancel <jobid>                         # 取消某作业
```

**全部 8 批跑完后**，要的就是 `OUTPUT_DIR`（默认 `code/results/`）整个目录：

```
results/
├── summary_main.csv          # 主 + 消融总表（FID/SSIM/PSNR/UNI2h）
├── ablation/summary.csv      # 消融表
├── cross_dataset/summary.csv # 跨数据集（含 degradation_rate）
├── interpret/<ds>/           # RQ4 三子分析
├── <变体>_<ds>_seed<种子>/    # 每个 run（checkpoint + 指标）
└── EXPERIMENTS_CATALOG.md     # 自述：实验→文件映射
```

把整个 `results/` 打包传回作者即可。**若嫌 checkpoint 太大**，可只回传
`*.csv` / `*.json` / `interpret/`（几十 MB），`checkpoints/` 单独留在集群。

失败时**尤其需要 `.err` 和 `results/logs/` 里对应的 `.log`**。谢谢！

---

## A6. 常见问题速查

| 现象 | 原因 / 处理 |
|---|---|
| `required pathway mask file(s) are missing` 后退出 | 计算节点无外网且 `GMT_HALLMARK` 没填 → 见 §A2.1 |
| `import torch` 报 `iJIT_NotifyEvent` | mkl 冲突 → 在 venv 里 `pip install "mkl==2024.0.0"`（`setup_env.sh` 里有注释行可解开） |
| 登录节点 `cuda avail = False` | 正常，登录节点没 GPU；以 GPU 作业里的自检为准 |
| 批 8 报"找不到 gene2image checkpoint" | 批 1 和/或批 4 还没跑完 → 等它们 COMPLETED 再提批 8（或用 `--dependency=afterok`） |
| 某批被 72h 砍掉 | 见 §A4"如果某批 72h 不够"——**不能原样重提**（会重训已完成的 run），按那里说的缩小范围或加 `--auto_resume` |
| UNI2h-FID 显示 N/A | 缺病理专用权重，自动降级，不影响主指标 |

有任何卡住，把对应的 `logs/g2i_bN-*.out` 和 `*.err` 发回即可。

> 说明：本项目用 8 张卡的方式是「**8 个单卡实验并行**」（一卡一个独立实验），不是把单个
> 模型拆到 8 卡（DDP）。代码里有 DDP 分支但未经验证，默认不用。

---
---

# §B 项目 / 方法总览（代跑不需要读）

**研究问题：**
- **RQ1** 可学习结构化通路瓶颈是否优于无结构编码器（GeneFlow）与固定打分（MUPAD 式）？
- **RQ2** 通路收益来自「结构化稀疏机制」还是「真实生物语义」？（randPath 作机制下界）
- **RQ3** 端到端可学习权重是否优于固定 ssGSEA 打分？（PathPrior 受控复现 MUPAD）
- **RQ4** 通路注意力能否给出模型内生、与生物学一致、且因果有效的「通路→形态」可解释映射？

更深文档：研究/实验设计 `docs/idea_report.md`｜逐文件实现指南 `docs/implementation.md`｜开发日志 `docs/dev_log.md`。上游基线说明 `code/GeneFlowREADME.md`。

---

## B1. 快速开始：一条命令跑完整篇论文的全部实验（本地 / 非 Slurm）

> Alpha 集群上请走 §A 的分批流程；这里是单机多卡的等价入口。

```bash
cd code
bash scripts/run_all.sh <最大并行任务数>     # 例：10 张卡 → bash scripts/run_all.sh 10
```

`scripts/run_all.sh` 是**唯一的总编排脚本**，覆盖论文全部四类实验 + 前置 + 结果汇总：

- **第一个参数 = 最大并行任务数**（默认 **10**）。**一个任务占用一张 GPU**（通过 `CUDA_VISIBLE_DEVICES` 绑定）；用 `wait -n` 队列调度——**一个任务结束，下一个立刻补上**，始终保持至多 N 个任务并发。
- GPU 编号默认 `0..N-1`；可用 `GPUS="0 2 3 5"` 指定具体卡（数量即并发数）。
- 阶段化执行：**① 前置**（修图像路径、构通路掩码、构跨数据集对齐掩码，串行做以避免并行任务争抢同一文件）→ **② 训练+评估**（主实验/消融/跨数据集，GPU 队列）→ **③ RQ4 可解释性**（依赖训练产物）→ **④ 汇总成 CSV**。
- 预览全计划而不执行：`DRY_RUN=1 bash scripts/run_all.sh 10`。
- 全部产物落在一个 `results/` 文件夹下，并生成自述目录 `results/EXPERIMENTS_CATALOG.md`（见 §B5）。

**默认计划规模**：63 个训练+评估任务（6 变体 × 3 数据集 × 3 种子 = 54，跨数据集 3 组 × 3 种子 = 9）+ 3 个 RQ4 任务 + 9 个前置（`INCLUDE_REACTOME=1` 再加 3 个任务 + 1 个前置）。

常用环境变量（均有默认值）：

```bash
DATASETS="c1 c2 p1"   VARIANTS="gene2image geneflow randpath pathprior notrans nomask"   SEEDS="42 43 44"
EPOCHS=100  BATCH_SIZE=16  EVAL_BATCH=8  GEN_STEPS=100  WORKERS=4  DB=hallmark
INCLUDE_CROSS=1  INCLUDE_INTERPRET=1  INCLUDE_REACTOME=0  INTERPRET_SEED=42
GPUS=""(默认 0..N-1)  EXTRA="--use_amp"(训练额外参数)  EVAL_EXTRA=""(评估额外参数)
DATA_ROOT=data/processed_data  MASK_DIR=data/pathway_masks  OUT_ROOT=results
```

例：只在 c1 上跑主对照两种子做冒烟 → `DATASETS=c1 VARIANTS="gene2image geneflow" SEEDS="42 43" EPOCHS=5 bash scripts/run_all.sh 2`。

> ⚠️ 显存：`idea_report` 按 H100 80GB 估算（峰值 ~78GB）。若卡显存较小（如 V100 32GB），调小 `BATCH_SIZE` 并保留 `--use_amp`（`EXTRA` 默认已含）。

---

## B2. 环境

conda 环境（PyTorch 2.2.x + cu121；开发用 `Gene2Image` / 服务器 `zw@Gene2Image`）：

```bash
# 已知坑：环境内 mkl 2025 与 torch 2.2 冲突（undefined symbol: iJIT_NotifyEvent）
pip install "mkl==2024.0.0"                 # 修复 torch import
pip install gseapy torchmetrics scikit-image timm einops safetensors
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 期望 True
```

依赖清单见 `code/requirements.txt`（不含 torch/torchvision，需按 cu121 自装）。可选：`cellpose`（仅 segmentation 版 spatial loss 用）；`UNI2-h` / `HE2RNA(sequoia)` 权重（病理专用指标，缺失时自动降级为 N/A，不影响主流程）。

> 下面命令默认 `python` 已指向该环境；也可设 `PY=/path/to/env/python` 并以 `PY=$PY bash scripts/...` 传入。

---

## B3. 数据

三个预处理 Xenium 黑色素瘤样本（从 Zenodo [`records/17429142`](https://zenodo.org/records/17429142) 下载并解压到 `code/data/processed_data/`，步骤见 `DATA_SETUP.md`）：

| 短名 | 目录 | 基因数 G | 用途 |
|------|------|---------|------|
| c1 | `Xenium_V1_hSkin_Melanoma_Base_FFPE` | 282 | 主实验 / 消融 / 跨数据集源 / RQ4 |
| c2 | `Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE` | 382 | 主实验 / 跨数据集 |
| p1 | `Xenium_Prime_Human_Skin_FFPE` | 5006 | 主实验 / 跨数据集目标 / 通路扩展消融 |

每个数据集含 `adata.h5ad`（log1p 归一化表达）+ `cell_patch_256_aux/input/`（256×256×4 的 H&E RGB + DAPI 单细胞图）。

**数据准备由 `run_all.sh` 自动完成**（修图像路径 + 构掩码）。如需手动：

```bash
cd code
# (1) 修复 cell_image_paths.json 内的失效绝对路径（原作者集群路径 → 本地），产出 *_local.json
python scripts/fix_image_paths.py \
  --json data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json \
  --local_root data/processed_data            # c2/p1 各跑一次
# (2) 构造通路掩码（Hallmark ∩ gene_names，去 <3 基因通路；产出 real/rand/none + W_ssgsea）
python scripts/build_pathway_mask.py --adata <adata.h5ad> --prefix c1 --db hallmark --out_dir data/pathway_masks
#     --ssgsea_mode expression(默认,按训练集均表达加权) | equal(1/k)；通路扩展消融加 --db hallmark_reactome
```

产物：`code/data/pathway_masks/{ds}_{db}_{real,rand,none}.npz`（含 `A[P,G]`、`pathway_names`、`gene_names`，real 还含 `W_ssgsea`）。掩码列顺序严格对齐数据集 `gene_names`，训练时 `rectified_main.py` 会逐基因名校验，不匹配即硬报错（防止静默 token 污染）。

---

## B4. 实验总览

| 实验 | 论文 | 内容 | 主要产出（results/ 下） |
|------|------|------|------------------------|
| **主实验** | 2.1 | Gene2Image vs GeneFlow，3 数据集 × 3 种子，每个 run 训练+评估 | `summary_main.csv` |
| **消融** | 2.2 | randPath / PathPrior / noTrans / noMask（+ 可选 Reactome），相对主方法各翻一开关 | `ablation/summary.csv` |
| **跨数据集** | 2.3 | c1→c2 / c2→c1 / c1→p1 迁移，报告泛化退化率 | `cross_dataset/summary.csv` |
| **RQ4 可解释性** | 2.4 | CLS→通路注意力（A 内生性 / B 与 GSEA 一致性 / C 通路干预因果） | `interpret/<ds>/` |

指标：FID↓、SSIM↑、PSNR↑、UNI2-h FID↓（病理语义，缺权重时 N/A）；跨数据集另有 **degradation_rate** =(fid_cross−fid_same)/fid_same↓；RQ4-B 报 top-k 重合率 + Spearman 排序相关。

### 模型变体（三正交开关）

| 变体 | 编码器 CLI | 相对主方法翻转的开关 | 角色 |
|------|-----------|----------------------|------|
| **gene2image**（主方法）| `--encoder_type pathway --pathway_mask {ds}_hallmark_real.npz` | —（满配） | RQ1 |
| **geneflow**（基线）| `--encoder_type rna` | 移除整个通路编码器 | SOTA / 下界 |
| **randpath** | `--pathway_mask {ds}_hallmark_rand.npz` | 真实→随机同密度掩码 | RQ2 机制 |
| **pathprior** | `--pathway_mask {ds}_hallmark_real.npz --no_learnable_pathway` | 可学习→固定 ssGSEA | RQ3 击穿 MUPAD |
| **notrans** | `--pathway_mask {ds}_hallmark_real.npz --no_pathway_transformer` | 去 Pathway Transformer | 通路协同 |
| **nomask** | `--pathway_mask {ds}_hallmark_none.npz` | 稀疏→全连接 | 结构化稀疏 |
| *gene2imageReactome*（可选）| `--pathway_mask {ds}_hallmark_reactome_real.npz` | 通路库 Hallmark→Hallmark+Reactome | 2.2 附加消融 |

维度链：`[B,G] → 通路 token [B,P,48] →(+CLS)→ 细胞嵌入 [B,256] → [B,512]`（硬对齐 UNet，主干零改动）。

---

## B5. 结果与输出布局

跑完后所有产物在一个 `results/` 文件夹下；`run_all.sh` 启动即写出自述目录 `results/EXPERIMENTS_CATALOG.md`。**实验 ↔ 文件对应：**

| 看哪个文件 | 对应实验 | 内容 |
|---|---|---|
| `summary_main.csv` | 2.1 主 + 2.2 消融 | 每 (变体, 数据集) 的 FID/SSIM/PSNR/UNI2h-FID 多种子均值±std |
| `ablation/summary.csv` | 2.2 消融 | 同上 + 每变体翻转的开关 / 目标 RQ 标签 |
| `cross_dataset/summary.csv` | 2.3 跨数据集 | 每 (模型, 设置) 的 fid_cross / fid_same / **degradation_rate** 均值±std |
| `interpret/<ds>/gsea_consistency.json` | 2.4 RQ4-B | top-k 重合 + Spearman（模型主导通路 vs GeneFlow 基因重要性 GSEA） |
| `interpret/<ds>/{attention.csv, A_endogeneity.json}` | 2.4 RQ4-A | CLS→通路注意力、注意力熵、主导通路 |
| `interpret/<ds>/{intervention.csv, C_causal.json}` | 2.4 RQ4-C | 通路干预形态位移、主导/随机特异性比 |

目录树：

```
results/
├── EXPERIMENTS_CATALOG.md                 # 自述：实验→文件映射 + 运行配置
├── summary_main.csv                        # 2.1+2.2 主表
├── ablation/summary.csv                    # 2.2 消融表
├── cross_dataset/                          # 2.3 跨数据集
│   ├── summary.csv                         #   含 degradation_rate
│   └── <src>_to_<tgt>_seed<seed>/
│       ├── checkpoints/best_checkpoint.pt
│       ├── eval_on_<tgt>/evaluation_summary.json   # 跨面板 (fid_cross)
│       └── eval_on_<src>/evaluation_summary.json   # 同面板参考 (fid_same)
├── interpret/<ds>/                         # 2.4 RQ4 三子分析
├── <variant>_<ds>_seed<seed>/              # 2.1/2.2 每个 run
│   ├── checkpoints/best_checkpoint.pt      #   含 model_config（供 eval 重建编码器）
│   ├── training_losses.csv                 #   epoch / train_loss / val_loss
│   ├── evaluation_summary.json             #   FID/SSIM/PSNR/UNI2h
│   └── gene_importance_scores.csv          #   single 模型梯度基因重要性（喂给 RQ4-B）
└── logs/<job>.log                          # 每个任务的 stdout/stderr
```

> 服务器跑完通常只需把整个 `results/` 取回。若不需要权重，可只取 CSV/JSON 与 `interpret/`（几十 MB），`checkpoints/` 体积大可单独保留。

汇总（`run_all.sh` 末尾自动执行；也可手动）：

```bash
python scripts/summarize_results.py --results_root results --out_dir results
```

---

## B6. 单项手动运行（精细控制时用）

```bash
cd code
# 单次训练（已含训练后 gene importance 分析；single）
python rectified/rectified_main.py --model_type single --img_size 256 --img_channels 4 \
  --adata <adata.h5ad> --image_paths <..._local.json> --output_dir results/gene2image_c1_seed42 \
  --encoder_type pathway --pathway_mask data/pathway_masks/c1_hallmark_real.npz \
  --batch_size 16 --epochs 100 --gen_steps 100 --seed 42 --use_amp        # 烟测加 --debug --debug_samples 200 --epochs 1

# 评估（pathway checkpoint 自动从 model_config 重建编码器；eval 划分用同 --seed）
python rectified/rectified_evaluate.py --model_path <run>/checkpoints/best_checkpoint.pt \
  --model_type single --img_size 256 --img_channels 4 \
  --adata <adata.h5ad> --image_paths <..._local.json> --output_dir <run> --seed 42 --gen_steps 100

# 批量主实验+消融 / 跨数据集（也可被 run_all.sh 调度）
bash scripts/run_experiments.sh gene2image c1 42      # 单个（train+eval）；或 ... all
bash scripts/run_cross_dataset.sh c1 p1 42            # 单组；或 ... all

# RQ4 可解释性（仅 single；--geneflow_importance 指向 geneflow run 的 gene_importance_scores.csv）
python analysis/pathway_interpret.py --model_path results/gene2image_c1_seed42/checkpoints/best_checkpoint.pt \
  --adata <adata.h5ad> --image_paths <..._local.json> --out_dir results/interpret/c1 \
  --geneflow_importance results/geneflow_c1_seed42/gene_importance_scores.csv --analysis A B C
```

跨数据集评估走 `--cross_dataset_eval`：在目标面板掩码上重建编码器，并按 (通路, 基因) 名把源域学到的权重移植过去（共享通路语义空间），由 `run_cross_dataset.sh` 自动加上。

---

## B7. 方法概述

编码器（`src/pathway_encoder.py`）三模块：
- **A 掩码嵌入** `PathwayMaskEmbedding`：固定二值掩码 `A[P,G]`（buffer，不训练）决定稀疏连接；每条非零 (通路,基因) 边一个可学习 `d_token=48` 维权重，用 edge-list + `scatter_add` 高效实现 `t_p = Σ W_{p,g}·x_g + b_p`，从不构造稠密 `P×G×D` 张量；`l1_penalty()=‖W‖₁` 做隐式特征选择。PathPrior 则冻结权重并用 `W_ssgsea` 初始化。
- **B 通路 Transformer** `PathwayTransformer`：通路 token 间自注意力（无位置编码，通路无序），建模通路协同。
- **C CLS 聚合**：CLS token 池化为细胞嵌入；其对各通路的注意力即 RQ4 可解释信号。

`PathwaySingleEncoder`（主线，`[B,G]→[B,512]`）/ `PathwayMultiEncoder`（附线，复用 GeneFlow 多头细胞聚合）。L1 经 `compute_l1_penalty`（`rectified_train.py`）统一接入损失。生成主干 `rectified/rectified_flow.py` 与 `src/unet.py` 完全复用 GeneFlow、**未改动**；`baseline/`（扩散对照）亦保持上游不变。

---

## B8. 代码结构

```
code/
├── src/
│   ├── pathway_encoder.py     # [新] 通路编码器：掩码嵌入 + Pathway Transformer + CLS
│   ├── single_model.py        # [改] encoder_type 分支 + l1_penalty()
│   ├── multi_model.py         # [改] 同上（multi 附线）
│   ├── utils.py               # [改] CLI 新增 Pathway Encoder 参数组 + --cross_dataset_eval
│   ├── unet.py / rectified/rectified_flow.py   # [KEEP] GeneFlow 主干，未改动
│   └── ...
├── rectified/
│   ├── rectified_main.py      # [改] 载掩码 + 列名校验 + model_config(含 gene_names) 入 checkpoint
│   ├── rectified_train.py     # [改] L1 解耦 compute_l1_penalty + torch2.2 AMP 兼容
│   ├── rectified_evaluate.py  # [改] 修损坏 import + pathway 重建 + 跨数据集按名移植权重
│   └── rectified_generate.py  # [改] 修损坏 import
├── scripts/
│   ├── run_all.sh             # [新] 总编排：全部实验 + GPU 队列（MAX_PARALLEL，一卡一任务）
│   ├── run_experiments.sh     # [改] 6变体×3数据集×3种子，每 run train+eval
│   ├── run_cross_dataset.sh   # [改] 跨数据集（修 eval --seed）
│   ├── build_pathway_mask.py  # [新] 通路掩码 real/rand/none + W_ssgsea(equal/expression)
│   ├── build_cross_masks.py   # [新] 跨数据集通路名对齐掩码
│   ├── fix_image_paths.py     # [新] cell_image_paths 路径重映射
│   └── summarize_results.py   # [新] 主/消融/跨数据集三表汇总
├── slurm/                     # [新] Alpha Centauri 提交脚本（见 §A）
│   ├── setup_env.sh           #   建 venv + 装依赖
│   ├── smoke_test.slurm       #   30 分钟冒烟
│   ├── alpha_batch.slurm      #   参数化分批（BATCH=1..8）
│   └── alpha_8gpu_train.slurm #   一把梭（不分批，风险高）
├── analysis/
│   └── pathway_interpret.py   # [新] RQ4 三子分析（A 内生 / B GSEA一致性 / C 因果干预）
├── notebooks/                 # [新] 关键步骤可视化
└── data/{processed_data, pathway_masks}
```

---

## B9. 当前状态

代码层面**完整、可运行，并已通过两轮多智能体审查 + 一次需求验收审计**（25/25 项满足）。本地已用单元/集成测试验证：GSEA 一致性、跨面板权重移植（含真实模型前向）、GPU 调度器并发+空卡复用、掩码列名对齐、各脚本参数与产出落点一致等。

**尚未执行**：真实 100 epoch、多种子的完整训练（数据与算力在外部 GPU 服务器）。因此论文的所有性能数字目前为占位，需在服务器实跑 `run_all.sh` 产出后填入。
