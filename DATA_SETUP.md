# 数据获取与放置说明（跑代码前必读）

本项目需要 3 个 Xenium 人皮肤黑色素瘤样本（C1 / C2 / P1）。数据**已由作者处理好并归档在 Zenodo**，
你自己联网下载即可，无需向作者索取、也无需从 10x 下原始数据自己处理。

全程只需 3 步：① 下数据 ② 下一个通路库文件 ③ 跑自检。之后训练脚本会自动完成其余准备工作。

---

## 第 1 步：下载并解压数据（Zenodo）

数据归档：**Zenodo 记录 17429142** → https://zenodo.org/records/17429142 （开放下载）

下载后解压，使三个样本目录正好落在 `code/data/processed_data/` 下，**目录名必须与下面完全一致**
（训练脚本里写死了这三个名字，写错会找不到数据）：

```
code/data/processed_data/
├── Xenium_V1_hSkin_Melanoma_Base_FFPE/            (C1, ~282 基因, ~107k 细胞)
│   ├── adata.h5ad                                  ← 基因表达（模型输入）
│   └── cell_patch_256_aux/
│       └── input/
│           ├── cell_image_paths.json               ← cell_id → 图像路径 的清单
│           └── cell_images/                         ← 256×256 患处 patch 图像 (*_original.tif，4 通道：RGB+DAPI)
├── Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/    (C2, ~382 基因)
│   └── （结构同上）
└── Xenium_Prime_Human_Skin_FFPE/                  (P1, ~5006 基因)
    └── （结构同上）
```

> ⚠️ 解压后的层级以归档实际为准。**关键是**：每个样本目录下要能找到
> ① `adata.h5ad`，② `cell_patch_256_aux/input/cell_image_paths.json`，
> ③ 该 json 指向的 `.tif` 图像文件（在 `cell_patch_256_aux/input/cell_images/` 下，文件名形如 `*_original.tif`）。
> 这三样齐了即可，子目录别再多套一层。

### 关于 json 里的"集群路径"——不用手动改

`cell_image_paths.json` 里的路径是原作者的集群绝对路径（`/depot/natallah/.../processed_data/...`），
在你的机器上是失效的。**不用动它**：训练入口 `scripts/run_all.sh` 会在 PHASE 0 自动调用
`scripts/fix_image_paths.py`，按 `processed_data/` 之后的部分重映射到你的本地目录，
生成 `cell_image_paths_local.json` 供训练使用。

如果你想提前手动验证重映射是否成功（可选）：

```bash
cd code
python scripts/fix_image_paths.py \
  --json data/processed_data/Xenium_V1_hSkin_Melanoma_Base_FFPE/cell_patch_256_aux/input/cell_image_paths.json \
  --local_root data/processed_data
```

成功的标志是末尾打印 `Paths existing after remap: N/N`（命中数 = 总数，C1 约 106980/106980）。
若出现 `WARNING: some paths still missing`，说明 `.tif` 没解压到位或层级不对，回到第 1 步检查。

---

## 第 2 步：准备通路库 GMT 文件（数据之外唯一要单独下的东西）

构建通路掩码需要 MSigDB 的 Hallmark 基因集文件。GPU 服务器若不能联网，请在能联网的机器先下好再拷过去。

- **必需**：Hallmark → 文件名形如 `h.all.v2023.2.Hs.symbols.gmt`
  下载页：https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H
  （选 "Hallmark gene sets" 的 **symbols** 版 `.gmt`）
- **可选**（仅当要做 P1 的 Reactome 粒度消融，即 `INCLUDE_REACTOME=1` 时才需要）：
  Reactome → `c2.cp.reactome.v2023.2.Hs.symbols.gmt`（同页 C2 → CP:REACTOME，symbols 版）

把文件路径在跑训练时用环境变量传入（见第 4 步）。**不传会在 PHASE 0.5 直接报错中止**，
提示缺哪个掩码——这是有意的 fail-fast，不是 bug。

> 备选：若服务器能联网，可不下 `.gmt`，脚本会通过 `gseapy` 在线从 Enrichr 取库。但服务器通常隔离，
> 推荐还是离线备好 `.gmt`。

---

## 第 3 步：跑数据自检（强烈建议，1 分钟）

正式训练前，确认数据齐、能对上。在 `code/` 目录下执行：

```bash
cd code
python - <<'PY'
import os, json, anndata as ad
ROOT = "data/processed_data"
DS = {
  "c1": "Xenium_V1_hSkin_Melanoma_Base_FFPE",
  "c2": "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE",
  "p1": "Xenium_Prime_Human_Skin_FFPE",
}
ok = True
for tag, d in DS.items():
    base = os.path.join(ROOT, d)
    h5  = os.path.join(base, "adata.h5ad")
    js  = os.path.join(base, "cell_patch_256_aux/input/cell_image_paths.json")
    if not os.path.exists(h5): print(f"[{tag}] ✗ 缺 adata.h5ad: {h5}"); ok=False; continue
    if not os.path.exists(js): print(f"[{tag}] ✗ 缺 cell_image_paths.json: {js}"); ok=False; continue
    a = ad.read_h5ad(h5, backed="r"); G = a.n_vars; N = a.n_obs; a.file.close()
    paths = json.load(open(js))
    # 抽查前 50 条图像路径是否能在本地命中（按 processed_data/ 之后重映射）
    def local(p):
        m = "processed_data/"; i = p.rfind(m)
        return os.path.join(ROOT, p[i+len(m):]) if i>=0 else p
    sample = list(paths.values())[:50]
    hit = sum(os.path.exists(local(p)) for p in sample)
    print(f"[{tag}] ✓ adata: {N} 细胞 × {G} 基因 | 图像清单 {len(paths)} 条 | 抽查 50 条命中 {hit}/50")
    if hit == 0: print(f"      ⚠ 图像一张都没命中，检查 .tif 是否解压到 cell_patch_256_aux/input/cell_images/"); ok=False
print("\n结果:", "全部通过 ✅" if ok else "有问题 ❌（按上面提示修）")
PY
```

期望看到三行都是 `✓`，基因数大致 C1≈282 / C2≈382 / P1≈5006，且抽查命中 50/50。

---

## 第 4 步：开跑

数据和 `.gmt` 就位后，一条命令跑全部实验（`<N>` = 可用 GPU 数，一卡一个独立实验）：

```bash
cd code
GMT_HALLMARK=/绝对路径/h.all.v2023.2.Hs.symbols.gmt \
  bash scripts/run_all.sh <N>
```

脚本会自动：重映射图像路径 → 构建通路掩码(.npz) → 训练+评估6个变体 → RQ4 可解释性 → 汇总成 CSV。
（如需 Reactome 消融，额外加 `INCLUDE_REACTOME=1 GMT_REACTOME=/绝对路径/c2.cp.reactome...gmt`。）

先小规模验证管线能跑通，可加 `DRY_RUN=1` 只打印计划不真跑，或参考 `code/slurm/smoke_test.slurm`。

---

## 常见问题速查

| 现象 | 原因 / 处理 |
|---|---|
| PHASE 0.5 报 "required pathway mask missing" 中止 | 没传 `GMT_HALLMARK`，或服务器无网又没给 `.gmt`。按第 2 步备好并传入。 |
| 训练日志显示 "0 cells with both expression and images" | 图像 `.tif` 没解压到位 / json 路径对不上。跑第 3 步自检，看抽查命中数。 |
| `Unknown dataset` | `processed_data/` 下的样本目录名写错了，必须与第 1 步列出的三个名字逐字一致。 |
| `import torch` 报 `iJIT_NotifyEvent` | MKL 符号冲突，`pip install mkl==2024.0` 修复（与数据无关，环境问题）。 |
