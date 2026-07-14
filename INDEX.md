# Gene2Image —— 最新版本（唯一权威目录）

> 整理时间：2026-07-10。本目录 `Gene2Image_最新/` 是把散落在 `/root/ctw` 里的
> Gene2Image 代码与论文的**最新版本**合并到一处的干净副本。以后只认这一份。

## 目录内容
| 子项 | 是什么 | 来源 |
|---|---|---|
| `code/` | 可部署的最终代码（capella 合并版：val_mse 选择、NaN/梯度守卫、DOPRI5 修复、UNI2-h 路径修复、1-GPU job-array slurm） | = `Gene2Image_final/code`（与 `Gene2Image_capella_merged/.../code` 字节一致） |
| `paper/main.tex` | IEEE TMI 手稿，859 行 | = `gene/论文/论文/main.tex`（md5 aca2f63…，与两个 paper 打包一致） |
| `paper/refs.bib` | 参考文献 | 同上 |
| `paper/submission_checklist.md` | 投稿前收尾清单（含 \TBD 回填项） | 同上 |
| `paper/COMPILE_README.txt` | 编译说明 | paper 打包内 |
| `README.md` `DATA_SETUP.md` `env_g2i.sh` `gmt/` `docs/` | 运行/数据/环境说明、pathway gmt、设计文档 | = `Gene2Image_final/` |

- **论文标题（最新）**：*Learnable Pathway Conditioning for Gene-Expression-to-Histopathology
  Generation: A Controlled Study of Mechanism, Semantics, and Learnability*（IEEEtran, journal）。
- **论文状态**：正文六大块 + Discussion/Conclusion 已润色；全文 **36 处 `\TBD`** 待完整多种子训练跑完回填；作者块仍为 Anonymous。**本机无 LaTeX，需在你机器上编译**。
- **代码状态**：单细胞路径（与 GeneFlow 可比，已确认「不动」）；实验尚未跑，量化结果均为占位。

## `/root/ctw` 里其余 gene 相关条目

**已归档（2026-07-10 移入 `../gene_archive/`，可逆，见其 ARCHIVE_README.md）**
所有被本目录取代的冗余项：4 个旧代码目录（`Gene2Image_final/`、`Gene2Image_capella_merged/`、
`Gene2Image_review_v2/`、`Gene2Image_run/`）、5 个冗余打包（含 `Gene2Image_final.tar.gz`、
两个字节相同的 paper tar）、6 个顶层报告 md。

**科学版源头 / 历史（原地保留，未归档）**
- `gene/最终代码/最终代码/Gene2Image-main/` — 合入 capella 前的「科学版最新」（含 logs/pbs、42KB 开发 README），从未推到 GitHub，是 final 的科学祖先。
- `gene/初始代码和文章/.../Gene2Image_github/` — 最初版（含 .git、237 行初稿论文），历史起点。
- `gene/论文/论文/` — 最新论文源（main.tex 与本目录 `paper/main.tex` 字节一致）。

**待推送的 git 补丁（本机无 git 认证，需你/师兄 `git apply` 后 push）**
- `Gene2Image_capella-science.patch`、`Gene2Image_main-science.patch`。

**外部基线 / 输入（保留）**
- `Gene2Image-capella.zip` — 师兄的 capella 分支下载（合并输入）。
- `geneflow/GeneFlow-main.zip` — NeurIPS'25 baseline 代码。

**结果（无效、体积大）**
- `之前代码结果/之前代码结果.gz`（851 MB）— 之前 54-run 结果，已判定**无效**（val_loss 选择不公 + 抢占截断 + UNI2-h FID=NaN），仅留证据用，重跑后可删。

## 下一步（外部，代码不覆盖）
1. 从 **Zenodo 17429142** 下载数据（无自动脚本，手动）。
2. 取 **MahmoodLab/UNI2-h** 门控权重、装 opencv，才能算 biological FID。
3. capella.slurm 上设 `--account`，先跑一个 c1 pilot 标定 walltime / 确认 50 epoch 收敛。
4. 跑完把 `\TBD` 回填、把前瞻语气改成过去时（见 `paper/submission_checklist.md`）。
