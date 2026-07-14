# Gene2Image — 投稿前收尾清单

> 状态（截至本轮润色）：正文六大块 + Discussion/Conclusion 已过一遍受控润色，
> 主线一致、claim 不过头、术语统一、可复现细节按真实代码写。
> 全文校验：mechanism/controllability 已清零；MUPAD 全大写一致；
> "attributable" 全带 "primarily"；摘要 246 词（<250）；引用全解析、无孤儿；
> `\begin/\end` 23/23、行内 `$` 342（偶数）。
> **本机无 LaTeX**，PDF 需在你机器上编译。源文件：`main.tex` + `refs.bib`。

---

## A. 结果回填（等完整多种子训练跑完）

- [ ] **填 `\TBD`**：全文 36 处占位（Table III/IV/V 主/消融/跨panel + RQ4 文字），
      从 `results/summary_main.csv`、`results/ablation/summary.csv`、
      `results/cross_dataset/summary.csv`、`results/interpret/<ds>/` 填入。
- [ ] **前瞻语气 → 过去时/完成时**（结果落定后再改，现在改=造结论）：
      - 摘要末句 "is evaluated for …" → "we evaluated / improved …"
      - 主结果 "We use P1 … as the most stringent test" → 实际发现
      - 消融 "these variants jointly test whether …" → 实际排序/结论
      - RQ4 (A/B/C) 的 "we report / we measure whether …" → 实际数值与判断
      - Conclusion 末句 "Whether … is the central question the pipeline is built to answer" → 实际答案
- [ ] **删** Experiments 开头斜体声明 "All quantitative results below are placeholders (\TBD) …"。
- [ ] **删** Limitation **(5) Pending results** 整条（结果出来后不再适用）。
- [ ] `specificity ratio \TBD`、`top-k \TBD`、`Spearman \TBD` 等填数后，判断方向是否成立再措辞。

## B. 图 / 内容补齐

- [ ] **图 2（定性对比）**：从 `results/<variant>_<ds>_seed42/` 出图，替换 `\figplaceholder`。
- [ ] **图 3（注意力热图 + 通路干预）**：从 `results/interpret/<ds>/` 出图，替换 `\figplaceholder`。
      （图 1 架构图已是成品 TikZ，无需替换。）
- [ ] **E（非零边数 / 平均每通路基因数）**：跑 `build_pathway_mask.py` 后拿 C1/C2/P1 的 E，
      补进 Method「Sparse realization」段（现只有 "E ≪ P·G" 与 ~0.36M 参数，缺具体 E）。
- [ ] **作者块**：替换 `Anonymous Author(s)`；补 `\thanks`（资助、数据可用性声明）。
- [ ] `\markboth` running header 与最终标题一致（现："Learnable Pathway Conditioning for Gene-to-Image Generation"）。

## C. 投稿前清理（**必做，别让内部注释进投稿包**）

- [ ] **剥离 `main.tex` 内部注释**：头部 `%% NOTE ON RESULTS …`、`%% FIGURES …` 说明块。
- [ ] **剥离 `refs.bib` 核查注释**：`% Verified …`、`% Metadata web-verified …`、
      `% UPDATE 2026-07 …`、`% metadata … verify … before camera-ready` 等全部删。
- [ ] 全局搜确认无残留：`% TODO` / `% verify` / `% 待核` / `% VERIFY`。
- [ ] （可选）我可以出一版**注释全剥离**的 `refs.bib` 给投稿包，内部这版保留注释。

## D. 引用终核（camera-ready 前对 DOI 补全）

- [ ] **长作者名补全**：`ssgsea`(Barbie)/`reactome`(Gillespie)/`uni`(Chen) 现用 `and others`，对 DOI 补全整名单。
- [ ] **Barbie 2009** 页码 `108–112` 对 DOI 复核。
- [ ] **预印本出处更新（见刊后）**：
      - `mupad` — MUPAD, arXiv:2604.03635（正文用 **MUPAD** 全大写，已对 PDF 正文核实；
        作者 HuggingFace 写 "MuPaD"，但论文正文是 MUPAD，以论文为准）。
      - `pearl` — PEaRL, arXiv:2510.03455。
- [ ] **UNI2-h**：`uni2h`(HF 模型卡) + `uni`(UNI 原文) 双引；若 UNI2-h 日后出独立论文，替换模型卡引用。
- [ ] 已核实无需再查：ssGSEA=Barbie 2009、GSEA=Subramanian 2005、Hallmark=Liberzon 2015、
      Reactome=Gillespie 2022、TOSICA=Chen 2023（TOSICA **是** fixed mask + learnable masked projection，已在正文与 idea_report 更正）。

## E. 编译与格式

- [ ] **完整编译（必带 bibtex）**：`pdflatex main → bibtex main → pdflatex main → pdflatex main`。
- [ ] 检查 `.log`：无 `Undefined reference`、无 `Citation undefined`、留意 `Overfull \hbox`。
- [ ] IEEE TMI 格式核对：摘要一段、无公式/表格、**≤250 词**（现 246 ✓）；页数/图表规范。

## F. 实验 / 代跑侧（可选，但影响结果可信度）

- [x] **DOPRI5 欠积分（known_issues #5）已破例修复**：`rectified_flow.py` 采样循环末加
      「欠积分守卫」，步预算耗尽仍 `t<1` 时用有界固定步补到 `t=1.0`；另修 FSAL 第 7 级
      取错状态致误差估计错误。二者只改推理期数值、不动训练权重、对各变体同等生效
      （已在 `dev_log.md` 2026-07-08 记录破例；论文方法/可复现节已如实披露）。
- [ ] 结果回来后仍 `grep "final t=" results/logs/*.log` 复核：应全部 `1.0000`；若出现
      `WARNING ... step budget exhausted` 集中在弱臂，说明守卫在兜底（可接受，但值得记录）。
- [ ] 交代代跑同学**回传 `results/logs/`**（不只 CSV），否则上一条无法自查。

---

### 一句话
文字层面已收尾；**能不能投，取决于 A（结果）+ B（图）**。结果一回来，按 A/B 填、按 C/D 清理，就是投稿版。
