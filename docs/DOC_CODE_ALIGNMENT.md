# 文档 ↔ 代码 对齐修正记录（2026-07-10）

> 用 8-agent 工作流把设计文档逐条对实际代码核验,**以代码为准**改了文档。
> 共 29 处不符(5 HIGH / 14 MEDIUM / 10 LOW)已改入 idea_report / implementation / dev_log /
> known_issues / user_requirements。下面是**写论文时必须照代码写**的关键事实清单。

## ★ 写进论文前必看(HIGH,写错=方法与实验对不上)

1. **训练损失/速度场是"正弦路径解析速度",不是线性流匹配的 `x1−x0`。**
   代码 `rectified_flow.py:91`:目标 = `(x1−x0)·(π/2)·cos(π/2·t)`;路径 `x(t)=sin(π/2·t)·x1+(1−sin)·x0`,
   训练时另加 `(1−t)·0.05` 小噪声。**Method 里损失公式必须带 `(π/2)cos(π/2·t)` 因子**,
   否则等于报了另一个模型。(已改 idea_report §3.6)

2. **best-checkpoint / early-stop 选 `val_mse`(纯速度 MSE,不含 L1/空间项),不是复合 `val_loss`。**
   `rectified_train.py:705`(变量名叫 `best_val_loss` 但存的是 val_mse)。这是"科学版"相对旧无效跑的
   核心公平性修复,**论文/rebuttal 可明说**:跨变体用同一纯 MSE 选模,避免 L1 尺度差造成不公。(代码正确,无需改)

3. **PathPrior 的固定权重不是"标量广播成 d_token"**,而是 ssGSEA 标量 × 一个固定种子(1234567)、
   零均值单位范数的 per-edge `d_token` 随机 profile(`W=标量·profile`)。因为 `c·ones` 会被 pre-norm
   LayerNorm 抹平、使 PathPrior 输入无关。**描述 RQ3 基线时按此写**。(已改 implementation §3.1)

4. **PathPrior 的 ssGSEA(expression 模式)权重用"全数据集(含验证集)平均表达",非仅训练集** →
   对 RQ3 固定基线有一个小而对称的 train/val 泄漏(代码 docstring 自认)。**论文写"full-dataset mean
   expression"并可一句话交代该泄漏对称、不偏向 Gene2Image**。(已改 dev_log / implementation §3.2)

5. **RQ4-C 干预形态偏移 = 像素级 L2**(代码 `pathway_interpret.py:396`),UNI2-h 嵌入距离/核形态**未实现**。
   RQ4-C 只跑消融(置零),增强(×3)代码有但默认不跑;基线与干预共享同一初始噪声。**别把 UNI2-h/核形态
   写成现指标**。(已改 idea_report §2.4 / implementation §3.8)

## ★ DOPRI5 欠积分:风险已被代码消除(改变了上次的写作决策)
- 上轮我提示"#5 DOPRI5 欠积分可能污染 RQ2/RQ3"——**现已发现代码早加了守卫**:`rectified_flow.py:364-380`
  步数耗尽时强制补积分到 t=1。故**不再是活跃风险**,论文无需为此加保留声明。(已改 known_issues #5)
- 连带纠正:文档"绝不改动主干"不准确——`rectified_flow.py` 有最小改动(该守卫+FSAL 修复),
  但**速度场/损失未动**,编码器隔离仍成立。主干里 **#3(UNet 时间嵌入 t∈[0,1] 退化)/#7(x_t 随机扰动
  与干净 target 不一致) 仍未修**,若报绝对 FID 需破例或注明。

## 其余对齐要点(MEDIUM/LOW,主要影响 Method/实现描述精度)
- **Pathway Transformer 非 `nn.TransformerEncoder`**:自写 `_CLSAttentionLayer`(MHA+**GELU**)ModuleList+
  final_norm,为导出 CLS 注意力;功能等价 pre-norm/batch_first。(implementation §3.1)
- **通路注意力取"末层、多头平均",非跨层平均**。(idea_report §2.4-A / implementation §3.8)
- **L1 正则**:统一 `compute_l1_penalty(model,l1_weight)`,训练/验证共 **2 处**调用(DDP 在函数内判),
  无 hasattr 兜底(缺 `l1_penalty()` 直接报错);原编码器的 L1 罚的是**首个 nn.Linear(基因→隐层)**,
  非 `encoder[0]`(默认 LayerNorm 增益)。(implementation §3.4)
- **checkpoint 配置键名是 `config`,不是 `model_config`**;含 gene_names/val_mse/pathway 参数。(implementation §5.1)
- **掩码脚本**:`build_real_mask`+`load_pathway_library`(无 `build_mask`);CLI 有 `--prefix`(必填)、
  `--gene_json`(非 `--gene_names_from`)、`--ssgsea_mode/--gmt/--min_genes/--rand_seeds`;输出 `{prefix}_{db}_{variant}.npz`。(implementation §3.2)
- **编码器分支**是"`if pathway … else RNAEncoder`"(rna 是兜底,非显式 elif)。(implementation §3.3)
- **完整训练默认 50 epoch**(脚本 `EPOCHS=50`,对齐 GeneFlow train.sh;GeneFlow 论文=100)。dev_log 旧写"100ep"已改;README 一直是 50,正确。
- `--cross_dataset_eval` 参数已补进 §3.6 列表;PathPrior 由 `--no_learnable_pathway` 触发(main 只响应不设 flag)。

## 已确认与文档一致(无需改,写作可放心引用)
维度链 d_token=48/d_cell=256/out=512、模块 A 的 edge-list `scatter_add`(不建稠密 P×G×D)、
`l1_penalty()=‖W‖₁`、noTrans=通路 token 均值池化、掩码逐基因名硬校验(不匹配即报错)、
跨数据集按(通路,基因)名移植权重、run_all.sh(MAX_PARALLEL=10/一卡一任务/63+3+9 计划/degradation_rate 汇总)、
NaN/isfinite 守卫、grad-clip=1.0、AMP torch2.2 兼容——均属实。
