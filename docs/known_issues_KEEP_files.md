# [KEEP] 主干文件内的已知问题（未修复，需你决策）

> 生成：2026-06 代码审查。这三条问题位于 `implementation.md` 标记为 **[KEEP]**
> 的主干文件（`src/unet.py`、`rectified/rectified_flow.py`）内。按"绝不改动主干以
> 隔离编码器贡献、保证 GeneFlow vs Gene2Image 对比公平"的原则，**本轮未修改**。
> 它们**不破坏变体之间的相对对比**（两臂共享同一主干，被同等影响），但会**压低两臂
> 的绝对生成质量 / 让写进论文的 FID 数字偏高**。请决定是否破例修复，或在论文中注明。

## #3 UNet 时间嵌入对 t∈[0,1] 退化
- 位置：`src/unet.py:51` `timestep_embedding(timesteps, dim, max_period=10000)`
- 问题：`max_period=10000` 是为整数扩散步数（0~1000）设计的。flow-matching 的 t∈[0,1]
  时，所有频率分量近似常数 → `emb(0)` 与 `emb(1)` 余弦相似度 ~0.97，网络几乎拿不到
  "走到轨迹哪一步"的信息，压低两臂绝对生成质量。
- 若破例修复：训练与两个采样器一致地把 t 缩放，例如 `timestep_embedding(timesteps*1000, ...)`
  （DiT / flow-matching 惯例）。**三处必须一致**（训练 + Euler + DOPRI5），否则训练/推理失配。
- 影响：两臂绝对 FID 偏高；相对对比不受影响。

## #5 DOPRI5 采样可能欠积分（t 未到 1.0 就退出）
- 位置：`rectified/rectified_flow.py:332` `while t < 1.0 and step_count < num_steps`
- 问题：自适应步长无安全余量。模型拟合差/训练早期/较弱的消融臂误差大 → dt 被缩到极小，
  100 步只积分到 t<1 就因 step 预算耗尽退出，返回**半噪声图**并被 FID/SSIM 打分。
  模型拟合好时 3~7 步即到 t=1（gen_steps=100 形同虚设）。
- 风险点：会让 randPath/PathPrior/noTrans 这些**预期较弱的消融臂** FID 虚高，**污染 RQ2/RQ3 对比**。
  （这条对消融的相对对比是有影响的，是三条里最该考虑破例的。）
- 若破例修复：循环后断言 `t >= 1-eps`，或最后一步强制 `dt = 1.0 - t` 落到 t=1；
  或直接改用固定步 Euler（`EulerSolver` 已存在但从未被实例化）。

## #7 stochastic_noise 与 velocity target 不一致
- 位置：`rectified/rectified_flow.py:85-87`
- 问题：`x_t` 加了 `(1-t)*0.05` 的随机噪声，但回归 target（`velocity`，line 91）是**干净路径**
  的解析导数，不含该噪声项。同一个加噪后的 `x_t` 对应多个 target → 不可约标签噪声。
- 影响：幅度小（≤5%），两臂共享。属真实方法学瑕疵但不破坏相对对比。
- 若破例修复：删除 line 85-87 的 stochastic_noise 注入，让 x_t 落在 target 对应的干净路径上。

## 建议
- 若只想要**干净的相对结论**（变体之间谁更好）：三条都可以不改，但 **#5 建议至少加一个
  "采样结束时 t 是否到 1" 的断言/日志**，以便发现哪些 run 欠积分、必要时剔除。
- 若论文要报**绝对质量数字**：#3 影响最大（建议破例），#5 次之，#7 最小。
- 任何破例修改都应在 `dev_log.md` 记录"偏离 [KEEP] 的理由"，保持可追溯。
