# 基础训练脚本：线性 warmup、余弦下降与最低学习率保持

train_basic.py 已改为在每次 optimizer.step() 之前更新参数组的 lr。模型结构、tokenizer 与数据读取不变。训练输出和 metrics.jsonl 都增加 lr。

默认设置：最大学习率 0.001，最低为 0.0001，warmup_steps=20，decay_end_step=180。

| step（从 0 开始） | 学习率 | 阶段 |
|---|---|---|
| 0 | 0 | warmup 起点 |
| 10 | 0.0005 | 线性上升 |
| 20 | 0.001 | 峰值 / 余弦起点 |
| 100 | 0.00055 | 余弦中点 |
| 180 及以后 | 0.0001 | 保持最低值 |

遵循从 step=0 开始的公式，因此第一次更新的 lr=0，不改变模型参数；后续逐渐增大。日志里的 step 表示已经完成的更新次数，所以打印 step=10 时对应函数输入 step=9。

`--decay-end-step` 是余弦结束的绝对位置，不是余弦持续多少步；余弦阶段长度为 decay_end_step-warmup_steps。它独立于 --steps，因此可以在保持同一调度的情况下，恢复并扩展训练到最低学习率平台。--warmup-steps 0 表示跳过 warmup。要求 0 <= warmup_steps < decay_end_step。

## 新实验

在 fasttrack 目录执行，不使用旧固定学习率 checkpoint：

```sh
../.venv/bin/python train_basic.py --data-dir artifacts/my_tokens --out runs/lr_schedule --steps 200
```

200 步对应 step=0～199，最后 20 次更新使用最低学习率。比较你先前固定学习率实验的相同步数、验证损失与相同提示词生成。

## 断点恢复

调度是由当前 step 和配置直接计算的，没有单独创建 PyTorch scheduler。checkpoint 保存 warmup、余弦结束位置、最低学习率和调度版本；恢复已完成步数之后，计算下一步的 lr 即可。

```sh
../.venv/bin/python train_basic.py --data-dir artifacts/my_tokens --out runs/lr_schedule --steps 250 --resume runs/lr_schedule/last.pt
```

原 200 步实验已经到达最低学习率，此命令继续到总共 250 步，额外 50 步保持最低值。若训练中断在更早位置，同样恢复原曲线。改变 warmup/decay-end-step 会被 checkpoint 设置核对拒绝。

旧固定学习率 checkpoint 不含本次调度配置，会被拒绝作为新调度实验的断点，防止混淆实验条件；旧权重仍可正常用于 generate_text.py 生成。模型与 optimizer 的保存方式未改变。

本地检查覆盖：warmup/峰值/余弦中点/最低平台、无 warmup、非法参数，以及相同环境 CPU 下分段恢复与连续训练的权重一致性。未验证真实服务器上的训练效果。
