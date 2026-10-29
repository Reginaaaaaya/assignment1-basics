# 一关一关搭建 Transformer

这次从你已经做过的 tokenizer 往后学。**今天只打开 LEARNING.md 的第 1 关，运行 `python lesson.py 1`。** 不需要读原始 PDF，也不用一次读完整套代码。

## 文件怎么读

| 学习顺序 | 文件 | 只关注什么 |
|---|---|---|
| 1 | lesson.py 的 lesson1、data_loader.py 的 get_batch | 输入与标签为何错一位 |
| 2 | lesson.py 的 lesson2 | Embedding 和输出形状 |
| 3 | model.py 的 Attention、rope | Q/K/V、因果 mask、位置 |
| 4 | model.py 的 FeedForward、Block | FFN、归一化、残差 |
| 5 | model.py 的 LM、lesson.py 的 lesson5 | 模型输出、loss、一次更新 |
| 6 | data_loader.py、prepare_demo_data.py | token 文件、磁盘读取、批次 |
| 7 | train_basic.py | 反复训练与独立验证 |
| 8 | checkpoint.py | 保存与恢复训练 |
| 9 | generate_basic.py | 自回归生成 token ID |

详细说明、预期结果、每关小练习和过关标准都在 [LEARNING.md](LEARNING.md)。每次只推进一关，先自己解释输出，再进入下一关。

## 现在怎么运行

在原仓库根目录：

```sh
cd fasttrack
../.venv/bin/python lesson.py 1
```

你已有的本地环境能够运行基础课程，只依赖 torch 和 numpy，不需要 HF tokenizers。后面文档里的 `python` 指已激活该环境的 Python；也可以一直写 `../.venv/bin/python`。

新环境可运行 `python -m pip install 'torch>=2.6,<3' 'numpy>=2,<3'`。服务器先按设备驱动选好 CUDA 版 PyTorch，别把 Mac 环境复制过去。

## 代码调整了什么

- model.py 改为直观版本：普通配置类，分开的 Q/K/V，显式注意力公式，独立 FFN，中文形状注释。
- 数据、训练、保存、生成分文件。基础训练先用 CPU、固定学习率，不引入 BF16、梯度累积等额外概念。
- 使用已有 tokenizer 输出的 `list[int]` 即可接真实数据，不要求重做 tokenizer。
- 原先的整套入口保留为后续参考：pipeline.py 使用 reference_model.py，原运行说明保存在 ADVANCED.md。现在先不用读它们。

基础版与原参考版的模型权重键名不同，**checkpoint 不能混用**。基础版使用 .npy，原参考版使用 .bin，也不能直接互换。

## 验证与边界

`python check_model.py` 检查基础模型的形状、因果性、RoPE、固定批量学习和权重保存加载。数字演示语料只证明流程可运行，不代表语言能力。服务器性能、显存和真实语言质量仍需实测。显式注意力会生成 T×T 矩阵，基础版优先学习，小上下文使用；大规模训练时再理解并使用参考版的 SDPA。

本项目按你的自学目标使用 torch.nn 等常用库；不是课程合规提交，也不以通过原作业全部 adapters 测试为目标。原先 cs336_basics 中的 tokenizer 代码保持原样。
