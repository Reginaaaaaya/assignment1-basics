# 把你自己的 tokenizer 接到基础训练框架

这一步只做三件事：补齐可用接口、编码少量 TinyStories、确认模型可以读取这些 ID。先不做大词表性能优化，也不期待几十步就生成流畅故事。

## 1. 哪些代码保留了，哪些是补充

**原来的 cs336_basics/tokenizer.py 与 train_bpe.py 完全不改。** 新增 completed_tokenizer.py，其中 Tokenizer 继承你的原类；普通文本仍用 `super().encode(text)` 调用你写的 BPE 合并循环。你原文件中的 pass 仍在，后续接入代码导入的是补全类：

```python
from cs336_basics.completed_tokenizer import Tokenizer
```

这不是另外引入一套 HF tokenizer，也没有更换你的 BPE 算法。使用继承是为了保留你的原文件，便于对照和逐步把补全内容学会。

| 补充 | 为什么需要 |
|---|---|
| 构造函数 | 用 bytes 比较词表，避免重复加入 EOS；不修改传入的词表 |
| encode 的外层 | 特殊 token 先整体识别，普通文本再交给原合并逻辑 |
| encode_iterable | 逐段产出 token ID，不一次读完整个输入迭代器 |
| decode | 合并字节后解码；对单个不完整 UTF-8 token 使用替代字符 |
| to_files/from_files | 保存并恢复词表、合并顺序和特殊 token |
| fingerprint | 自动标识词表版本，避免 checkpoint 与不同 tokenizer 混用 |

父类：你原来的 Tokenizer。子类：补全类。`super().encode(piece)` 表示调用父类方法；本次只需要知道普通文本最终还是走你自己的循环，不必系统学习所有继承语法。

## 2. 先检查补全接口

在仓库 fasttrack 目录执行：

```sh
../.venv/bin/python check_my_tokenizer.py
```

检查中英文往返、特殊 token、重叠标记、迭代器惰性以及词表保存恢复。它使用小词表，不进行大规模 BPE 训练。

如果你的 Python 环境尚未安装 regex，需要先安装它。这里依赖 torch、numpy、regex；不需要 HF tokenizers。不要误用根目录全部 pytest 来判断本补全类：原课程 adapters 仍导入原类，没有被本次修改。

## 3. 用你自己的 BPE 训练一份小词表，再编码

本次接入已在项目里准备好默认目录的小样本，可以直接进入第 4 节训练。下面是复现准备过程的命令；如果目录已经存在，不需要重复运行它。下载补充代码到另一份仓库时，则需要先执行这一步。

```sh
../.venv/bin/python prepare_my_tokens.py
```

默认分别读取 TinyStories train/valid 前 50,000 个字符，目标词表大小 512。步骤如下：

```text
训练集小片段 → 你原来的 run_train_bpe → vocab + merges
                                         ↓
                                  补全版 Tokenizer
                                         ↓
                              保存词表，再从文件加载
                                         ↓
                     同一 tokenizer 编码 train 和 valid
                                         ↓
                          .npy → 现有 get_batch / 模型
```

只有训练集参与 BPE 统计，验证集只编码、不拟合。文件中保留的 EOS 直接作为一个 token。小片段可能截断最后一篇文档，但不会截断 UTF-8 字符。TinyStories 前缀小样本不是随机代表性抽样，当前只做接入验证。

生成文件：

```text
artifacts/my_bpe/
  train_sample.txt       训练词表所用的小文本
  vocab.json             ID 与字节内容、特殊标记
  merges.json            按顺序保存的合并规则

artifacts/my_tokens/
  train.npy / valid.npy
  train.tokenizer.txt / valid.tokenizer.txt
  train.vocab.txt / valid.vocab.txt
  preparation.json       数据来源、样本范围和 token 数量
```

特殊标记、词表和合并规则一起计算 SHA256，作为现有 data_loader 的 tokenizer_id。因而训练 checkpoint 保存的是具体 tokenizer 的版本，不再是手写的 my-bpe-v1。

to_files 使用自有 JSON 格式 `cs336-bytes-hex-v1`，将 bytes 用十六进制字符串存储，包括不能独立解码的字节也能无损保存。from_files 读的是这种格式，不直接兼容 GPT-2 vocab/merges 或 HF tokenizer.json。

如果目录已经有词表或 token 文件，准备脚本会拒绝覆盖。希望做新实验时，换目录，例如 `--tokenizer-dir artifacts/bpe_v2 --data-dir artifacts/tokens_v2`；不要悄悄替换旧 checkpoint 对应的词表。

## 4. 接上现有训练循环

准备结束后运行：

```sh
../.venv/bin/python train_basic.py --data-dir artifacts/my_tokens --out runs/my_tokens --steps 50
```

train_basic.py、data_loader.py 和模型没有因此改写。区别只是 x/y 现在来自你的 tokenizer 编码的真实故事，而不是 0～15 周期数字。

先确认 ID 范围合法、训练与验证都能运行、checkpoint 能保存。50 步用于检查流程；自然语言上的 loss 不一定每次都降，生成也很可能较差。不要把它与数字演示实验的 loss 直接横比。

## 5. 同一词表解码，生成文字

```sh
../.venv/bin/python generate_text.py --prompt 'Once upon a time' --max-new-tokens 40
```

这会加载 artifacts/my_bpe 的词表和 runs/my_tokens/last.pt，核对指纹后，执行：

```text
提示文字 → 你的 encode → 模型逐 token 生成 → 你的 decode → 文字
```

与 generate_basic.py 的区别只在两端加上 tokenizer：现在输入提示文字，输出也是文字。词表不匹配会停止。初期模型可能生成无法组成完整 UTF-8 的字节序列，显示为替代字符；这不是把解码当成训练有效的证明。

## 6. 流式接口的边界要清楚

encode_iterable 会把每个字符串当成独立片段。把一个词或特殊标记任意切成两段，可能得到不同的 token 序列；并不保证 `encode_iterable(chunks)` 永远等于 `encode(''.join(chunks))`。准备脚本在完整 EOS 标记处切文档，避免跨文档合并和特殊标记断裂。

当前准备脚本故意只处理有限字符的小样本：文本和 token list 仍保存在内存。encode_iterable 本身惰性，并不意味着外层 `list(...)` 也不占内存。全量处理 2GB/12GB 语料时需要另加分批写盘，先不要把 --max-chars 调成几十亿。

你原来的 BPE trainer 仍是串行、每轮重新计数的版本，主要按第一个特殊标记分割；本次入口只使用 EOS。极小/退化语料可能耗尽可合并对，空特殊标记列表也不是本次路径。我们没有将它改造成全量大语料高性能训练器。先让小规模闭环可信，再决定是否优化训练器。

## 今天的学习重点

先执行接口检查，查看 preparation.json 中的词表大小与 train/valid token 数；数据未准备时再运行 prepare_my_tokens.py。打开 completed_tokenizer.py 的 encode：看看它如何把普通文本送回你的原实现。之后再跑 50 步模型训练，不需要一次阅读全部新增代码。

## 本次验证结果

- 补全层的小检查通过，包括原编码行为保持不变、特殊 token 重叠、Unicode、惰性迭代、保存加载和指纹。
- 在临时测试副本中，将课程 tokenizer 测试的工厂指向补全类：23 项通过，2 项 Linux 专用内存测试在本地 Mac 跳过。参考 tiktoken 使用仓库本地 GPT-2 词表构建，没有联网下载编码器；原测试和 adapters 未修改。
- 使用你原来的 BPE trainer，50,000 字符训练样本得到 512 项词表；train/valid 分别编码为 20,594 / 21,154 个 token。
- 临时副本的基础模型训练 50 步，固定抽样验证 loss 从第 10 步的约 6.09 降至第 50 步约 5.32，并完成文字生成。结果仅验证接入，生成文字尚不流畅。
- 项目只放入词表和 token 数据，不复制临时测试模型；你可以亲自运行第 4 节的命令观察训练。
