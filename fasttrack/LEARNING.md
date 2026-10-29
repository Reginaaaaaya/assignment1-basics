# 从你的 tokenizer 出发，一关一关搭建语言模型

这份讲义替代你当前阶段反复翻阅原 PDF 的需要，重点是 Transformer 和训练工程。每一关都按“目标 → 最少知识 → 看哪里 → 运行 → 小改动 → 过关标准”安排。**一次只做一关。** 后面的章节是备查材料，不是今天的任务清单。

你原来的 tokenizer 已经有 encode 和 decode，后续只需要它输出整数 ID。from_files、encode_iterable 和特殊 token 行为仍有待完善；先不在这里展开，也不把它们已完全正确当作前提。先用很小的 ID 序列学模型，接真实语料时再核对这些边界。

## 路线图：先看到输入，再认识模型，最后训练

```text
你已有的 tokenizer
       ↓ list[int]
第1关 输入 x 与下一 token 标签 y
第2关 Embedding 与输出分数
第3关 注意力 + 因果 mask + RoPE
第4关 FFN + RMSNorm + 残差 = Transformer Block
第5关 堆叠 Block + loss + 一次参数更新
第6关 Data Loader 与磁盘上的 token 数据
第7关 Training Loop 与验证
第8关 Checkpoint 与断点恢复
第9关 自回归生成
       ↓
之后才考虑真实语料长训练、学习率调度和性能优化
```

术语只记四个：B=batch_size，一次送几段；T=context_length，每段几个 token；D=model_dim，每个 token 用多长的向量表示；V=vocab_size，有多少种 token。H 是注意力头数，每头维度是 D/H。

## 第 1 关：模型到底要吃什么、回答什么

**目标：说清输入 x 和标签 y 的关系。今天只完成这一关。**

假设 tokenizer 输出 `[10,20,30,40,50,60]`。不必管这些数字对应什么文字，它们是词表编号。取前四项，如果 T=3：

```text
输入 x     10   20   30
目标 y     20   30   40
```

含义是：看到 10，预测 20；看到 10、20，预测 30；看到 10、20、30，预测 40。三个位置都提供一个监督信号。一次前向可以并行计算三个预测，但每个预测只允许用到它之前的历史。

为什么取 T+1 项？前三项做输入，后三项做答案。不是把 y 整体喂给模型，也不是把 ID 加 1 当答案：示例数字恰好有规律，真实 ID 没有这种数值规律。

只读 `lesson.py` 的 lesson1，再读 `data_loader.py` 的 get_batch，暂时跳过其余函数。在 fasttrack 目录运行：

```sh
python lesson.py 1
```

你会看到两行输入、两行标签，形状都是 `[2,3]`，以及检查结果 True。起点是随机抽的，因此不同运行设置可能抽到不同片段；正确关系是每行标签都比输入错后一个位置。

代码里的 `window[:-1]` 去掉最后一项，`window[1:]` 去掉第一项。`torch.stack` 把 B 个长度 T 的向量叠成 `[B,T]`。`torch.long` 是 64 位整数类型，适合 embedding 索引和分类标签。

**只做一个小改动：** 把 lesson1 的 batch_size 从 2 改成 1，预测形状再运行。之后改回去。

**过关标准：** 给你 `[7,3,9,2]`，你能写出 T=3 的 x/y，并解释 B 与 T 的区别。能做到再进下一关。

## 第 2 关：ID 如何进入模型，模型如何给候选打分

**目标：理解 `[B,T] → [B,T,D] → [B,T,V]`。**

`nn.Embedding(V,D)` 是一张可训练的 V 行、D 列的表。ID=3 就取第 3 行；重复的 ID 在输入时查到相同向量，进入注意力后才会因上下文和位置而不同。这里不显式创建 one-hot。

`nn.Linear(D,V)` 把每个位置的 D 维向量变成 V 个分数。它对所有位置共用同一组权重，不为每个位置单独造一层。

只读 lesson2，运行：

```sh
python lesson.py 2
```

预期形状：`[2,3] → [2,3,8] → [2,3,10]`。最后的 10 对应词表大小，8 对应内部特征维度。logits 是分数，可以负，也不要求加起来为 1。

**小改动：** 把 embedding_dim 和 output_head 的 in_features 一起改为 12，观察只有中间维度变化。最终还是给 10 种 token 打分。

**过关标准：** 不看代码说出 V 和 D 为什么不是一回事。当前例子只有查表和线性变换，还没有 Transformer，因此不能建立跨位置上下文关系。

## 第 3 关：注意力让当前位置读取历史

**目标：理解 Q/K/V、形状和为什么需要因果 mask。**

先只读 `model.py` 的 Attention。Q、K、V 都来自同一个输入 x，经过三组不同权重的 Linear：Q/K 用来计算位置间的匹配程度，V 提供被汇总的内容。不要把 Q/K/V 当成固定的自然语言字段。

本版刻意分开三个 Linear，避免先学合并 QKV 的优化写法。对于 B=2、T=3、D=8、H=2：

| 计算 | 形状 | 含义 |
|---|---|---|
| x | [2,3,8] | 两段，每段三个位置 |
| query/key/value | [2,3,8] | 三种投影 |
| reshape + transpose | [2,2,3,4] | 每个头单独处理 4 维特征 |
| Q @ K.transpose(-2,-1) | [2,2,3,3] | 每个位置对每个位置的分数 |
| softmax(scores) | [2,2,3,3] | 对可见位置的权重 |
| weights @ V | [2,2,3,4] | 每头汇总后的内容 |
| 拼回各头 | [2,3,8] | 恢复模型维度 |

公式是 `softmax(QKᵀ / sqrt(head_dim) + mask) V`。除以平方根是控制分数尺度，softmax 沿最后一维把同一行变成权重分布。`transpose(-2,-1)` 只交换最后两个轴，前面的 B/H 保留。

### 3.1 为什么屏蔽右上角

一行代表一个正在做预测的位置，一列代表它准备读取的位置：

```text
             可读取列 0 1 2
预测位置 0             ✓ × ×
预测位置 1             ✓ ✓ ×
预测位置 2             ✓ ✓ ✓
```

对角线可以保留，因为当前位置的输入 token 已知；未知的是它的下一个 token。把未来位置分数设为负无穷后，softmax 权重为零。否则位置 0 在预测位置 1 的 token 时能直接读到答案。

运行 `python lesson.py 3`。它修改后两个位置的输入，前两个位置的输出最大变化应为 0。这比只看 loss 更直接地检查有没有偷看未来。

**小改动：** 在 Attention.forward 临时打印 scores.shape，再恢复。不要为了观察效果删掉 mask 跑真实训练。

### 3.2 然后补上 RoPE，不必先背公式

在普通注意力公式中，没有任何变量直接表示第几个位置。RoPE 在 Q/K 做匹配前，按位置旋转每对坐标，让匹配分数能够利用相对位置信息。它只改变 Q/K 的值，不改变形状，也不旋转 V。

`rope` 分三步读：生成每对坐标的频率；用位置乘频率得到角度；对偶数/奇数坐标做二维旋转。`[:,None]` 增加一维方便广播；`0::2` 取偶数坐标，`1::2` 取奇数坐标。第一次只记调用发生在 QKᵀ 之前，后面再逐行读数学。

**过关标准：** 你能解释 scores 的最后两维，以及为什么只能保留左下三角。暂时不会推导 RoPE 也可以进下一关。

## 第 4 关：把注意力装进一个 Transformer Block

**目标：区分注意力、FFN、归一化、残差四件事。**

先只读 Block.forward，再读 FeedForward。一个 block 有两条子路径：

```text
x → RMSNorm → Attention → 与原 x 相加
  → RMSNorm → FeedForward → 与该子路径输入相加
```

RMSNorm 沿最后的 D 维按均方根缩放，再乘可训练尺度；与 LayerNorm 不同，它不先减均值。`eps` 防止除以极小数。pre-norm 就是先归一化再进入子层。

残差是 `x + 子层输出`，不是把两个向量拼接；两者形状必须一致。它让信息和梯度有较直接的路径，并不等于“绕过学习”。

FeedForward 对每个位置分别做非线性加工，所有位置共享权重；它本身不在位置之间交换信息。这里采用 SwiGLU：一条分支用 SiLU 激活后当门控，另一条提供内容，两者逐元素相乘，再降维。形状是 `[B,T,D] → [B,T,hidden] → [B,T,D]`。

运行 `python lesson.py 4`，Block 前后都是 `[2,4,8]`。参数 hidden 改变中间宽度，不改变 block 的输入输出维度。

**小改动：** hidden 从 16 改到 24，确认最终形状不变。

**过关标准：** 在代码中指出两次残差相加，解释“注意力做跨位置交流，FFN 做每位置加工”。

## 第 5 关：完整模型、loss 与一次参数更新

**目标：知道模型输出怎样变成可学习的损失。**

读 LM 的 `__init__` 和 forward。`__init__` 声明要学习的层，forward 规定如何使用它们。不要在 forward 中临时新建 Linear，否则不能以预期方式持续学习固定参数。

`nn.ModuleList` 登记一组 block，使 PyTorch 能找到它们的参数。普通 list 没有同样的自动登记行为。`model.parameters()` 则让优化器拿到这些参数。`super().__init__()` 初始化 nn.Module 的内部机制。

整体是 embedding → 多个 block → final norm → output head。不要在 LM.forward 最后加 softmax：交叉熵直接接受 logits，内部完成稳定的归一化相关运算。

运行 `python lesson.py 5`，logits 是 `[2,3,10]`。交叉熵接受 `[样本数,类别数]` 和 `[样本数]`，所以把 B 和 T 展平成 6：`[6,10]` 配 `[6]`。`reshape(-1,V)` 中 -1 表示让 PyTorch 推算该维度。

对于真实答案概率 p，单个位置的 loss 是 `-log(p)`；最后对所有位置平均。答案不必变成 one-hot，整数 ID 已足够。PPL=exp(平均 loss)，只有 tokenizer 与评估口径相同才适合直接比较。

更新只有三件核心事：

```python
optimizer.zero_grad()  # 清空上轮梯度，PyTorch 默认会累积梯度
loss.backward()       # 计算参数梯度，此时参数还没有更新
optimizer.step()      # 根据梯度真正修改参数
```

AdamW 不会替你完成 forward 或 backward。它维护梯度的一阶、二阶统计，并进行权重衰减。入门先理解更新流程，不必立即手推所有优化器公式。

**小改动：** 打印 embedding.weight.grad.shape，位置放在 backward 之后。过关标准是能解释“计算梯度”和“更新权重”发生在不同位置。

## 第 6 关：Data Loader——如何把磁盘上的 token 变成一批数据

**目标：把第 1 关推广到长数据，不重做 tokenizer。**

data loader 是一种职责，不强制使用 `torch.utils.data.DataLoader` 这个类。对这份作业的一维连续 token 序列，用 get_batch 随机抽起点更直接；以后需要多进程加载、变长 padding、分布式采样时再学 Dataset/DataLoader。

### 6.1 数据保存与磁盘读取

基础版采用 `.npy`，它保存 dtype、形状和数组内容。磁盘用 uint32 节省空间，取出的批次转为 torch.long。`np.load(..., mmap_mode='r')` 按需读取文件，而不是把所有 token 一次性复制到内存。注意：这只解决加载环节；你若先 `read()` 全部文本、encode 成完整 list，预处理仍然会占大量内存。

`get_batch` 做四件事：随机选 B 个起点；每个起点取 T+1 个 ID；切出 x/y；stack 成 `[B,T]` 并搬到模型所在 device。最后合法起点是 N-T-1，因此 randint 的排他上界是 N-T。

### 6.2 先用数字数据验证流程

运行 `python prepare_demo_data.py`。它生成周期数字序列 `0,1,...,15,0,1,...`，保存独立的 train.npy/valid.npy。它们共享人为规律，只用于观察学习和保存，不是语言泛化实验。

### 6.3 接你自己的 tokenizer

在你已有的 tokenizer 对象旁边调用下面接口即可，不用切换成 HF tokenizer：

```python
from data_loader import save_token_ids

# tokenizer 是你已经构造好的对象；这里不重新创建、不重新训练它。
train_ids = tokenizer.encode(train_text)
valid_ids = tokenizer.encode(valid_text)
vocab_size = max(tokenizer.itob.keys()) + 1
save_token_ids(train_ids, "artifacts/my_tokens/train.npy", vocab_size, "my-bpe-v1")
save_token_ids(valid_ids, "artifacts/my_tokens/valid.npy", vocab_size, "my-bpe-v1")
```

这个片段用于小规模接入演示，不是独立脚本；需要你现有的 tokenizer 对象和少量 train_text/valid_text。`itob` 是你现有代码里的 ID→bytes 字典。用 max(ID)+1 是为了覆盖所有索引，即使 ID 不连续也不会少分配 embedding 行。

两个数据集必须用同一份已训练词表和合并规则。`my-bpe-v1` 是你手动维护的版本标识，不会自动证明词表完全一致；改变词表就换版本，别只改名字复用旧数据。后续大规模实验应保存词表并计算真实文件指纹。

save_token_ids 也生成 .vocab.txt/.tokenizer.txt，训练脚本据此核对 train/valid。不要把旧 pipeline 的 .bin 直接改名为 .npy。你的特殊 token 处理尚需验证，接真实语料前应确认 EOS 没有被切碎。

**小改动：** 用 `[3,1,4,1,5,9]` 保存再加载，观察 dtype 和 get_batch 输出。

**过关标准：** 能解释磁盘数据、内存数组、训练 batch 三者分别是什么，以及为什么验证集不能单独训练一份词表。

## 第 7 关：Training Loop——把一次更新重复起来

**目标：看懂 train_basic.py 的五个编号步骤。**

先只读 main 中的 for 循环，不必读文件哈希等保护逻辑。基础版按 PDF 的核心训练结构保留：取 batch → forward → cross entropy → 清梯度/反向/裁剪/AdamW → 定期验证与保存。为了降低认知负担，先固定学习率、单设备、FP32，没有梯度累积。

运行：

```sh
python train_basic.py --steps 30
```

默认模型很小，数据是上一关的演示数据，CPU 即可。每十步打印 train_loss/valid_loss，并保存 metrics.jsonl 和 last.pt。30 步是验证流程，不能作为自然语言训练预算。

### 7.1 梯度裁剪在哪里

在 backward 之后、step 之前。`clip_grad_norm_` 限制所有参数梯度的总范数，修改的是梯度，不是 loss 或参数本身。若 loss/梯度出现非有限数，脚本停止，避免继续保存坏状态。

### 7.2 为什么训练与验证分开

训练批次用于更新权重；验证只检查另一份数据上的预测。`model.eval()` 切换层的行为，`@torch.no_grad()` 关闭梯度记录，两者作用不同。这个模型没有 dropout，但仍保留规范的模式切换。验证结束调用 model.train() 恢复模式。

本版每次验证用固定随机种子的单独 generator，因此抽到相同的一组窗口，便于比较，也不会推进训练采样的随机状态。它是抽样验证，不是扫描全部验证集。train_loss 是刚才一个 batch 的损失，valid_loss 是若干批平均，短期起伏不能直接证明过拟合。

### 7.3 怎么看是否在学习

先用 check_model.py 固定小批量过拟合，再用演示数据验证循环；最后才用真实 train/valid。固定批量能记住，只能说明具有学习能力，不证明数据划分或真实质量正确。

真实数据上训练和验证 loss 长期都不降：检查 ID 范围、标签错位、学习率；训练降而验证升：考虑过拟合或分布差异。不要看到一次上涨就下结论。先一次只改一个设置。

step 在基础版指一次 optimizer.step；T 指单段上下文；处理 token 数近似为 `steps × B × T`。随机有放回采样没有显式 epoch，不能把 30 steps 称为 30 遍语料。

**小改动：** 把 BATCH_SIZE 从 4 改为 2，在新输出目录训练，先预估每步处理 token 数怎样变化。

**过关标准：** 你能口述循环的五步，并知道只有 optimizer.step 真正更新参数。

## 第 8 关：Checkpoint——服务器中断后从哪里继续

**目标：区分“保存模型供推理”和“保存训练以便恢复”。**

打开 checkpoint.py，先看 state 字典：

| 字段 | 为什么保存 |
|---|---|
| model | 模型参数：当前学到了什么 |
| optimizer | AdamW 的动量、二阶统计和参数组状态 |
| step | 已完成的更新次数，避免重头计数 |
| model_config | 重新建立相同形状的模型 |
| training_info | 检查数据、词表版本、batch 和学习率是否匹配 |
| random_state | 恢复 CPU 随机采样进度 |
| cuda_random_state | CUDA 训练时记录 GPU 随机状态 |

state_dict 是一个参数名→张量的字典，不是自动保存所有 Python 代码；构建模型的类仍需保留。只存模型参数足以在结构已知时推理，但恢复 AdamW 训练还需要优化器和步数。本版未使用 NumPy 随机采样、scheduler、scaler，所以无对应状态；以后引入就要一起保存。

### 8.1 保存与加载的顺序

保存发生在本轮 optimizer.step 和验证之后；step 记录已完成数。先写 .tmp，再原子替换 last.pt，降低写一半中断损坏旧 checkpoint 的风险。加载时先建立同结构模型与 optimizer，核对设置，再恢复状态，最后恢复随机数。

本版用 weights_only=True 加载基础数据结构。不要把完整 Python 对象或来源不明的文件随意塞进流程。跨版本/硬件/非确定性 GPU 算子不能保证逐位复现；已验证的精确恢复仅指本地相同环境的 CPU 测试。

### 8.2 自己做一次恢复

完成上一关 30 步后：

```sh
python train_basic.py --steps 50 --resume runs/basic/last.pt
```

这表示从已完成 30 步继续到总共 50 步，即再训练 20 步。基础版固定学习率，所以允许扩展总步数。原高级参考版有学习率计划，要求原目标 steps 保持一致；不要把两种恢复规则混在一起。

检查日志应该追加 40/50，而不是重头出现 10/20。若设置或数据变更，脚本会拒绝恢复；想做新实验请换输出目录并不使用 resume。启动时计算文件哈希会顺序读过 token 文件，但不会全量装进内存，大数据需要等待。

**小改动：** 只打印 checkpoint 的 step 和 model_config，观察它们是普通数据。不要为看内容而打印所有权重。

**过关标准：** 能解释“只存 model.state_dict”为什么不等于完整断点续训。

## 第 9 关：从预测分数变成生成

**目标：区别训练并行预测与推理逐 token 生成。**

读 generate_basic.py 的 generate_ids。输入提示 ID → 模型计算 `[1,T,V]` → 只取最后位置 `[1,V]` → softmax → 抽取一个 ID → 拼回输入 → 重复。前面位置的输出不是没用，而是最后位置才对应当前要生成的下一项。

```sh
python generate_basic.py --prompt-ids 0,1,2 --max-new-tokens 12
```

这沿用数字演示 checkpoint，输出只是 ID。生成文字时，用你自己的 tokenizer.encode(prompt) 得到 prompt_ids，调用 generate_ids 后再 tokenizer.decode(result)。必须使用训练时同一词表；基础 checkpoint 内的 training_info 记录了版本标识，解码前要核对。不要拿数字演示的模型生成故事。

temperature 越低分布通常越尖锐，但不会保证质量；本关不加入 top-p，先看懂原始抽样。EOS ID 可作为停止条件传入；它必须来自相同 tokenizer。模型只保留最后 context 个 ID，本版滑动窗口重新计算位置，没有 KV cache，因此长生成慢。

训练时每个位置都可以得到答案的监督；推理时未来 token 不存在，只能一步一步生成。这是基础语言模型，不会因为能生成就自动成为聊天助手。

**过关标准：** 能解释为什么用 logits[:, -1, :]，以及为什么生成时没有 backward。

## 之后再学：学习率、性能与服务器真实训练

先完成第 1～9 关，再学习这些。它们不影响你先理解基本闭环，但会影响真实训练的稳定与效率。

- **Warmup + cosine：** 学习率先从小增到峰值，再逐渐降低。若使用 scheduler，checkpoint 必须保存其状态；训练总步数也会影响计划。高级参考版已有手动调度。
- **梯度累积：** 多个 microbatch 的 loss 按累积次数缩放，再 backward；最后才更新。有效 batch 为 microbatch 大小×累积次数，不能把 backward 次数当作更新次数。
- **混合精度：** 选择与 GPU 支持匹配的 BF16/FP16，减少内存与计算成本；FP16 常需 scaler。先理解 FP32 流程再上这些。
- **高效 attention：** 基础版显式构造 `[B,H,T,T]`，容易观察但耗内存；后续可使用 SDPA，设备满足条件时可能选择高效实现。不要第一次同时改模型结构、精度、数据格式和优化器。
- **真实数据与预算：** 先少量 TinyStories，保持训练与验证独立、词表相同。记录模型结构、token 数、GPU、时间、loss 和样例。原 PDF 的参考配置为 V=10000、T=256、D=512、hidden=1344、4 层、16 头；这是后续目标，不是入门默认。
- **实验：** 一次只变学习率或 batch 等一个因素；观察验证 loss 与生成。不同 tokenizer 的 PPL 不直接横比，数字演示指标也不能与语言数据相比。

服务器迁移只需要 Git 管理代码与文档，数据/权重另行准备，Mac 的 .venv 不复制到 Linux。基础版学习通过后，可以先在服务器使用小基础模型，实测 CUDA 可用、显存、数十步训练和恢复，再加大。高级参考版 HF tokenizer 的 checkpoint 与你自己 tokenizer 的基础训练路径是两条独立实验线，不能直接混用。

## 查阅用：常见报错

| 现象 | 优先检查 |
|---|---|
| embedding 索引越界 | 最大 ID 是否小于 V，是否拿错词表 |
| Expected Long/Int | 输入 ID 和标签是否 torch.long |
| device 不一致 | model 与 x/y 是否都在同一设备 |
| reshape 维度不匹配 | D/H、head_dim、T 的含义是否写错 |
| loss 低得不正常 | 是否泄漏未来或验证数据参与训练 |
| 恢复设置不匹配 | 是否改了模型、数据、词表版本或优化器设置 |
| 旧参考 checkpoint 加载失败 | 是否把 pipeline/reference_model 的权重用于基础模型 |

**现在停在第 1 关即可。** 你只需要运行一次 lesson.py 1，手写 `[7,3,9,2]` 对应的输入和标签。后面我们围绕你的输出逐步推进，不要求先把整份讲义读完。
