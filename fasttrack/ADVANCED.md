# CS336 加速学习版

目标：用主流库跑通小语言模型，并能够解释数据、模型和训练。不是 CS336 合规提交，不承诺原作业测试通过。保留原作业实现供对照。本项目为单机单卡教学基线，不含分布式、KV cache 或指令微调。

## 环境

在本目录中执行。推荐 Python 3.12 或 3.13；服务器先按 PyTorch 官方安装器选择与驱动兼容的 CUDA 版本并安装 torch，再安装其余依赖。不要复制 Mac 的 `.venv` 到 Linux。

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` 是兼容范围，不是完全复现锁文件。训练后保存环境：`python -m pip freeze > runs/environment.txt`（先创建 runs 目录）。

## 本地小规模验证

以下假定本目录位于原仓库的 `fasttrack/` 子目录中；数据路径均相对当前目录。先用小片段，避免全量处理十几 GB 数据。

```sh
python pipeline.py tokenizer --input ../data/TinyStoriesV2-GPT4-train.txt --output artifacts/tokenizer.json --vocab-size 1000 --max-chars 200000
python pipeline.py encode --input ../data/TinyStoriesV2-GPT4-train.txt --tokenizer artifacts/tokenizer.json --output artifacts/train.bin --max-chars 200000
python pipeline.py encode --input ../data/TinyStoriesV2-GPT4-valid.txt --tokenizer artifacts/tokenizer.json --output artifacts/valid.bin --max-chars 50000
python pipeline.py train --train artifacts/train.bin --valid artifacts/valid.bin --tokenizer artifacts/tokenizer.json --out runs/smoke --context 32 --dim 64 --layers 2 --heads 4 --hidden 128 --batch-size 2 --steps 20 --warmup 2 --eval-every 10 --eval-batches 2 --device cpu
python pipeline.py generate --checkpoint runs/smoke/best.pt --tokenizer artifacts/tokenizer.json --max-new-tokens 40
python check_model.py
```

这 20 步只验证运行，不能预期生成流畅故事。Mac 可自行将训练和生成的 device 改为 mps 测试；本地验证优先 CPU。

## 服务器真实实验

先运行 `nvidia-smi`，记录 GPU、显存和驱动；确认 CUDA torch 可用。用训练集训练词表，只用同一个词表编码 train/valid，验证集不参与词表训练。

```sh
python pipeline.py tokenizer --input ../data/TinyStoriesV2-GPT4-train.txt --output artifacts/tokenizer.json --vocab-size 10000
python pipeline.py encode --input ../data/TinyStoriesV2-GPT4-train.txt --tokenizer artifacts/tokenizer.json --output artifacts/train.bin
python pipeline.py encode --input ../data/TinyStoriesV2-GPT4-valid.txt --tokenizer artifacts/tokenizer.json --output artifacts/valid.bin
python pipeline.py train --train artifacts/train.bin --valid artifacts/valid.bin --tokenizer artifacts/tokenizer.json --out runs/tinystories --device cuda --precision bf16 --batch-size 8 --accum 4 --steps 4000 --eval-every 100 --eval-batches 20
python pipeline.py generate --checkpoint runs/tinystories/best.pt --tokenizer artifacts/tokenizer.json --device cuda --prompt 'Once upon a time' --max-new-tokens 200
```

这里的训练 token 预算是 `8 × 4 × 256 × 4000 = 32,768,000`，约为 PDF 建议预算的十分之一，目的是先获得一个可解释的实验。不能承诺达到作业 loss 门槛。BF16 不支持时用 `--precision fp32`；显存不足先减 batch-size，再提高 accum 保持有效批量。不要同时更改多个超参数。

断点恢复：重复原训练命令，额外加 `--resume runs/tinystories/last.pt`。`--steps` 是原计划的总步数，不是额外步数；恢复时关键训练参数必须相同。故障重启会从最近一次评估保存的位置继续。恢复保存模型、优化器、步数、随机数状态；跨硬件、版本或非确定性 GPU 算子仍可能有数值差异。

训练日志在 `metrics.jsonl`，包含训练 loss、验证 loss、困惑度、学习率、裁剪前梯度范数和会话吞吐量。验证用固定随机窗口，属于抽样估计而非完整验证集评估。吞吐量包含本会话评估开销，不含首次环境准备。checkpoint 只加载可信来源。

全量 tokenizer 训练虽流式读取，但 BPE 训练器仍需在内存维护统计量；第一次建议只取 1000 万字符：`--max-chars 10000000`。长文档超过 100 万字符时会分块，极端文档可能改变边界预分词。当前采用 HF 默认 ByteLevel 正则，与 PDF 指定正则及 tie-break 不保证一致。编码保存 little-endian uint32，避免不同词表大小导致 uint16 溢出。

## Git 与服务器

本机在仓库根目录执行：先 `git status` 查看原有修改，只暂存本次的 `fasttrack/`，然后自行 commit/push；服务器 clone/pull 同一分支。不要把 data、环境、token 二进制、权重提交 Git；本目录的 .gitignore 已排除生成物。数据可在服务器重新下载或单独传输。服务器没有原仓库 data 时需先准备数据；代码推送不会搬运这些语料。

尚未在服务器验证 GPU、显存占用或真实训练质量。先完成一轮短实验再决定是否跑 OWT。

## 参考

- PyTorch SDPA：https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch AdamW：https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- HF Tokenizers：https://huggingface.co/docs/tokenizers/en/api/trainers

本文件仅为旧高级参考流程。其模型实现现在位于 reference_model.py，pipeline.py 已指向该文件。基础版请从 README.md 与 LEARNING.md 开始；两版 checkpoint 和数据格式不能混用。
