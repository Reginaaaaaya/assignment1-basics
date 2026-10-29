"""一次只运行一关：python lesson.py 1。"""
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data_loader import get_batch
from model import Config, Attention, Block, LM


def lesson1():
    data = np.array([10, 20, 30, 40, 50, 60])
    print("原始 ID:", data)
    x, y = get_batch(data, batch_size=1, context_length=3)
    print("输入 x:\n", x)
    print("标签 y:\n", y)
    print("x.shape:", x.shape, "y.shape:", y.shape)
    print("y 的前两项等于 x 的后两项：", torch.equal(y[:, :-1], x[:, 1:]))
    print("任务：手写 data 前四项构成的 x/y；然后把 batch_size 改为 1。")


def lesson2():
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]]) # (2, 3)
    embedding = nn.Embedding(num_embeddings=10, embedding_dim=8) # (2, 3, 8)
    output_head = nn.Linear(in_features=8, out_features=10, bias=False) #(2,3,10)
    vectors = embedding(token_ids)
    logits = output_head(vectors)
    print("IDs:", token_ids.shape, "-> embedding:", vectors.shape, "-> logits:", logits.shape)
    print("这是形状演示，还没有 attention，不能当成完整模型。")
    print("任务：解释为什么 logits 的最后一维是 10，而不是 8。")


def lesson3():
    config = Config(vocab_size=10, context=4, dim=8, heads=2)
    attention = Attention(config)
    x = torch.randn(1, 4, 8)
    before = attention(x)
    changed = x.clone()
    changed[:, 2:] += 100
    after = attention(changed)
    print("输入形状:", x.shape, "输出形状:", before.shape)
    difference = (before[:, :2] - after[:, :2]).abs().max().item()
    print("修改后两个位置，前两个位置输出的最大变化:", difference)
    print("任务：找到 future_mask；解释为什么必须屏蔽右上角。")


def lesson4():
    config = Config(vocab_size=10, context=4, dim=8, heads=2, hidden=16)
    block = Block(config)
    x = torch.randn(2, 4, 8)
    print("Block:", x.shape, "->", block(x).shape)
    print("任务：只读 Block.forward，找出两次残差相加。")


def lesson5():
    config = Config(vocab_size=10, context=4, dim=8, heads=2, hidden=16)
    model = LM(config)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[2, 3, 4], [5, 6, 7]])

    logits = model(x)

    # 计算loss要reshape
    predictions = logits.reshape(-1, config.vocab_size)
    answers = y.reshape(-1)

    loss = F.cross_entropy(predictions, answers)
    print("logits:", logits.shape, "分类题:", predictions.shape, "答案:", answers.shape)
    print("更新前 loss:", loss.item())

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    loss.backward()
    print(model.embedding.weight.grad.shape)
    optimizer.step()

    print("已完成一次参数更新。任务：解释最后三行各自做什么。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=int, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    if args.step == 1:
        lesson1()
    elif args.step == 2:
        lesson2()
    elif args.step == 3:
        lesson3()
    elif args.step == 4:
        lesson4()
    else:
        lesson5()
