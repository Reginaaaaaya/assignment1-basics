"""第 1 / 6 关：从一维 token 序列中随机取出一批训练样本"""
from pathlib import Path
import numpy as np
import torch


def save_token_ids(token_ids, output_path, vocab_size, tokenizer_id):
    """接你自己的 tokenizer.encode(text) 输出；同一 tokenizer 使用同一标识。"""
    if not tokenizer_id:
        raise ValueError("请给这份 tokenizer 一个固定版本标识")
    token_ids = np.asarray(token_ids)
    if token_ids.ndim != 1 or token_ids.size == 0:
        raise ValueError("token_ids 应当是一维、非空列表或数组")
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise ValueError("token ID 必须是整数")
    if token_ids.min() < 0 or token_ids.max() >= vocab_size:
        raise ValueError("token ID 必须落在 [0, vocab_size) 范围内")
    output_path = Path(output_path)
    if output_path.suffix != ".npy":
        raise ValueError("基础版请使用 .npy 后缀")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, token_ids.astype(np.uint32))
    output_path.with_suffix(".tokenizer.txt").write_text(tokenizer_id, encoding="utf-8")
    output_path.with_suffix(".vocab.txt").write_text(str(vocab_size), encoding="utf-8")


def load_token_ids(path):
    # 按需读取磁盘内容，不一次性把大文件全部放进内存。
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("数据文件必须是一维整数数组")
    return data


def get_batch(data, batch_size, context_length, device="cpu", generator=None):
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size 和 context_length 必须大于 0")
    if len(data) <= context_length:
        raise ValueError("至少需要 context_length + 1 个 token")
    # high 是不包含的上界，每个起点都要能向后取 T+1 个 token。
    starts = torch.randint(0, len(data) - context_length, (batch_size,), generator=generator)
    inputs = []
    targets = []
    for start in starts.tolist():
        window = data[start:start + context_length + 1]
        input_ids = torch.tensor(np.array(window[:-1]), dtype=torch.long)
        target_ids = torch.tensor(np.array(window[1:]), dtype=torch.long)
        inputs.append(input_ids)
        targets.append(target_ids)
    # B 个 [T] 叠成 [B,T]，再搬到模型所在的设备。
    x = torch.stack(inputs).to(device)
    y = torch.stack(targets).to(device)
    return x, y
