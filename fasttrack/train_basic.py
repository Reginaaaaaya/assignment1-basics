"""第 7～8 关：先读 main 的训练循环，再读 evaluate/checkpoint。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from checkpoint import load_checkpoint, save_checkpoint
from data_loader import get_batch, load_token_ids
from model import Config, LM


# 基础版：单设备 FP32；学习率采用线性 warmup + cosine + 最低值保持。
BATCH_SIZE = 4
CONTEXT_LENGTH = 16
LEARNING_RATE = 0.001
EVAL_EVERY = 10
EVAL_BATCHES = 5


def get_learning_rate(step, warmup_steps=20, decay_end_step=180):
    """step 从 0 开始；decay_end_step 是余弦结束的位置，不是持续步数。"""
    if step < 0 or warmup_steps < 0 or decay_end_step <= warmup_steps:
        raise ValueError("要求 step >= 0，0 <= warmup_steps < decay_end_step")
    max_lr = LEARNING_RATE
    min_lr = max_lr * 0.1

    # 1. 线性上升：从 0 逐渐升到峰值。warmup_steps=0 时跳过。
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # 2. 到达结束位置后，保持最低学习率。
    if step >= decay_end_step:
        return min_lr

    # 3. 中间阶段：余弦从 1 降到 0，对应 max_lr 降到 min_lr。
    progress = (step - warmup_steps) / (decay_end_step - warmup_steps)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_factor


@torch.no_grad()
def evaluate(model, valid_data, device):
    """不计算梯度、不更新参数，使用固定的验证窗口。"""
    model.eval() # 切到验证模式
    generator = torch.Generator().manual_seed(123)
    total_loss = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(valid_data, BATCH_SIZE, CONTEXT_LENGTH, device, generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
        total_loss += loss.item()
    model.train() # 切回训练模式
    return total_loss / EVAL_BATCHES


def file_hash(path):
    # 分块读取，只在启动时检查一次，用于识别恢复时的数据变化。
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    if args.steps <= 0 or args.warmup_steps < 0 or args.decay_end_step <= args.warmup_steps:
        raise ValueError("steps 必须大于 0，且 0 <= warmup-steps < decay-end-step")
    torch.manual_seed(42)

    train_path = Path(args.data_dir) / "train.npy"
    valid_path = Path(args.data_dir) / "valid.npy"

    train_data = load_token_ids(train_path)
    valid_data = load_token_ids(valid_path)

    train_tokenizer = train_path.with_suffix(".tokenizer.txt").read_text(encoding="utf-8")
    valid_tokenizer = valid_path.with_suffix(".tokenizer.txt").read_text(encoding="utf-8")

    vocab_size = int(train_path.with_suffix(".vocab.txt").read_text())
    valid_vocab_size = int(valid_path.with_suffix(".vocab.txt").read_text())

    if train_tokenizer != valid_tokenizer or vocab_size != valid_vocab_size:
        raise ValueError("train 和 valid 必须由同一份 tokenizer 编码")
    
    config = Config(vocab_size=vocab_size, context=CONTEXT_LENGTH)
    model = LM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    training_info = {
        "train_hash": file_hash(train_path), "valid_hash": file_hash(valid_path),
        "tokenizer": train_tokenizer, "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE, "weight_decay": 0.01,
        # 调度由步数直接计算；保存这份配置即可恢复，无独立 scheduler 状态。
        "lr_schedule": {
            "type": "linear_warmup_cosine_hold_v1",
            "warmup_steps": args.warmup_steps,
            "decay_end_step": args.decay_end_step,
            "min_lr": LEARNING_RATE * 0.1,
        },
    }

    start_step = 0

    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, training_info)
        print("从已完成的步数恢复:", start_step)

    if args.steps <= start_step:
        raise ValueError("steps 是目标总步数，必须大于已完成步数")
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()

    for step in range(start_step, args.steps):
        # 在本次参数更新前，设置优化器使用的学习率。
        learning_rate = get_learning_rate(step, args.warmup_steps, args.decay_end_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        # 1. 取数据：x/y 都是 [B,T]，y 比 x 向后错开一个 token
        x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, args.device)

        # 2. 前向：每个位置得到 V 个候选 token 的分数
        logits = model(x)

        # 3. 把 B*T 个位置展开成 B*T 道分类题，计算损失
        predictions = logits.reshape(-1, vocab_size)
        answers = y.reshape(-1)
        loss = F.cross_entropy(predictions, answers)

        if not torch.isfinite(loss):
            raise RuntimeError("loss 非有限数；检查数据与学习率")
        
        # 4. 清空旧梯度 -> 反向计算新梯度 -> 裁剪 -> 更新参数。
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()

        # 5. 定期验证和保存。step 从 0 开始，completed_steps 从 1 开始。
        completed_steps = step + 1
        if completed_steps % EVAL_EVERY == 0 or completed_steps == args.steps:
            valid_loss = evaluate(model, valid_data, args.device)
            train_loss = loss.item()
            print(f"step={completed_steps} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} lr={learning_rate:.6f}", flush=True)
            row = {"step": completed_steps, "train_loss": train_loss, "valid_loss": valid_loss, "lr": learning_rate}
            with (output_dir / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            save_checkpoint(output_dir / "last.pt", model, optimizer, completed_steps, training_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="artifacts/basic")
    parser.add_argument("--out", default="runs/basic")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--decay-end-step", type=int, default=180,
                        help="余弦下降结束的 step 位置，此后保持最低学习率")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    main(args)
