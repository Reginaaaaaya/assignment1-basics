"""先接一小段真实语料：训练你自己的 BPE，然后编码 train/valid。"""
import argparse
import json
from pathlib import Path
import sys

# 无需安装包，直接使用当前仓库中的 cs336_basics。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.completed_tokenizer import Tokenizer
from cs336_basics.train_bpe import run_train_bpe
from data_loader import save_token_ids

EOS = "<|endoftext|>"


def read_sample(path, max_chars):
    # 此入口专门用于小样本接入，不要对十几 GB 语料 read() 全量读取。
    with open(path, encoding="utf-8", newline="") as file:
        text = file.read(max_chars)
    if not text:
        raise ValueError("语料不能为空: " + str(path))
    return text


def documents(text):
    """在 EOS 边界切片，既保留特殊标记，也不截断一个普通词。"""
    parts = text.split(EOS)
    for part in parts[:-1]:
        yield part + EOS
    if parts[-1]:
        yield parts[-1]


def prepare(args):
    if args.max_chars <= 0 or args.vocab_size < 257:
        raise ValueError("max-chars 必须大于 0，vocab-size 至少为 257")
    train_text = read_sample(args.train_text, args.max_chars)
    valid_text = read_sample(args.valid_text, args.max_chars)
    if len(set(train_text)) < 2:
        raise ValueError("训练样本过小或缺少有效内容")
    tokenizer_dir = Path(args.tokenizer_dir)
    data_dir = Path(args.data_dir)
    # 不静默覆盖之前生成的 tokenizer 和 token 数据。
    existing = [tokenizer_dir / "vocab.json", tokenizer_dir / "merges.json",
                data_dir / "train.npy", data_dir / "valid.npy"]
    if any(path.exists() for path in existing):
        raise ValueError("目标已有词表或数据；复用它们或换一个输出目录，避免覆盖旧实验")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 训练集的小片段写入文件，直接调用你已经写好的 BPE trainer。
    sample_path = tokenizer_dir / "train_sample.txt"
    with sample_path.open("w", encoding="utf-8", newline="") as file:
        file.write(train_text)
    print("1. 使用你的 run_train_bpe 训练词表……", flush=True)
    vocab, merges = run_train_bpe(str(sample_path), args.vocab_size, [EOS])
    tokenizer = Tokenizer(vocab, merges, [EOS])
    tokenizer.to_files(tokenizer_dir / "vocab.json", tokenizer_dir / "merges.json")
    # 立即重新加载，确保服务器/另一个进程可以恢复相同 tokenizer。
    restored = Tokenizer.from_files(tokenizer_dir / "vocab.json", tokenizer_dir / "merges.json")
    assert restored.fingerprint() == tokenizer.fingerprint()
    tokenizer_id = restored.fingerprint()
    vocab_size = max(restored.itob) + 1

    print("2. 用同一词表编码训练集和验证集……", flush=True)
    counts = {}
    for split, text in [("train", train_text), ("valid", valid_text)]:
        ids = list(restored.encode_iterable(documents(text)))
        if restored.decode(ids) != text:
            raise RuntimeError(split + " 编解码往返检查失败")
        if len(ids) <= 16:
            raise ValueError("样本太短，至少需要超过 16 个 token")
        save_token_ids(ids, data_dir / (split + ".npy"), vocab_size, tokenizer_id)
        counts[split] = len(ids)
        print(f"{split}: {len(text)} 字符 -> {len(ids)} tokens", flush=True)

    info = {"train_source": str(Path(args.train_text).resolve()),
            "valid_source": str(Path(args.valid_text).resolve()),
            "max_chars_per_split": args.max_chars, "vocab_size": vocab_size,
            "tokenizer_sha256": tokenizer_id, "token_counts": counts}
    (data_dir / "preparation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("3. 已接好，可运行：", flush=True)
    print(f"python train_basic.py --data-dir {data_dir} --out runs/my_tokens --steps 50")
    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", default="../data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid-text", default="../data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--tokenizer-dir", default="artifacts/my_bpe")
    parser.add_argument("--data-dir", default="artifacts/my_tokens")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-chars", type=int, default=50000)
    prepare(parser.parse_args())
